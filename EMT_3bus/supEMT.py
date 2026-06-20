# -*- coding: utf-8 -*-
"""
supEMT — EMT_3bus 패키지의 인프라/유틸 통합 모듈.

기존 분리 파일 (park.py, bus_manager.py, base_component.py)을 한 모듈로 묶은 것.
도메인 컴포넌트(VoltageSource, Generator 등)들이 공통으로 사용하는 기반 코드.

포함:
    1. BusManager           — bus ID → 0-indexed 슬롯 매핑
    2. Park 변환            — abc ↔ dq0  (Kundur 컨벤션, amplitude-invariant 2/3)
    3. BaseComponent        — 모든 EMT 소자의 통일 인터페이스
       + phasor_to_3phase_at — 위상자 → 시간영역 인스턴스 값 헬퍼
"""

import numpy as np


# ============================================================
# 1. BusManager
# ============================================================
class BusManager:
    """모선 ID(임의의 hashable) → 0-indexed 노드 슬롯 매핑.

    - 3상(또는 임의 phases) 시스템에서 각 모선은 phases개의 연속 노드를 차지한다.
    - 14-bus, IEEE 39-bus 등 임의 크기 계통으로 확장 가능.
    - 컴포넌트는 생성 시점에 register()를 호출해 슬롯을 캐싱하고, 이후 stamping
      단계에서는 캐싱된 슬롯을 직접 사용해 재해상하지 않는다.
    """

    def __init__(self, phases=3):
        self.phases = phases
        self._slot_of = {}        # bus_id → 0-indexed 슬롯
        self._next_slot = 0

    # -------- 등록/조회 --------
    def register(self, bus_id):
        """모선을 등록하고 슬롯 번호를 반환. 이미 등록된 경우 기존 슬롯 반환."""
        if bus_id not in self._slot_of:
            self._slot_of[bus_id] = self._next_slot
            self._next_slot += 1
        return self._slot_of[bus_id]

    def slot_of(self, bus_id):
        """등록된 모선의 슬롯 반환 (미등록 시 KeyError)."""
        return self._slot_of[bus_id]

    def node_range(self, bus_id):
        """모선의 phases개 노드 인덱스 (start, end_exclusive)를 반환."""
        s = self._slot_of[bus_id]
        return s * self.phases, (s + 1) * self.phases

    # -------- 메타 정보 --------
    @property
    def num_buses(self):
        return self._next_slot

    @property
    def num_nodes(self):
        return self._next_slot * self.phases

    def all_buses(self):
        return list(self._slot_of.keys())

    def __repr__(self):
        return f"BusManager(buses={self.all_buses()}, phases={self.phases})"


# ============================================================
# 2. Park 변환 (abc ↔ dq0)
# ============================================================
# Park 변환 — Kundur (3.131, 3.137) / PSS/E 컨벤션, amplitude-invariant 2/3 scaling.
#
# θ: 회전자 d-축이 고정된 a-상 기준축에 대해 진행한 전기각도 [rad].
#    동기기 EMT에서는 보통 θ(t) = ω_0·t + δ(t)
#    (ω_0: 기저 각주파수, δ: 동기 기준 좌표계에서의 회전자 각).
#
# 3상 평형 정현파 입력의 경우:
#     v_a(t) = V_m·cos(ω_s·t + α),  θ(t) = ω_s·t
# 일 때:
#     v_d = V_m·cos(α),  v_q = V_m·sin(α),  v_0 = 0
# 즉 d-q 좌표에서는 DC 일정값.  V_dq = v_d + j·v_q = V̄_a (peak phasor).

_PI23 = 2.0 * np.pi / 3.0


def abc_to_dq0(v_abc, theta):
    """Forward Park: 3상 abc → dq0 (amplitude-invariant 2/3 scaling).

    Args:
        v_abc: ndarray shape (3,) — [v_a, v_b, v_c]
        theta: float — 회전자 d-축 전기각도 [rad]

    Returns:
        ndarray shape (3,) — [v_d, v_q, v_0]
    """
    a, b, c = v_abc[0], v_abc[1], v_abc[2]
    v_d = (2.0/3.0) * (
        a * np.cos(theta)
        + b * np.cos(theta - _PI23)
        + c * np.cos(theta + _PI23)
    )
    v_q = -(2.0/3.0) * (
        a * np.sin(theta)
        + b * np.sin(theta - _PI23)
        + c * np.sin(theta + _PI23)
    )
    v_0 = (a + b + c) / 3.0
    return np.array([v_d, v_q, v_0])


def dq0_to_abc(v_dq0, theta):
    """Inverse Park: dq0 → 3상 abc.

    Args:
        v_dq0: ndarray shape (3,) — [v_d, v_q, v_0]
        theta: float — 회전자 d-축 전기각도 [rad]

    Returns:
        ndarray shape (3,) — [v_a, v_b, v_c]
    """
    d, q, z = v_dq0[0], v_dq0[1], v_dq0[2]
    v_a = d * np.cos(theta)         - q * np.sin(theta)         + z
    v_b = d * np.cos(theta - _PI23) - q * np.sin(theta - _PI23) + z
    v_c = d * np.cos(theta + _PI23) - q * np.sin(theta + _PI23) + z
    return np.array([v_a, v_b, v_c])


# ============================================================
# 3. BaseComponent + 위상자 helper
# ============================================================
# 모든 EMT 소자의 통일된 기본 클래스.
# 단순 인터페이스 정의만 포함하며, 실제 소자 클래스(Generator, Load, Line 등)에서
# 상속하여 구현한다. 각 소자가 자신의 슬롯 인덱스를 알고 있으므로 메서드는 전체
# G/I/V 벡터를 받아 in-place로 stamping/갱신한다.
#
# == 시간 영역 (사다리꼴 EMT) ==
#     stamp_G(G_matrix):              G 행렬에 등가 전도도 누적
#     stamp_history_current(I_total): 우변 I 벡터에 이력 전류원 주입
#     update_branch(V_nodes):         풀린 V_nodes로부터 분기 전류/전압 이력 갱신
#     predict_state(dt, t):           시간 의존 / 비선형 소자(전압원, 발전기 등)의 예측
#
# == 위상자 정상상태 (bumpless start) ==
#     stamp_Y_phasor(Y, ω):           60 Hz 등가 복소 admittance를 Y에 누적
#     stamp_I_phasor(I_phasor, ω):    노턴 전류원 위상자를 I_phasor에 누적 (전원성 소자)
#     initialize_states(V_phasor, ω, dt):
#                                     위상자로부터 모든 내부 상태(분기 전류, v_C 등)를
#                                     t=-Δt 시점의 인스턴스 값으로 세팅


def phasor_to_3phase_at(V_phasor, omega, t, n_phases=3):
    """단일 위상자(peak amplitude, complex) → 시각 t의 n-phase 인스턴스 값.

    수식:
        v_ph(t) = Re{ V_phasor · exp(j(ωt + θ_ph)) }
        θ_a = 0,  θ_b = -2π/3,  θ_c = +2π/3   (정상상회전)

    Args:
        V_phasor: 단일 복소수 (peak amplitude)
        omega:    각주파수 [rad/s]
        t:        평가 시각 [s]
        n_phases: 상 수 (기본 3)

    Returns:
        ndarray of shape (n_phases,)
    """
    full_offsets = np.array([0.0, -2.0 * np.pi / 3.0, +2.0 * np.pi / 3.0])
    offsets = full_offsets[:n_phases]
    rotation = np.exp(1j * (omega * t + offsets))
    return (V_phasor * rotation).real


class BaseComponent:
    # ---- 시간 영역 ----
    def stamp_G(self, G_matrix):
        pass

    def stamp_history_current(self, I_total):
        pass

    def update_branch(self, V_nodes):
        pass

    def predict_state(self, dt, t):
        pass

    # ---- 위상자 정상상태 (bumpless start) ----
    def stamp_Y_phasor(self, Y_matrix, omega):
        pass

    def stamp_I_phasor(self, I_vector, omega):
        pass

    def initialize_states(self, V_phasor, omega, dt):
        pass

    def update_phasor_voltage(self, V_phasor):
        pass
