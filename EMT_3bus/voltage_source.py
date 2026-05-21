# -*- coding: utf-8 -*-
"""
VoltageSource: 이상 전압원 + 작은 내부 저항 (Norton 등가 stamping).

용도:
    동기기 모델을 도입하기 전 단계의 "검증대(test bench)" 컴포넌트.
    네트워크(라인·부하)가 60 Hz 사인파에 대해 제대로 풀리는지를 단순 산술 함수로
    검증하기 위해 사용한다.

3상 전압 (peak [p.u.]):
    V_a(t) = V_mag · cos(ω t + φ)
    V_b(t) = V_mag · cos(ω t + φ - 2π/3)
    V_c(t) = V_mag · cos(ω t + φ + 2π/3)

이산화 없음 (미분 방정식 미보유, 매 스텝 t를 대입해 함수값 샘플링).
Thevenin → Norton 변환만 수행:
    G_int    = 1 / R_internal       (모선 대각에 stamping)
    I_norton = V_phase(t) / R_int   (매 스텝 우변 I_total에 +로 주입, 모선으로 전류 유입)
"""

import numpy as np

from .supEMT import BaseComponent


class VoltageSource(BaseComponent):
    def __init__(self, bus, bus_manager,
                 V_mag=1.0, V_angle_deg=0.0, f_sys=60.0, R_internal=0.001):
        """
        Args:
            bus:          연결될 모선 ID (BusManager에 등록됨)
            bus_manager:  공유 BusManager
            V_mag:        peak 전압 [p.u.]  (RMS=1.0이면 V_mag=√2)
            V_angle_deg:  a상 위상각 [deg]
            f_sys:        계통 주파수 [Hz]
            R_internal:   내부 직렬 저항 [p.u.]  (작을수록 stiff한 이상 전원)
        """
        self.bus = bus
        self._slot = bus_manager.register(bus)
        self._phases = bus_manager.phases

        self.V_mag = V_mag
        self.V_angle = V_angle_deg * np.pi / 180.0
        self.omega = 2.0 * np.pi * f_sys
        self.R_internal = R_internal
        self.G_int = 1.0 / R_internal

        # 3상 위상 오프셋: a=0, b=-120°, c=+120° (정상상회전)
        n = self._phases
        full_offsets = np.array([0.0, -2.0 * np.pi / 3.0, +2.0 * np.pi / 3.0])
        self._phase_offsets = full_offsets[:n]

        self._t = 0.0   # 현재 시각 (predict_state에서 갱신)

    # ------------------------------------------------------------
    # 시간 정보 갱신 (이산화 아님, 단지 함수값 평가용 t 캐싱)
    # ------------------------------------------------------------
    def predict_state(self, dt, t):
        self._t = t

    # ------------------------------------------------------------
    # G 행렬 stamping (shunt: 내부 저항을 모선 대각에)
    # ------------------------------------------------------------
    def stamp_G(self, G_matrix):
        n = self._phases
        i0 = self._slot * n
        for ph in range(n):
            G_matrix[i0 + ph, i0 + ph] += self.G_int

    # ------------------------------------------------------------
    # 노턴 전류원 주입: I_norton = V_phase(t) / R_internal
    # 부호 규약: 능동 전원이 모선으로 전류 유입 → I_total[bus]에 + 누적
    # ------------------------------------------------------------
    def stamp_history_current(self, I_total):
        v_phase = self.V_mag * np.cos(
            self.omega * self._t + self.V_angle + self._phase_offsets
        )
        i_norton = v_phase * self.G_int
        n = self._phases
        i0 = self._slot * n
        I_total[i0:i0+n] += i_norton

    # 메모리/이력 없음 → update_branch는 no-op (BaseComponent 상속 그대로 사용)

    # ============================================================
    # 위상자 정상상태 (bumpless start)
    # ============================================================
    def stamp_Y_phasor(self, Y_matrix, omega):
        """모선 대각에 G_int = 1/R_internal (실수)를 누적."""
        Y_matrix[self._slot, self._slot] += self.G_int

    def stamp_I_phasor(self, I_vector, omega):
        """노턴 전류원 위상자: I_norton = V_source_phasor / R_internal (peak amplitude)."""
        V_p = self.V_mag * np.exp(1j * self.V_angle)   # a상 위상자 (peak)
        I_vector[self._slot] += V_p * self.G_int

    def initialize_states(self, V_phasor, omega, dt):
        # 내부 상태 없음 → no-op
        pass
