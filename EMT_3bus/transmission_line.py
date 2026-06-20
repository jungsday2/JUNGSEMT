# -*- coding: utf-8 -*-


import numpy as np

from .supEMT import BaseComponent, phasor_to_3phase_at


# ============================================================
# 기본 3-버스 송전선 데이터 (60 Hz, p.u.)
# ============================================================
F_SYS = 60.0
W_S = 2.0 * np.pi * F_SYS

# pu 단위이다.
LINE_DATA_3BUS = {
    (1, 2): {'R': 0.010, 'L': 0.05 / W_S, 'C': 0.10 / W_S},
    (2, 3): {'R': 0.020, 'L': 0.06 / W_S, 'C': 0.12 / W_S},
    (3, 1): {'R': 0.015, 'L': 0.04 / W_S, 'C': 0.08 / W_S},
}


class TransmissionLine(BaseComponent):
    def __init__(self, bus_from, bus_to, R, L, C, dt, bus_manager,
                 damping_L=0.15, damping_C=0.75):
       
        self.bus_from = bus_from
        self.bus_to = bus_to

        # 라인 활성 상태 (False면 트립 = 모든 stamping no-op)
        self.active = True

        # BusManager에 등록 → 0-indexed 슬롯 캐싱
        self._from_slot = bus_manager.register(bus_from)
        self._to_slot = bus_manager.register(bus_to)
        self._phases = bus_manager.phases

        self.R, self.L, self.C, self.dt = R, L, C, dt
        self.damping_L = damping_L
        self.damping_C = damping_C

        # ---- R-L 직렬 : R-(L‖Rp) ParaEMT 방식 ----
        alpha_L = dt / (2.0 * L)
        beta_L  = damping_L * alpha_L        # = 1/Rp
        Delta_L = 1.0 + R * (alpha_L + beta_L)
        self.G_series = (alpha_L + beta_L) / Delta_L
        self.icf_L    = (1.0 - R * (alpha_L - beta_L)) / Delta_L
        self.Gv1_L    = (alpha_L - beta_L) / Delta_L

        # ---- R-C 직렬 션트 사다리꼴 계수 (R_dC=0이면 순수 C와 동일) ----
        # G_shunt = 1 / (R_dC + Δt/C), R_dC = damping_C · Δt/C
        # → G_shunt = 1 / ((1 + damping_C) · Δt/C) = C / ((1 + damping_C) · Δt)
        self._has_shunt = C > 0.0
        self.R_dC = damping_C * dt / C if (damping_C > 0 and C > 0.0) else 0.0
        self.G_shunt = C / ((1.0 + damping_C) * dt) if C > 0.0 else 0.0
        self.dt_over_C = dt / C if C > 0.0 else 0.0

        # ---- 분기 전류 / 단자 전압 이력 ----
        n = self._phases
        self.i_series_prev      = np.zeros(n)   # 전체 직렬 분기 전류 (i_L + i_Rp)
        self.i_shunt_from_prev  = np.zeros(n)   # 션트 R-C 직렬 분기 전류
        self.i_shunt_to_prev    = np.zeros(n)
        self.v_from_prev        = np.zeros(n)   # 모선 전압 (R-L 이력용)
        self.v_to_prev          = np.zeros(n)
        self.v_C_from_prev      = np.zeros(n)   # 콘덴서 단자 전압 (R-C 이력용)
        self.v_C_to_prev        = np.zeros(n)

    # ------------------------------------------------------------
    # G 행렬 stamping (Modified Nodal Analysis)
    # ------------------------------------------------------------
    def stamp_G(self, G_matrix):
        """3상 평형 가정으로 본 라인의 등가 전도도를 G 행렬에 누적한다 (in-place)."""
        if not self.active:
            return
        i0 = self._from_slot * self._phases
        j0 = self._to_slot * self._phases

        for ph in range(self._phases):
            i, j = i0 + ph, j0 + ph
            # 직렬 (R-L) ‖ R_dL: G_series는 댐핑까지 반영된 effective conductance
            G_matrix[i, i] += self.G_series
            G_matrix[j, j] += self.G_series
            G_matrix[i, j] -= self.G_series
            G_matrix[j, i] -= self.G_series
            # 양단 R_dC + C/2 직렬 (대지)
            G_matrix[i, i] += self.G_shunt
            G_matrix[j, j] += self.G_shunt

    # ------------------------------------------------------------
    # 이력 전류원 → 우변 I_total 주입
    # ------------------------------------------------------------
    def stamp_history_current(self, I_total):
        """이번 스텝의 우변 I_total에 본 라인의 이력 전류 기여를 누적한다 (in-place).

        I_hist_RL은 G_pure 기준 (댐핑 저항은 이력 없음).
        """
        if not self.active:
            return
        v_diff_prev = self.v_from_prev - self.v_to_prev
        I_hist_series = self.icf_L * self.i_series_prev + self.Gv1_L * v_diff_prev

        I_hist_shunt_from = -self.G_shunt * (
            self.v_C_from_prev + self.dt_over_C * self.i_shunt_from_prev
        )
        I_hist_shunt_to = -self.G_shunt * (
            self.v_C_to_prev + self.dt_over_C * self.i_shunt_to_prev
        )

        n = self._phases
        i0 = self._from_slot * n
        j0 = self._to_slot * n
        # 직렬: from 측 -, to 측 + / 션트: 모선에서 -
        I_total[i0:i0+n] -= I_hist_series + I_hist_shunt_from
        I_total[j0:j0+n] += I_hist_series - I_hist_shunt_to

    # ------------------------------------------------------------
    # 분기 전류 / 콘덴서 단자 전압 / 이력 갱신
    # ------------------------------------------------------------
    def update_branch(self, V_nodes):
        """이번 스텝에서 풀린 모선 전압으로 분기 전류와 v_C 이력을 갱신한다."""
        if not self.active:
            return
        n = self._phases
        i0 = self._from_slot * n
        j0 = self._to_slot * n
        V_from = V_nodes[i0:i0+n]
        V_to   = V_nodes[j0:j0+n]

        # 직렬 분기 전체 전류 (i_L + i_Rp)
        v_diff_prev   = self.v_from_prev - self.v_to_prev
        I_hist_series = self.icf_L * self.i_series_prev + self.Gv1_L * v_diff_prev
        i_series_curr = self.G_series * (V_from - V_to) + I_hist_series

        # 션트 R-C 직렬 분기 전류
        I_hist_shunt_from = -self.G_shunt * (
            self.v_C_from_prev + self.dt_over_C * self.i_shunt_from_prev
        )
        I_hist_shunt_to = -self.G_shunt * (
            self.v_C_to_prev + self.dt_over_C * self.i_shunt_to_prev
        )
        i_shunt_from_curr = self.G_shunt * V_from + I_hist_shunt_from
        i_shunt_to_curr   = self.G_shunt * V_to   + I_hist_shunt_to

        # 콘덴서 단자 전압 갱신: v_C(t) = v_C(t-Δt) + (Δt/C)·(i(t) + i(t-Δt))
        v_C_from_curr = self.v_C_from_prev + self.dt_over_C * (
            i_shunt_from_curr + self.i_shunt_from_prev
        )
        v_C_to_curr = self.v_C_to_prev + self.dt_over_C * (
            i_shunt_to_curr + self.i_shunt_to_prev
        )

        # 메모리 갱신
        self.i_series_prev     = i_series_curr
        self.i_shunt_from_prev = i_shunt_from_curr
        self.i_shunt_to_prev   = i_shunt_to_curr
        self.v_from_prev       = V_from.copy()
        self.v_to_prev         = V_to.copy()
        self.v_C_from_prev     = v_C_from_curr
        self.v_C_to_prev       = v_C_to_curr

    # ============================================================
    # 위상자 정상상태 (bumpless start)
    # ============================================================
    def stamp_Y_phasor(self, Y_matrix, omega):
        """60 Hz에서의 등가 복소 admittance를 Y에 누적.

        직렬 가지 : Y_RL_phasor = 1/(R + jωL) + 1/R_dL  (R_dL ‖ R-L)
        션트 가지 : Y_shunt_phasor = 1 / (R_dC + 2/(jωC))  (R_dC + C/2 직렬)
        """
        if not self.active:
            return
        # R-(L‖Rp) : Rp >> ωL 이므로 위상자는 1/(R+jωL) 로 근사 (오차 < 0.2%)
        Y_series = 1.0 / (self.R + 1j * omega * self.L)
        if self._has_shunt:
            Z_C_half = 2.0 / (1j * omega * self.C)
            Y_shunt = 1.0 / (self.R_dC + Z_C_half)
        else:
            Y_shunt = 0.0

        f, t = self._from_slot, self._to_slot
        Y_matrix[f, f] += Y_series + Y_shunt
        Y_matrix[t, t] += Y_series + Y_shunt
        Y_matrix[f, t] -= Y_series
        Y_matrix[t, f] -= Y_series

    def initialize_states(self, V_phasor, omega, dt):
        """위상자 V_phasor로부터 모든 분기 전류·전압 이력을 t=-Δt 시점 인스턴스 값으로 세팅."""
        if not self.active:
            return
        n = self._phases
        V_from_p = V_phasor[self._from_slot]
        V_to_p   = V_phasor[self._to_slot]
        V_diff_p = V_from_p - V_to_p

        # 직렬 분기 위상자 i_total ≈ i_RL = V_diff/(R+jωL)  (Rp >> ωL)
        Z_RL = self.R + 1j * omega * self.L
        I_RL_p = V_diff_p / Z_RL

        # 션트 R-C 직렬 분기 위상자 (C=0이면 shunt 없음)
        if self._has_shunt:
            Z_C_half = 2.0 / (1j * omega * self.C)
            Z_shunt = self.R_dC + Z_C_half
            I_shunt_from_p = V_from_p / Z_shunt
            I_shunt_to_p   = V_to_p   / Z_shunt
            V_C_from_p = V_from_p - self.R_dC * I_shunt_from_p
            V_C_to_p   = V_to_p   - self.R_dC * I_shunt_to_p
        else:
            I_shunt_from_p = 0.0 + 0.0j
            I_shunt_to_p   = 0.0 + 0.0j
            V_C_from_p     = 0.0 + 0.0j
            V_C_to_p       = 0.0 + 0.0j

        # 위상자 → t=-Δt 인스턴스 값 (3상)
        t_prev = -dt
        self.v_from_prev       = phasor_to_3phase_at(V_from_p,       omega, t_prev, n)
        self.v_to_prev         = phasor_to_3phase_at(V_to_p,         omega, t_prev, n)
        self.i_series_prev     = phasor_to_3phase_at(I_RL_p,         omega, t_prev, n)
        self.i_shunt_from_prev = phasor_to_3phase_at(I_shunt_from_p, omega, t_prev, n)
        self.i_shunt_to_prev   = phasor_to_3phase_at(I_shunt_to_p,   omega, t_prev, n)
        self.v_C_from_prev     = phasor_to_3phase_at(V_C_from_p,     omega, t_prev, n)
        self.v_C_to_prev       = phasor_to_3phase_at(V_C_to_p,       omega, t_prev, n)


# ============================================================
# Factory: 3-버스 데이터 사전 → TransmissionLine 객체 리스트
# ============================================================
def build_3bus_lines(dt, bus_manager, line_data=None,
                     damping_L=0.0, damping_C=0.0):
    """LINE_DATA_3BUS의 1-indexed 사전을 받아 TransmissionLine 리스트를 만든다.

    Args:
        dt:          시뮬레이션 시간 스텝 [s]
        bus_manager: 모선 슬롯을 관리하는 BusManager
        line_data:   사용자 정의 라인 데이터 (None이면 LINE_DATA_3BUS 사용)
        damping_L:   인덕터 병렬 댐핑 계수 (모든 라인에 동일 적용, 0=꺼짐)
        damping_C:   콘덴서 직렬 댐핑 계수 (모든 라인에 동일 적용, 0=꺼짐)

    Returns:
        list[TransmissionLine]
    """
    if line_data is None:
        line_data = LINE_DATA_3BUS

    return [
        TransmissionLine(
            bus_from=f_bus, bus_to=t_bus,
            R=p['R'], L=p['L'], C=p['C'],
            dt=dt, bus_manager=bus_manager,
            damping_L=damping_L, damping_C=damping_C,
        )
        for (f_bus, t_bus), p in line_data.items()
    ]

"""
TransmissionLine: 송전선 lumped pi-model + 사다리꼴 이산화 (수치 댐핑 옵션 포함).

토폴로지:
    bus_from --(Rs_C + C/2 ground shunt)--[R -- (L ‖ Rp) 직렬]--(Rs_C + C/2 ground shunt)-- bus_to

Dommel 사다리꼴(Trapezoidal) 이산화:
    R-L 직렬 :  R-(L‖Rp) ParaEMT 방식. α=Δt/2L, β=damping_L·α=1/Rp, Δ=1+R(α+β)
                G_series = (α+β)/Δ
                icf_L    = (1-R(α-β))/Δ
                Gv1_L    = (α-β)/Δ
                I_hist   = icf_L·i_total(t-Δt) + Gv1_L·v_ij(t-Δt)
                i_total  = i_L + i_Rp  (전체 분기 전류 추적)

    R-C 직렬 :  i_C(t)     = G_RC · v_bus(t) + I_hist_RC
                G_RC       = 1 / (R_dC + Δt/C)        (C는 라인 총 C, C/2가 한 끝에)
                I_hist_RC  = -G_RC · ( v_C(t-Δt) + (Δt/C) · i_C(t-Δt) )
                v_C(t)     = v_C(t-Δt) + (Δt/C) · ( i_C(t) + i_C(t-Δt) )
                ※ damping_C=0 이면 R_dC=0 으로 순수 C와 수학적 동일.
                  v_C는 항상 별도 상태로 보관 (R_dC=0 일 때 v_C ≡ v_bus 로 자동 추적됨).

3상 평형(상호 인덕턴스 무시) 가정으로 상별 스칼라 식을 그대로 적용한다.

수치 댐핑 (사다리꼴 적분의 Nyquist 진동 억제):
    damping_L : 인덕터 병렬 R_dL = (2L/Δt) / damping_L
    damping_C : 콘덴서 직렬 R_dC = damping_C · Δt / C
    권장 범위 [0, ~0.1]. 0이면 댐핑 없음 (기본값).
"""