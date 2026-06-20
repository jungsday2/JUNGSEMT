# -*- coding: utf-8 -*-
"""
Load: Constant-Z 부하 모델 (수치 댐핑 옵션 포함).

P, Q (per-unit, 3-phase total)와 정격 전압 V_nom으로부터 등가 임피던스
    Z = |V_nom|² / (P - jQ) = R + jX
를 산출하고 X의 부호에 따라 다음 세 가지 가지로 분기한다.

    X > 0 (Q > 0, 유도성)  → R-L 직렬   (L = X/ω)
    X < 0 (Q < 0, 용량성)  → R-C 직렬   (C = 1/(ω·|X|))
    X = 0                 → 순수 R

각 가지는 사다리꼴(Trapezoidal)로 이산화되어 G에 stamping되고, 이력 전류원으로
관리된다. 3상 평형 가정으로 상별 스칼라 식을 그대로 적용한다.

수치 댐핑 (R-L 가지에 한해 적용, ParaEMT 방식):
    params['damping_L'] = β (>0) → Rp = 2L/(β·Δt) 가 L에만 병렬 (R은 항상 직렬)
    토폴로지: Vbus → R → (L ‖ Rp) → GND
    G_eq = (α+β)/Δ, icf = (1-R(α-β))/Δ, Gv1 = (α-β)/Δ, Δ = 1+R(α+β), α = Δt/(2L)
    분기 전류 상태는 전체 전류 i_total = i_L + i_Rp 추적.
    R-C 가지는 R 자체가 댐핑 역할을 겸하므로 추가 옵션 없음.

NOTE:
- Constant-Z는 흡수 전력이 |V|²에 비례하는 모델이다 (Constant-PQ나 ZIP과 트레이드오프).
- R-C 직렬은 콘덴서 단자 전압 v_C 를 모선 전압과 별도의 상태 변수로 보관한다.
- bumpless start는 추후 단계에서 initialize_states()로 보완 예정.
"""

import numpy as np

from .supEMT import BaseComponent, phasor_to_3phase_at


class Load(BaseComponent):
    def __init__(self, bus, params, dt, bus_manager):
        self.bus = bus
        self.params = params
        self.dt = dt

        self._slot = bus_manager.register(bus)
        self._phases = bus_manager.phases

        # ---- 입력 ----
        P = params['P']                       # 3상 총 유효전력 [p.u.]
        Q = params['Q']                       # 3상 총 무효전력 [p.u.] (유도성: +)
        V_nom = params.get('V_nom', 1.0)      # 정격 전압 [p.u.]
        f_sys = params.get('f_sys', 60.0)     # 계통 주파수 [Hz]
        damping_L = params.get('damping_L', 0.15)  # 인덕터 병렬 댐핑 계수 (RL 가지에만)
        damping_C = params.get('damping_C', 0.0)  # 션트 병렬 댐핑 계수 (RC 가지에만)
        omega = 2.0 * np.pi * f_sys

        if P == 0.0 and Q == 0.0:
            raise ValueError("Load with P=Q=0 has no defined impedance.")

        # ---- Z = |V|²/(P - jQ) = R + jX ----
        denom = P * P + Q * Q
        R = V_nom * V_nom * P / denom
        X = V_nom * V_nom * Q / denom
        self.R, self.X = R, X

        # ---- 분류 + 사다리꼴 이산화 계수 ----
        n = self._phases
        if abs(X) < 1e-12:
            # ===== 순수 R =====
            self.kind = 'R'
            self.G_eq = 1.0 / R
            self.G_eq_pure = self.G_eq
            # 이력 상태 없음

        elif X > 0:
            # ===== R - (L‖Rp) : ParaEMT 방식, R은 항상 직렬, Rp는 L에만 병렬 =====
            self.kind = 'RL'
            self.L = X / omega
            alpha = dt / (2.0 * self.L)          # = dt/(2L)
            beta  = damping_L * alpha             # = 1/Rp
            Delta = 1.0 + R * (alpha + beta)
            self.G_eq = (alpha + beta) / Delta
            self.icf  = (1.0 - R * (alpha - beta)) / Delta   # 이력 전류 계수
            self.Gv1  = (alpha - beta) / Delta                # 이력 전압 계수
            # 전체 분기 전류 이력 (i_L + i_Rp 합산)과 단자 전압 이력
            self.i_total_prev = np.zeros(n)
            self.v_prev = np.zeros(n)

        else:
            # ===== R-C 직렬 (X < 0) =====
            self.kind = 'RC'
            self.C = -1.0 / (omega * X)   # X<0 이므로 C>0
            self.G_eq = 1.0 / (R + dt / (2.0 * self.C))
            self.G_eq_pure = self.G_eq    # RC는 댐핑 옵션 없음
            # 분기 전류 / 콘덴서 단자 전압 (v_C ≠ v_bus, 별도 내부 상태!)
            self.i_branch_prev = np.zeros(n)
            self.v_C_prev = np.zeros(n)

        self.damping_L = damping_L

        # PF 위상자 전압 (stamp_I_phasor에서 사용; initialize_states/update_phasor_voltage에서 갱신)
        self.V_term_p = 1.0 + 0j

    # ============================================================
    # 운전점 변경 (mid-simulation 외란용)
    # ============================================================
    def set_operating_point(self, P=None, Q=None):
        """부하 운전점을 시뮬레이션 도중 변경한다.

        호출 후 simulator.rebuild_G() 호출 필수 (G 행렬에 새 G_eq 반영하기 위해).
        분기 전류·콘덴서 전압 이력은 보존됨 (인덕터 전류 연속성 유지).

        제약: kind는 변경 불가 (RL → RC 등 토폴로지 점프는 미지원).
              가능: P, Q 양수 그대로 → 같은 RL 가지에서 값만 변경.
        """
        if P is not None:
            self.params['P'] = P
        if Q is not None:
            self.params['Q'] = Q

        P_new, Q_new = self.params['P'], self.params['Q']
        if P_new == 0.0 and Q_new == 0.0:
            raise ValueError("Cannot set both P and Q to zero")

        V_nom = self.params.get('V_nom', 1.0)
        f_sys = self.params.get('f_sys', 60.0)
        damping_L = self.params.get('damping_L', 0.0)
        omega = 2.0 * np.pi * f_sys
        dt = self.dt

        # Z = R + jX
        denom = P_new * P_new + Q_new * Q_new
        R = V_nom * V_nom * P_new / denom
        X = V_nom * V_nom * Q_new / denom

        new_kind = 'R' if abs(X) < 1e-12 else ('RL' if X > 0 else 'RC')
        if new_kind != self.kind:
            raise ValueError(
                f"Load step changed kind from {self.kind} to {new_kind}; not supported."
            )

        self.R, self.X = R, X

        if self.kind == 'R':
            self.G_eq_pure = 1.0 / R
            self.G_eq = self.G_eq_pure
        elif self.kind == 'RL':
            self.L = X / omega
            alpha = dt / (2.0 * self.L)
            beta  = damping_L * alpha
            Delta = 1.0 + R * (alpha + beta)
            self.G_eq = (alpha + beta) / Delta
            self.icf  = (1.0 - R * (alpha - beta)) / Delta
            self.Gv1  = (alpha - beta) / Delta
        else:  # 'RC'
            self.C = -1.0 / (omega * X)
            self.G_eq = 1.0 / (R + dt / (2.0 * self.C))
            self.G_eq_pure = self.G_eq

    # ------------------------------------------------------------
    # G 행렬 stamping (shunt → 모선 대각만)
    # ------------------------------------------------------------
    def stamp_G(self, G_matrix):
        """대지로 가는 션트이므로 해당 모선의 3상 대각 성분에만 G_eq를 누적한다."""
        n = self._phases
        i0 = self._slot * n
        for ph in range(n):
            G_matrix[i0 + ph, i0 + ph] += self.G_eq

    # ------------------------------------------------------------
    # 이력 전류원 → 우변 I_total
    # ------------------------------------------------------------
    def stamp_history_current(self, I_total):
        if self.kind == 'R':
            return  # 순수 저항은 이력 없음

        I_hist = self._compute_I_hist()
        n = self._phases
        i0 = self._slot * n
        I_total[i0:i0+n] -= I_hist

    # ------------------------------------------------------------
    # 분기 전류 / 내부 상태 갱신
    # ------------------------------------------------------------
    def update_branch(self, V_nodes):
        if self.kind == 'R':
            return

        n = self._phases
        i0 = self._slot * n
        V = V_nodes[i0:i0+n]

        I_hist = self._compute_I_hist()

        if self.kind == 'RL':
            # i_total = i_L + i_Rp (전체 분기 전류 추적)
            i_total = self.G_eq * V + I_hist
            self.v_prev = V.copy()
            self.i_total_prev = i_total
        else:  # 'RC'
            i_curr = self.G_eq * V + I_hist
            v_C_curr = self.v_C_prev + (self.dt / (2.0 * self.C)) * (
                i_curr + self.i_branch_prev
            )
            self.v_C_prev = v_C_curr
            self.i_branch_prev = i_curr

    # ------------------------------------------------------------
    # 내부: 이력 전류 계산 (stamp 단계와 update 단계가 동일 식을 공유)
    # ------------------------------------------------------------
    def _compute_I_hist(self):
        if self.kind == 'RL':
            # I_hist = icf · i_total(t-Δt) + Gv1 · v(t-Δt)  (ParaEMT R-(L‖Rp) 공식)
            return self.icf * self.i_total_prev + self.Gv1 * self.v_prev
        else:  # 'RC'
            # I_hist = -G_eq · (v_C(t-Δt) + (Δt/(2C))·i(t-Δt))
            return -self.G_eq * (
                self.v_C_prev + (self.dt / (2.0 * self.C)) * self.i_branch_prev
            )

    # ============================================================
    # 위상자 정상상태 (bumpless start)
    # ============================================================
    def stamp_Y_phasor(self, Y_matrix, omega):
        """Const-Z: 부하 어드미턴스를 Y 행렬에 직접 stamping (PSCAD R-L 직결과 동일 특성)."""
        V_nom = self.params.get('V_nom', 1.0)
        P     = self.params['P']
        Q     = self.params['Q']
        Y_load = (P - 1j * Q) / (V_nom ** 2)
        Y_matrix[self._slot, self._slot] += Y_load
        # [Const-PQ] pass  # 부하는 PQ 주입으로 표현 — Y 행렬에 기여 없음

    def stamp_I_phasor(self, I_vector, omega):
        pass  # Const-Z: 어드미턴스가 Y 행렬에 반영됨 — 별도 전류 주입 없음
        # [Const-PQ] I_load = (P - jQ) / conj(V)  (부하는 흡수 → 음의 주입)
        # P = self.params['P']
        # Q = self.params['Q']
        # I_load = (P - 1j * Q) / np.conj(self.V_term_p)
        # I_vector[self._slot] -= I_load

    def update_phasor_voltage(self, V_phasor):
        """NR 반복 중 단자 전압 위상자만 갱신 (EMT 상태 보존)."""
        self.V_term_p = V_phasor[self._slot]

    def initialize_states(self, V_phasor, omega, dt):
        """위상자 V_phasor로부터 분기 전류·v_C·v_prev를 t=-Δt 시점 인스턴스 값으로 세팅."""
        self.V_term_p = V_phasor[self._slot]  # stamp_I_phasor용 동기화
        if self.kind == 'R':
            return  # 내부 상태 없음

        n = self._phases
        V_p = V_phasor[self._slot]
        t_prev = -dt

        if self.kind == 'RL':
            # Rp >> ωL 이므로 정상상태 i_total ≈ i_L = V/(R+jωL)
            I_p = V_p / (self.R + 1j * omega * self.L)
            self.v_prev       = phasor_to_3phase_at(V_p, omega, t_prev, n)
            self.i_total_prev = phasor_to_3phase_at(I_p, omega, t_prev, n)
        else:  # 'RC'
            # R-C 직렬: 분기 전류 + 콘덴서 전압
            Z = self.R + 1.0 / (1j * omega * self.C)
            I_p = V_p / Z
            V_C_p = V_p - self.R * I_p   # v_C = v_bus - R · i
            self.i_branch_prev = phasor_to_3phase_at(I_p,   omega, t_prev, n)
            self.v_C_prev      = phasor_to_3phase_at(V_C_p, omega, t_prev, n)
