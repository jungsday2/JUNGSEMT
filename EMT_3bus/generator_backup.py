# -*- coding: utf-8 -*-
"""
Generator: 동기발전기 GENROU 6th order EMT 모델 (Thevenin 예측 인터페이스).

상태 변수 (6개):
    E_q_p     — q축 transient EMF              (시상수 T'_d0)
    E_d_p     — d축 transient EMF              (시상수 T'_q0)
    psi_d_pp  — d축 subtransient 댐퍼 자속       (시상수 T''_d0)
    psi_q_pp  — q축 subtransient 댐퍼 자속       (시상수 T''_q0)
    delta     — 회전자 각도 [rad] (동기 기준 좌표계)
    omega_dev — 속도 편차 Δω [pu]

외부 입력:
    E_fd  — 계자 전압  (현재는 상수, 추후 AVR 추가)
    T_m   — 기계적 토크 (현재는 상수, 추후 거버너 추가)

가정:
    - Standard GENROU (stator transient algebraic, 미분항 무시)
    - Saturation 없음 (S(1.0)=S(1.2)=0)
    - $X''_q = X''_d$ (round rotor 가정)

Thevenin 예측 인터페이스 매핑:
    단자 등가 회로:  v_term_abc(t) = R_a · i_abc(t) + L''_d · di_abc/dt + e''_abc(t)
                  →  R-L 직렬 + 시변 EMF (사다리꼴 이산화 라인 R-L과 동형)

    stamp_G        : G_gen = 1/(R_a + 2L''_d/Δt)  단자 모선 대각에
    predict_state  : ω, δ, rotor states 외삽 → e''_abc(t) 산출
    stamp_history_current : I_norton = G_gen·e''_abc(t) + I_hist^RL
    update_branch  : 풀린 V_term → Park dq → i_d, i_q 역산
                     → 4 rotor electrical eq. + swing eq. 사다리꼴 적분
Governor/AVR 은 향후 업데이트 예정

NOTE:
    Stage 2 (현재) — 골격만. 모든 stamping 메서드는 BaseComponent의 no-op 상속.
    Stage 3부터 실제 stamp_G, predict, history, update 채워나감.
"""

import numpy as np

from .supEMT import BaseComponent, phasor_to_3phase_at, abc_to_dq0, dq0_to_abc


# ============================================================
# 기본 파라미터 (Kundur Example 4.1 — 555 MVA round-rotor 발전기)
# 사용자가 params={...}로 일부 또는 전체 덮어쓸 수 있음
# ============================================================
DEFAULT_PARAMS = {
    # 저항
    'R_a':     0.003,    # 고정자 저항 [pu]

    # 리액턴스 [pu]
    'X_d':     1.81,     # d축 동기
    'X_q':     1.76,     # q축 동기
    'X_d_p':   0.30,     # d축 transient (X'_d)
    'X_q_p':   0.65,     # q축 transient (X'_q)
    'X_d_pp':  0.23,     # d축 subtransient (X''_d) — q축도 동일 가정
    'X_l':     0.15,     # leakage

    # 개회로 시상수 [s]
    'T_d0_p':  8.0,      # T'_d0
    'T_q0_p':  1.0,      # T'_q0
    'T_d0_pp': 0.03,     # T''_d0
    'T_q0_pp': 0.07,     # T''_q0

    # 기계
    'H':       3.5,      # 관성 상수 [s]
    'D':       0.0,      # 댐핑 [pu torque / pu speed]

    # 외부 입력 (상수 — 추후 AVR/Governor로 대체)
    'E_fd':    1.0,      # 계자 전압 [pu]
    'T_m':     0.8,      # 기계 토크 [pu]

    # 시스템 주파수
    'f_sys':   60.0,     # [Hz]
}


class Generator(BaseComponent):
    def __init__(self, bus, bus_manager, dt, params=None, is_slack=False, V_mag=1.0, V_angle=0.0):
        """
        Args:
            bus:         연결될 모선 ID
            bus_manager: 공유 BusManager
            dt:          시뮬레이션 시간 스텝 [s]
            params:      파라미터 사전 (None 또는 부분 dict이면 DEFAULT_PARAMS와 병합)
            is_slack:    True면 초기화 시 전압원으로 동작하여 T_m, E_fd를 자동 계산하고 관성을 무한대로 설정.
            V_mag:       is_slack=True일 때의 목표 전압 크기 (pu)
            V_angle:     is_slack=True일 때의 목표 전압 위상 (rad)
        """
        self.bus = bus
        self._slot = bus_manager.register(bus)
        self._phases = bus_manager.phases
        self.dt = dt

        self.is_slack = is_slack
        self.V_mag = V_mag
        self.V_angle = V_angle
        self.G_int_slack = 1e6  # 슬랙 모선 초기화용 매우 큰 컨덕턴스

        # 파라미터 병합
        p = dict(DEFAULT_PARAMS)
        if params is not None:
            p.update(params)
        self.params = p

        if self.is_slack:
            p['H'] = 1e6  # 슬랙 모선은 동적으로 무한대 모선(관성 무한대)으로 동작

        # 자주 쓰는 값을 속성으로 캐싱
        self.R_a      = p['R_a']
        self.X_d      = p['X_d']
        self.X_q      = p['X_q']
        self.X_d_p    = p['X_d_p']
        self.X_q_p    = p['X_q_p']
        self.X_d_pp   = p['X_d_pp']
        self.X_q_pp   = p['X_d_pp']    # round rotor 가정 (X''_q = X''_d)
        self.X_l      = p['X_l']
        self.T_d0_p   = p['T_d0_p']
        self.T_q0_p   = p['T_q0_p']
        self.T_d0_pp  = p['T_d0_pp']
        self.T_q0_pp  = p['T_q0_pp']
        self.H        = p['H']
        self.D        = p['D']
        self.E_fd     = p['E_fd']
        self.T_m      = p['T_m']
        self.f_sys    = p['f_sys']
        self.omega_0  = 2.0 * np.pi * self.f_sys   # 기저 각주파수 [rad/s]

        # subtransient 인덕턴스 (단자 회로용): L''_d = X''_d / ω_0  [H, pu]
        self.L_d_pp   = self.X_d_pp / self.omega_0

        # ============================================================
        # 상태 변수 (6개) — Stage 6의 Phasor 초기화 전까지 0
        # ============================================================
        self.E_q_p      = 0.0         # E'_q
        self.E_d_p      = 0.0         # E'_d
        self.psi_d_pp   = 0.0         # ψ''_d
        self.psi_q_pp   = 0.0         # ψ''_q
        self.delta      = 0.0         # 회전자 각 [rad]  (= δ(t-Δt) 이전 스텝 값)
        self.delta_prev2 = 0.0        # δ(t-2Δt) — 2점 선형 외삽용
        self.omega_dev  = 0.0         # Δω [pu]

        # ============================================================
        # EMT 인터페이스용 보조 상태
        # ============================================================
        n = self._phases
        # R-L 직렬 (R_a + L''_d) 사다리꼴 이산화 — 라인과 동일 패턴
        self.G_gen_pure = 1.0 / (self.R_a + 2.0 * self.L_d_pp / dt)
        self.alpha_gen  = (2.0 * self.L_d_pp / dt - self.R_a) / (2.0 * self.L_d_pp / dt + self.R_a)

        # 단자 분기 전류 i_RL 이력 (3상)
        self.i_term_prev = np.zeros(n)
        # 단자 전압 이력 v_term (3상)  — 사다리꼴 I_hist 계산용
        self.v_term_prev = np.zeros(n)
        # 내부 시변 EMF e''_abc 이력 (Stage 5: Norton stamping에 사용)
        self.e_pp_abc_prev = np.zeros(n)
        # 이번 스텝 외삽으로 추정한 e''_abc(t)  (predict_state에서 채움)
        self.e_pp_abc_now  = np.zeros(n)

        # dq 좌표 stator 전류 이전 스텝
        self.i_d_prev = 0.0
        self.i_q_prev = 0.0

        # predict_state에서 외삽한 회전자 각도 (Park θ에 사용)
        self.delta_pred = 0.0
        self.theta_pred = 0.0

        # 현재 시뮬레이션 시각 (predict_state에서 갱신)
        self._t = 0.0

    # ============================================================
    # Stage 3: 단자 R-L 사다리꼴 인터페이스 (e''_abc = 0 가정)
    # ============================================================
    # 단자 회로:  v_term(t) = R_a · i_term(t) + L''_d · di_term/dt + e''(t)
    # i_term의 부호 규약: 모선 → 발전기 내부로 흐르는 전류 (load의 i_branch와 동일)
    #
    # Stage 3에서는 e''_abc(t) = 0 → 발전기는 사실상 R_a + L''_d 직렬 션트 부하처럼 동작.
    # Stage 5에서 e''_abc(t)가 회전자 상태로부터 계산되면 Norton 전류원 항이 추가됨.

    def stamp_G(self, G_matrix):
        """단자 R_a + L''_d 직렬 사다리꼴 등가 컨덕턴스를 G에 누적 (3상 평형)."""
        n = self._phases
        i0 = self._slot * n
        for ph in range(n):
            G_matrix[i0 + ph, i0 + ph] += self.G_gen_pure

    # ============================================================
    # Stage 5: predict_state — δ 외삽 + e''_abc(t) 예측
    # ============================================================
    def predict_state(self, dt, t):
        """솔브 직전 호출. 회전자 상태 외삽 + e''_abc(t) 산출.

        2점 선형 외삽 (1차):
            δ_pred = 2·δ(t-Δt) - δ(t-2Δt)
            ψ''_pred ≈ ψ''(t-Δt)   (T''_d0=30ms 시상수, Δt=50µs → 변화 무시 수준)

        정상상태에선 δ(t-Δt) = δ(t-2Δt) → δ_pred = δ_ss로 자동 일치.
        외란 시 history 두 점의 기울기로 외삽.
        """
        self._t = t
        # 선형 외삽 (initialize_states에서 delta_prev2 = delta_ss)
        self.delta_pred = 2.0 * self.delta - self.delta_prev2
        self.theta_pred = self.omega_0 * t + self.delta_pred

        # e''_dq (현재 회전자 자속 사용; ω≈1 가정으로 e''_d = -ψ''_q, e''_q = +ψ''_d)
        e_dpp = -self.psi_q_pp
        e_qpp = +self.psi_d_pp

        # dq → abc 역변환 (theta_pred 사용)
        self.e_pp_abc_now = dq0_to_abc(np.array([e_dpp, e_qpp, 0.0]), self.theta_pred)

    # ============================================================
    # Stage 5: stamp_history_current — R-L 이력 + EMF Norton 추가
    # ============================================================
    def stamp_history_current(self, I_total):
        """이력 전류원 + EMF Norton 전류원을 우변 I_total에 누적.

        i_term(t) = G·(V(t) - e''(t)) + G·(V(t-Δt) - e''(t-Δt)) + α·i(t-Δt)
                  = G·V(t) - G·e''(t) + I_hist_full
        I_hist_full = G·V(t-Δt) - G·e''(t-Δt) + α·i(t-Δt)

        i_term은 모선→지면 방향이므로 MNA 부호 규약상:
            I_inj[bus] -= [-G·e''(t) + I_hist_full]
                       = G·(e''(t) + e''(t-Δt)) - (G·V(t-Δt) + α·i(t-Δt))
                       = G·(e''_now + e''_prev) - I_hist_RL_old
        """
        I_hist_RL = self.G_gen_pure * self.v_term_prev + self.alpha_gen * self.i_term_prev
        I_emf = self.G_gen_pure * (self.e_pp_abc_now + self.e_pp_abc_prev)

        n = self._phases
        i0 = self._slot * n
        I_total[i0:i0+n] += I_emf - I_hist_RL

    def update_branch(self, V_nodes):
        """Stage 5: 단자 R-L+EMF 분기 갱신 + GENROU 6 ODE 명시적 Euler 적분.

        i_term: 풀린 V_term 사용해 trap 식 그대로 적용 (EMF 항 포함).
        rotor:  algebraic stator로 i_d, i_q 재산출 후 6 ODE 적분.
        """
        n = self._phases
        i0 = self._slot * n
        V_term_abc = V_nodes[i0:i0+n]

        # ===== (1) 단자 분기 전류 (Stage 5: EMF 포함 trap) =====
        # i(t) = G·(V(t) - e''(t)) + G·(V(t-Δt) - e''(t-Δt)) + α·i(t-Δt)
        I_hist_full = (
            self.G_gen_pure * (self.v_term_prev - self.e_pp_abc_prev)
            + self.alpha_gen * self.i_term_prev
        )
        i_term_abc_curr = self.G_gen_pure * (V_term_abc - self.e_pp_abc_now) + I_hist_full
        

        # ===== (2) Park 변환 (predict_state에서 미리 만들어둔 theta_pred 재사용) =====
        v_dq = abc_to_dq0(V_term_abc, self.theta_pred)
        v_d, v_q = v_dq[0], v_dq[1]

        i_term_dq0_cur = abc_to_dq0(i_term_abc_curr, self.theta_pred)
        i_d_term, i_q_term = i_term_dq0_cur[1], i_term_dq0_cur[0]
        
        # ===== (3) Algebraic stator → i_d, i_q (Kundur convention) =====
        #     v_d = -R_a·i_d + X''_q·i_q - ψ''_q
        #     v_q = -R_a·i_q - X''_d·i_d + ψ''_d
        Det = self.R_a * self.R_a + self.X_d_pp * self.X_q_pp
        i_d = (self.R_a * (-v_d - self.psi_q_pp) + self.X_q_pp * (self.psi_d_pp - v_q)) / Det
        i_q = (self.R_a * (self.psi_d_pp - v_q) + self.X_d_pp * (v_d + self.psi_q_pp)) / Det

        # ===== (4) GENROU 4 electrical ODE (simple form, no saturation, no γ-correction) =====
        dE_q_p_dt = (
            self.E_fd - self.E_q_p - (self.X_d - self.X_d_p) * i_d
        ) / self.T_d0_p

        dpsi_d_pp_dt = (
            self.E_q_p - self.psi_d_pp - (self.X_d_p - self.X_l) * i_d
        ) / self.T_d0_pp

        dE_d_p_dt = (
            -self.E_d_p + (self.X_q - self.X_q_p) * i_q
        ) / self.T_q0_p

        dpsi_q_pp_dt = (
            -self.E_d_p - self.psi_q_pp - (self.X_q_p - self.X_l) * i_q
        ) / self.T_q0_pp

        # ===== (5) 전기 토크 + Swing equation =====
        T_e = self.psi_d_pp * i_q - self.psi_q_pp * i_d   # ω≈ω_0 가정
        domega_dev_dt = (self.T_m - T_e - self.D * self.omega_dev) / (2.0 * self.H)
        ddelta_dt = self.omega_0 * self.omega_dev

        # ===== (6) 명시적 Euler 적분 ===== trap 식이 이미 i_term에 적용되어 있으므로 rotor ODE도 명시적 Euler로 간단히 적분
        dt = self.dt
        self.E_q_p     += dt * dE_q_p_dt
        self.E_d_p     += dt * dE_d_p_dt
        self.psi_d_pp  += dt * dpsi_d_pp_dt
        self.psi_q_pp  += dt * dpsi_q_pp_dt
        self.omega_dev += dt * domega_dev_dt
        # δ 적분 전 history shift: 현재 self.delta(=δ(t-Δt))를 다음 스텝의 δ(t-2Δt) 슬롯으로 이동
        self.delta_prev2 = self.delta
        self.delta     += dt * ddelta_dt

        # ===== (7) 메모리 갱신 =====
        self.i_d_prev = i_d
        self.i_q_prev = i_q
        self.i_term_prev   = i_term_abc_curr.copy()
        self.v_term_prev   = V_term_abc.copy()
        # 다음 스텝의 e''(t-Δt) = 이번 스텝에서 stamping에 사용했던 e''(t)
        self.e_pp_abc_prev = self.e_pp_abc_now.copy()

    # ============================================================
    # 위상자 정상상태 (bumpless start)
    # ============================================================
    def stamp_Y_phasor(self, Y_matrix, omega):
        """단자 모선 대각에 Y_gen 누적. 슬랙 모선이면 전압을 강제로 고정하기 위해 큰 컨덕턴스 사용."""
        if self.is_slack:
            Y_matrix[self._slot, self._slot] += self.G_int_slack
        else:
            Y_gen = 1.0 / (self.R_a + 1j * omega * self.L_d_pp)
            Y_matrix[self._slot, self._slot] += Y_gen

    def stamp_I_phasor(self, I_vector, omega):
        """Norton 전류원을 I_phasor에 누적."""
        if self.is_slack:
            V_p = self.V_mag * np.exp(1j * self.V_angle)
            I_vector[self._slot] += V_p * self.G_int_slack
        else:
            e_pp_dq = -self.psi_q_pp + 1j * self.psi_d_pp
            e_pp_phasor = e_pp_dq * np.exp(1j * self.delta)
            Y_gen = 1.0 / (self.R_a + 1j * omega * self.L_d_pp)
            I_vector[self._slot] += Y_gen * e_pp_phasor

    def initialize_states(self, V_phasor, omega, dt):
        """Stage 6: 위상자 V_phasor에서 회전자 6 상태 + 단자 측 이력 모두 역산.

        is_slack=True인 경우, 외부 네트워크가 요구하는 P, Q를 측정하여 self.T_m을 자동 설정합니다.
        """
        V_term_p = V_phasor[self._slot]

        if self.is_slack:
            # 슬랙: 목표 전압원으로 동작하여 흐르는 실제 전류 I_gen을 역산
            V_p = self.V_mag * np.exp(1j * self.V_angle)
            I_norton = V_p * self.G_int_slack
            # 모선으로 들어오는 Norton 주입 전류 중, 자체 병렬 어드미턴스(G_int)를 제외한 나머지 = 계통으로 공급된 전류
            I_gen_phasor = I_norton - V_term_p * self.G_int_slack
            
            S_gen = V_term_p * np.conj(I_gen_phasor)
            P_op = S_gen.real
            Q_op = S_gen.imag
            
            # 동특성 해석 중 토크 불균형을 막기 위해 T_m을 현재 계통이 요구하는 유효전력으로 고정
            self.T_m = P_op
        else:
            # ===== 운전점 =====
            P_op = self.T_m
            Q_op = self.params.get('Q_op', 0.0)
            
            # ===== 발전기 전류 위상자 (machine convention) =====
            I_gen_phasor = (P_op - 1j * Q_op) / np.conj(V_term_p)

        # ===== 가상 EMF (q-축 방향) → δ_ss =====
        Z_q_phasor = self.R_a + 1j * self.X_q
        E_q_fic = V_term_p + Z_q_phasor * I_gen_phasor
        self.delta = np.angle(E_q_fic) - np.pi / 2.0
        # 정상상태에서 δ(t-2Δt) = δ(t-Δt) = δ_ss → 첫 predict_state 호출 시
        # δ_pred = 2·δ_ss - δ_ss = δ_ss로 일관성 유지
        self.delta_prev2 = self.delta
        self.omega_dev = 0.0

        # ===== dq 변환 =====
        V_dq = V_term_p   * np.exp(-1j * self.delta)
        I_dq = I_gen_phasor * np.exp(-1j * self.delta)
        v_d_ss, v_q_ss = V_dq.real, V_dq.imag
        i_d_ss, i_q_ss = I_dq.real, I_dq.imag

        # ===== ψ''을 STATOR ALGEBRAIC에서 backsolve (자동 일치) =====
        #     v_d = -R·i_d + X''_q·i_q - ψ''_q  →  ψ''_q = -R·i_d + X''_q·i_q - v_d
        #     v_q = -R·i_q - X''_d·i_d + ψ''_d  →  ψ''_d = v_q + R·i_q + X''_d·i_d
        self.psi_q_pp = -self.R_a * i_d_ss + self.X_q_pp * i_q_ss - v_d_ss
        self.psi_d_pp =  v_q_ss + self.R_a * i_q_ss + self.X_d_pp * i_d_ss

        # ===== E', E_fd를 회전자 ODE 정상상태에서 backsolve =====
        # ψ''_d ODE: 0 = E'_q - ψ''_d - (X'_d - X_l)·i_d  →  E'_q = ψ''_d + (X'_d - X_l)·i_d
        # E'_q ODE:  0 = E_fd - E'_q - (X_d - X'_d)·i_d   →  E_fd = E'_q + (X_d - X'_d)·i_d
        self.E_q_p = self.psi_d_pp + (self.X_d_p - self.X_l) * i_d_ss
        self.E_fd  = self.E_q_p    + (self.X_d   - self.X_d_p) * i_d_ss   # OVERRIDE
        # ψ''_q ODE: 0 = -E'_d - ψ''_q - (X'_q - X_l)·i_q  →  E'_d = -ψ''_q - (X'_q - X_l)·i_q
        self.E_d_p = -self.psi_q_pp - (self.X_q_p - self.X_l) * i_q_ss

        # ===== e''_phasor (정상상태) =====
        e_pp_dq = -self.psi_q_pp + 1j * self.psi_d_pp
        e_pp_phasor = e_pp_dq * np.exp(1j * self.delta)

        # ===== i_term_phasor (load convention, INTO gen) =====
        I_term_phasor = -I_gen_phasor

        # ===== 단자 측 t=-Δt 인스턴스 값 =====
        n = self._phases
        t_prev = -dt
        self.v_term_prev   = phasor_to_3phase_at(V_term_p,      omega, t_prev, n)
        self.i_term_prev   = phasor_to_3phase_at(I_term_phasor, omega, t_prev, n)
        self.e_pp_abc_prev = phasor_to_3phase_at(e_pp_phasor,   omega, t_prev, n)

        # ===== dq stator 전류 t=-Δt (DC 정상상태 그대로) =====
        self.i_d_prev = i_d_ss
        self.i_q_prev = i_q_ss 