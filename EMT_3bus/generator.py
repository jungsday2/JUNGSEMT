# -*- coding: utf-8 -*-
"""
Generator: 동기발전기 GENROU 6th order EMT 모델 (Thevenin 예측 인터페이스).

ParaEMT 8차 상태 변수 모델로 확장 중 (Step 1 적용됨):
    i_d, i_q  — 고정자 전류
    i_fd      — 계자 전류
    i_1d      — d축 댐퍼 권선 전류
    i_1q, i_2q— q축 댐퍼 권선 전류
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
    'E_fd':    2.01,      # 계자 전압 [pu]
    'T_m':     0.823,      # 기계 토크 [pu]

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
        # 등가 회로(EC) 파라미터 변환 및 Schur 행렬 초기화 (ParaEMT 8차 모델)
        # ============================================================
        self._convert_to_ec_parameters()
        self._build_schur_matrices()

        # ============================================================
        # 상태 변수 (8개 + 보조 자속) — Phasor 초기화 전까지 0
        # ============================================================
        self.i_d = 0.0
        self.i_q = 0.0
        self.i_fd = 0.0
        self.i_1d = 0.0
        self.i_1q = 0.0
        self.i_2q = 0.0
        self.delta      = 0.0         # 회전자 각 [rad]  (= δ(t-Δt) 이전 스텝 값)
        self.omega_dev  = 0.0         # Δω [pu]

        self.psi_d = 0.0
        self.psi_q = 0.0
        self.psi_fd = 0.0
        self.psi_1d = 0.0
        self.psi_1q = 0.0
        self.psi_2q = 0.0

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

    @property
    def psi_d_pp(self):
        """main_emt_3bus.py 하위 호환성을 위한 alias"""
        return self.psi_1d

    @property
    def psi_q_pp(self):
        """main_emt_3bus.py 하위 호환성을 위한 alias"""
        return self.psi_2q

    def _convert_to_ec_parameters(self):
        """PSS/E 표준 파라미터를 Kundur/ParaEMT 등가회로(Equivalent Circuit) 파라미터로 변환"""
        self.L_l = self.X_l
        self.L_ad = self.X_d - self.X_l
        self.L_aq = self.X_q - self.X_l

        # Field
        self.L_fd = (self.X_d_p - self.X_l) * self.L_ad / (self.L_ad - (self.X_d_p - self.X_l))
        self.R_fd = (self.L_ad + self.L_fd) / self.T_d0_p

        # 1q damper
        self.L_1q = (self.X_q_p - self.X_l) * self.L_aq / (self.L_aq - (self.X_q_p - self.X_l))
        self.R_1q = (self.L_aq + self.L_1q) / self.T_q0_p

        # 1d damper (subtransient)
        z_d = self.X_d_pp - self.X_l
        y_d = self.L_ad * self.L_fd / (self.L_ad + self.L_fd)
        self.L_1d = y_d * z_d / (y_d - z_d)
        self.R_1d = (y_d + self.L_1d) / self.T_d0_pp

        # 2q damper (subtransient)
        z_q = self.X_q_pp - self.X_l
        y_q = self.L_aq * self.L_1q / (self.L_aq + self.L_1q)
        self.L_2q = y_q * z_q / (y_q - z_q)
        self.R_2q = (y_q + self.L_2q) / self.T_q0_pp
        
        # 자속 갱신용 자체/상호 인덕턴스 (ParaEMT 기반)
        self.L_ffd = self.L_ad * self.L_ad / (self.X_d - self.X_d_p)
        self.L_11d = self.L_ad * self.L_ad / (self.X_d - self.X_d_pp)
        self.L_11q = self.L_aq * self.L_aq / (self.X_q - self.X_q_p)
        self.L_22q = self.L_aq * self.L_aq / (self.X_q - self.X_q_pp)
        self.L_f1d = self.L_ad

    def _build_schur_matrices(self):
        """ParaEMT 방식의 Schur reduction 행렬 및 평균 저항 사전 계산"""
        dt = self.dt
        w0 = self.omega_0
        self.alpha_mac = 99.0 / 101.0  # ParaEMT 수치 댐핑 계수 (자동 적용)
        alpha = self.alpha_mac
        
        # d축 인덕턴스 매트릭스 (Stator, Field, 1d-Damper)
        Ld = np.array([
            [self.L_ad + self.L_l, self.L_ad,             self.L_ad],
            [self.L_ad,            self.L_ad + self.L_fd, self.L_ad],
            [self.L_ad,            self.L_ad,             self.L_ad + self.L_1d]
        ])
        
        c0 = (1.0 + alpha) / (dt * w0)
        c1 = (1.0 + alpha) / dt
        
        self.Rd1 = np.zeros((3, 3))
        self.Rd1[0, :] = np.array([self.R_a, 0.0, 0.0]) + c0 * Ld[0, :]
        self.Rd1[1, :] = np.array([0.0, self.R_fd, 0.0]) + c1 * Ld[1, :]
        self.Rd1[2, :] = np.array([0.0, 0.0, self.R_1d]) + c1 * Ld[2, :]
        
        self.Rd2 = np.zeros((3, 3))
        self.Rd2[0, :] = np.array([self.R_a * alpha, 0.0, 0.0]) - c0 * Ld[0, :]
        self.Rd2[1, :] = np.array([0.0, self.R_fd * alpha, 0.0]) - c1 * Ld[1, :]
        self.Rd2[2, :] = np.array([0.0, 0.0, self.R_1d * alpha]) - c1 * Ld[2, :]

        self.Rd1inv = np.linalg.inv(self.Rd1[1:, 1:])
        self.Rd_reduced = self.Rd1[0, 0] - self.Rd1[0, 1:] @ self.Rd1inv @ self.Rd1[1:, 0]
        self.Rd_coe = self.Rd1[0, 1:] @ self.Rd1inv

        # q축 인덕턴스 매트릭스 (Stator, 1q-Damper, 2q-Damper)
        Lq = np.array([
            [self.L_aq + self.L_l, self.L_aq,             self.L_aq],
            [self.L_aq,            self.L_aq + self.L_1q, self.L_aq],
            [self.L_aq,            self.L_aq,             self.L_aq + self.L_2q]
        ])
        
        self.Rq1 = np.zeros((3, 3))
        self.Rq1[0, :] = np.array([self.R_a, 0.0, 0.0]) + c0 * Lq[0, :]
        self.Rq1[1, :] = np.array([0.0, self.R_1q, 0.0]) + c1 * Lq[1, :]
        self.Rq1[2, :] = np.array([0.0, 0.0, self.R_2q]) + c1 * Lq[2, :]

        self.Rq2 = np.zeros((3, 3))
        self.Rq2[0, :] = np.array([self.R_a * alpha, 0.0, 0.0]) - c0 * Lq[0, :]
        self.Rq2[1, :] = np.array([0.0, self.R_1q * alpha, 0.0]) - c1 * Lq[1, :]
        self.Rq2[2, :] = np.array([0.0, 0.0, self.R_2q * alpha]) - c1 * Lq[2, :]

        self.Rq1inv = np.linalg.inv(self.Rq1[1:, 1:])
        self.Rq_reduced = self.Rq1[0, 0] - self.Rq1[0, 1:] @ self.Rq1inv @ self.Rq1[1:, 0]
        self.Rq_coe = self.Rq1[0, 1:] @ self.Rq1inv

        # 평균 등가 저항 (Round-rotor average) 및 3x3 abc 등가 행렬
        self.R_av = (self.Rd_reduced + self.Rq_reduced) / 2.0
        
        R0 = self.R_a + c0 * self.X_l  # 0-sequence 간이 근사
        Rs = (R0 + 2.0 * self.R_av) / 3.0
        Rm = (R0 - self.R_av) / 3.0
        self.R_equiv = np.array([[Rs, Rm, Rm], [Rm, Rs, Rm], [Rm, Rm, Rs]])
        self.G_equiv = np.linalg.inv(self.R_equiv)

    def stamp_G(self, G_matrix):
        """미리 축약된 ParaEMT 방식의 Thevenin 등가 컨덕턴스 G_equiv를 G에 스탬핑"""
        n = self._phases
        i0 = self._slot * n
        for i in range(n):
            for j in range(n):
                G_matrix[i0 + i, i0 + j] += self.G_equiv[i, j]

    # ============================================================
    def predict_state(self, dt, t):
        """ParaEMT 방식의 2점 선형 외삽(predictX) 구현"""
        self._t = t
        
        # 2점 외삽
        self.pd_omega_dev = 2.0 * self.pv_omega_dev_1 - self.pv_omega_dev_2
        self.pd_id = 2.0 * self.pv_id_1 - self.pv_id_2
        self.pd_iq = 2.0 * self.pv_iq_1 - self.pv_iq_2
        self.pd_EFD = 2.0 * self.pv_EFD_1 - self.pv_EFD_2
        self.pd_u_d = 2.0 * self.pv_u_d_1 - self.pv_u_d_2
        self.pd_u_q = 2.0 * self.pv_u_q_1 - self.pv_u_q_2
        
        # 각도 적분 (Trapezoidal)
        self.pd_delta = self.pv_delta_1 + dt * self.omega_0 * 0.5 * (self.pv_omega_dev_1 + self.pd_omega_dev)
        self.theta_pred = self.omega_0 * t + self.pd_delta

    def stamp_history_current(self, I_total):
        """ParaEMT 방식의 R_av 보상 Norton 전류(updateIg) 주입 구현"""
        # EFD2efd (field resistance / Lad)
        EFD2efd = self.R_fd / self.L_ad
        
        # 과거 전류 벡터 (MNA 네트워크 주입 부호 관점, 즉 Load convention으로 변경하기 위해 i_d, i_q의 부호 반전)
        pv_i_d_1 = np.array([-self.pv_id_1, self.pv_ifd_1, self.pv_i1d_1])
        pv_i_q_1 = np.array([-self.pv_iq_1, self.pv_i1q_1, self.pv_i2q_1])
        
        # History EMF (d-축)
        temp1 = np.dot(self.Rd2[0, :], pv_i_d_1)
        self.pv_his_d_1 = -self.alpha_mac * self.pv_ed_1 + self.alpha_mac * self.pv_u_d_1 + temp1
        
        temp2 = np.dot(self.Rd2[1, :], pv_i_d_1)
        self.pv_his_fd_1 = -self.alpha_mac * self.pv_EFD_1 * EFD2efd + temp2
        
        temp3 = np.dot(self.Rd2[2, :], pv_i_d_1)
        self.pv_his_1d_1 = temp3
        
        # History EMF (q-축)
        temp4 = np.dot(self.Rq2[0, :], pv_i_q_1)
        self.pv_his_q_1 = -self.alpha_mac * self.pv_eq_1 + self.alpha_mac * self.pv_u_q_1 + temp4
        
        temp5 = np.dot(self.Rq2[1, :], pv_i_q_1)
        self.pv_his_1q_1 = temp5
        
        temp6 = np.dot(self.Rq2[2, :], pv_i_q_1)
        self.pv_his_2q_1 = temp6
        
        # Schur reduction
        self.pv_his_red_d_1 = self.pv_his_d_1 - (self.Rd_coe[0] * (self.pv_his_fd_1 - self.pd_EFD * EFD2efd) + self.Rd_coe[1] * temp3)
        self.pv_his_red_q_1 = self.pv_his_q_1 - (self.Rq_coe[0] * temp5 + self.Rq_coe[1] * temp6)
        
        ed_temp = self.pd_u_d + self.pv_his_red_d_1
        eq_temp = self.pd_u_q + self.pv_his_red_q_1
        
        # 돌극성 보정 (Saliency compensation)
        self.ed_mod = ed_temp - (self.Rd_reduced - self.Rq_reduced) / 2.0 * self.pd_id
        self.eq_mod = eq_temp + (self.Rd_reduced - self.Rq_reduced) / 2.0 * self.pd_iq
        
        id_src = self.ed_mod / self.R_av
        iq_src = self.eq_mod / self.R_av
        
        # dq0 -> abc 변환 (Kundur 컨벤션 사용, generator OUT 기준)
        I_norton_abc = dq0_to_abc(np.array([id_src, iq_src, 0.0]), self.theta_pred)
        
        n = self._phases
        i0 = self._slot * n
        I_total[i0:i0+n] += I_norton_abc

    def update_branch(self, V_nodes):
        """ParaEMT 8차 상태 변수의 암시적/명시적 사다리꼴 적분(updateX) 및 pv_ 상태 변수 Shift"""
        n = self._phases
        i0 = self._slot * n
        V_term_abc = V_nodes[i0:i0+n]

        # 1. 단자 전압을 Park 변환하여 ed, eq 추출 (theta_pred 사용)
        v_dq = abc_to_dq0(V_term_abc, self.theta_pred)
        nx_ed = v_dq[0]
        nx_eq = v_dq[1]

        # 2. 고정자 전류 id, iq 계산
        nx_id = (self.ed_mod - nx_ed) / self.R_av
        nx_iq = (self.eq_mod - nx_eq) / self.R_av

        # 3. 회전자 전류 ifd, i1d, i1q, i2q 계산 (Schur 축약 역산)
        EFD2efd = self.R_fd / self.L_ad
        
        # d-축
        v1_d = self.pd_EFD * EFD2efd - self.pv_his_fd_1 + self.Rd1[1, 0] * nx_id
        v2_d = -self.pv_his_1d_1 + self.Rd1[2, 0] * nx_id
        nx_ifd = self.Rd1inv[0, 0] * v1_d + self.Rd1inv[0, 1] * v2_d
        nx_i1d = self.Rd1inv[1, 0] * v1_d + self.Rd1inv[1, 1] * v2_d

        # q-축
        v1_q = -self.pv_his_1q_1 + self.Rq1[1, 0] * nx_iq
        v2_q = -self.pv_his_2q_1 + self.Rq1[2, 0] * nx_iq
        nx_i1q = self.Rq1inv[0, 0] * v1_q + self.Rq1inv[0, 1] * v2_q
        nx_i2q = self.Rq1inv[1, 0] * v1_q + self.Rq1inv[1, 1] * v2_q

        # 4. 자속 계산
        nx_psyd = -(self.L_ad + self.L_l) * nx_id + self.L_ad * nx_ifd + self.L_ad * nx_i1d
        nx_psyq = -(self.L_aq + self.L_l) * nx_iq + self.L_aq * nx_i1q + self.L_aq * nx_i2q
        
        # 회전자 자속 갱신
        nx_psyfd = -self.L_ad * nx_id + self.L_ffd * nx_ifd + self.L_f1d * nx_i1d
        nx_psy1d = -self.L_ad * nx_id + self.L_f1d * nx_ifd + self.L_11d * nx_i1d
        nx_psy1q = -self.L_aq * nx_iq + self.L_11q * nx_i1q + self.L_aq * nx_i2q
        nx_psy2q = -self.L_aq * nx_iq + self.L_aq * nx_i1q + self.L_22q * nx_i2q

        # 5. 전기 토크 및 속도, 각도 적분 (Swing equation)
        nx_pe = nx_psyd * nx_iq - nx_psyq * nx_id
        
        omega_pu = 1.0 + self.pv_omega_dev_1
        domega_dev_dt = (self.T_m / omega_pu - nx_pe - self.D * self.pv_omega_dev_1) / (2.0 * self.H)
        nx_omega_dev = self.pv_omega_dev_1 + domega_dev_dt * self.dt
        
        nx_delta = self.pv_delta_1 + self.dt * self.omega_0 * 0.5 * (self.pv_omega_dev_1 + nx_omega_dev)
        
        nx_omega_pu = 1.0 + nx_omega_dev
        nx_u_d = -nx_psyq * nx_omega_pu
        nx_u_q = nx_psyd * nx_omega_pu

        # 6. 현재 상태 변수 갱신
        self.i_d, self.i_q = nx_id, nx_iq
        self.i_fd, self.i_1d = nx_ifd, nx_i1d
        self.i_1q, self.i_2q = nx_i1q, nx_i2q
        
        self.psi_d, self.psi_q = nx_psyd, nx_psyq
        self.psi_fd, self.psi_1d = nx_psyfd, nx_psy1d
        self.psi_1q, self.psi_2q = nx_psy1q, nx_psy2q
        
        self.delta, self.omega_dev = nx_delta, nx_omega_dev

        # 7. pv_ 상태 변수 Shift (History 업데이트: 1 -> 2, nx -> 1)
        self.pv_delta_2, self.pv_omega_dev_2 = self.pv_delta_1, self.pv_omega_dev_1
        self.pv_id_2, self.pv_iq_2 = self.pv_id_1, self.pv_iq_1
        self.pv_ifd_2, self.pv_i1d_2 = self.pv_ifd_1, self.pv_i1d_1
        self.pv_i1q_2, self.pv_i2q_2 = self.pv_i1q_1, self.pv_i2q_1
        self.pv_ed_2, self.pv_eq_2 = self.pv_ed_1, self.pv_eq_1
        self.pv_EFD_2, self.pv_u_d_2, self.pv_u_q_2 = self.pv_EFD_1, self.pv_u_d_1, self.pv_u_q_1
        
        self.pv_delta_1, self.pv_omega_dev_1 = nx_delta, nx_omega_dev
        self.pv_id_1, self.pv_iq_1 = nx_id, nx_iq
        self.pv_ifd_1, self.pv_i1d_1 = nx_ifd, nx_i1d
        self.pv_i1q_1, self.pv_i2q_1 = nx_i1q, nx_i2q
        self.pv_ed_1, self.pv_eq_1 = nx_ed, nx_eq
        self.pv_EFD_1, self.pv_u_d_1, self.pv_u_q_1 = self.E_fd, nx_u_d, nx_u_q

        # main_emt_3bus.py 플로팅 호환성 유지를 위한 기록용 변수 갱신
        self.i_d_prev, self.i_q_prev = nx_id, nx_iq
        self.e_pp_abc_now = dq0_to_abc(np.array([-self.psi_2q, self.psi_1d, 0.0]), self.theta_pred)

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
        
        self.omega_dev = 0.0

        # ===== dq 변환 =====
        V_dq = V_term_p   * np.exp(-1j * self.delta)
        I_dq = I_gen_phasor * np.exp(-1j * self.delta)
        v_d_ss, v_q_ss = V_dq.real, V_dq.imag
        i_d_ss, i_q_ss = I_dq.real, I_dq.imag

        # ===== dq stator 전류 t=-Δt (DC 정상상태) =====
        self.i_d_prev = i_d_ss
        self.i_q_prev = i_q_ss 
        
        # ============================================================
        # NEW ParaEMT 8th-order State Initialization
        # ============================================================
        self.i_d = i_d_ss
        self.i_q = i_q_ss
        self.i_1d = 0.0
        self.i_1q = 0.0
        self.i_2q = 0.0
        
        self.psi_d = v_q_ss + self.R_a * i_q_ss
        self.psi_q = -v_d_ss - self.R_a * i_d_ss
        
        # Field current & E_fd
        self.i_fd = (self.psi_d + (self.L_ad + self.L_l) * i_d_ss) / self.L_ad
        if not self.is_slack:
            self.E_fd = self.i_fd * self.L_ad  # Override parameter E_fd for consistency
            
        # Fluxes
        self.psi_fd = -self.L_ad * i_d_ss + (self.L_ad + self.L_fd) * self.i_fd
        self.psi_1d = -self.L_ad * i_d_ss + self.L_ad * self.i_fd
        self.psi_1q = -self.L_aq * i_q_ss
        self.psi_2q = -self.L_aq * i_q_ss

        # Predictor/History 1-step past states
        self.pv_delta_1 = self.delta
        self.pv_omega_dev_1 = 0.0
        self.pv_id_1 = i_d_ss
        self.pv_iq_1 = i_q_ss
        self.pv_ifd_1 = self.i_fd
        self.pv_i1d_1 = self.i_1d
        self.pv_i1q_1 = self.i_1q
        self.pv_i2q_1 = self.i_2q
        self.pv_ed_1 = v_d_ss
        self.pv_eq_1 = v_q_ss
        self.pv_EFD_1 = self.E_fd
        
        omega_pu = 1.0 + self.pv_omega_dev_1
        self.pv_u_d_1 = -self.psi_q * omega_pu
        self.pv_u_q_1 = self.psi_d * omega_pu
        
        # Predictor/History 2-step past states
        self.pv_delta_2 = self.delta
        self.pv_omega_dev_2 = 0.0
        self.pv_id_2 = i_d_ss
        self.pv_iq_2 = i_q_ss
        self.pv_ifd_2 = self.i_fd
        self.pv_i1d_2 = self.i_1d
        self.pv_i1q_2 = self.i_1q
        self.pv_i2q_2 = self.i_2q
        self.pv_ed_2 = v_d_ss
        self.pv_eq_2 = v_q_ss
        self.pv_EFD_2 = self.E_fd
        self.pv_u_d_2 = self.pv_u_d_1
        self.pv_u_q_2 = self.pv_u_q_1

        # ===== e''_phasor (정상상태) =====
        e_pp_dq = -self.psi_2q + 1j * self.psi_1d
        e_pp_phasor = e_pp_dq * np.exp(1j * self.delta)

        # ===== i_term_phasor (load convention, INTO gen) =====
        I_term_phasor = -I_gen_phasor

        # ===== 단자 측 t=-Δt 인스턴스 값 =====
        n = self._phases
        t_prev = -dt
        self.v_term_prev   = phasor_to_3phase_at(V_term_p,      omega, t_prev, n)
        self.i_term_prev   = phasor_to_3phase_at(I_term_phasor, omega, t_prev, n)
        self.e_pp_abc_prev = phasor_to_3phase_at(e_pp_phasor,   omega, t_prev, n)