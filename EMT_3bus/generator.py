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

from .supEMT import BaseComponent, abc_to_dq0, dq0_to_abc


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
    'X_q_pp':  0.23,     # q축 subtransient (X''_q) — round rotor 가정
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
    'E_fd':    1.8897,      # 계자 전압 [pu]
    'T_m':     0.8208,      # 기계 토크 [pu]

    # 시스템 주파수
    'f_sys':   60.0,     # [Hz]
}


# ============================================================
# TGOV1: IEEE 증기터빈 거버너 (단순화 재열기 모델)
# ============================================================
DEFAULT_TGOV1_PARAMS = {
    'R':    0.05,   # 속도 드룹 [pu]
    'T1':   0.5,    # 가버너 시정수 [s]
    'T2':   2.1,    # 재열기 리드 시정수 [s]
    'T3':   7.0,    # 재열기 지연 시정수 [s]
    'Vmax': 1.05,   # 게이트 최대값 [pu]
    'Vmin': 0.0,    # 게이트 최소값 [pu]
    'Dt':   0.0,    # 터빈 댐핑 계수 [pu]
}


class TGOV1:
    """TGOV1 증기터빈 거버너 (IEEE Std 421.5).

    전달함수:
        Pm = [gref - ω_dev/R] · 1/(1+sT1) · (1+sT2)/(1+sT3)  −  Dt·ω_dev

    상태변수:
        P1: 1차 지연 출력 (게이트 위치, 리미터 적용)
        P2: lead-lag 출력 (재열기 출력)

    사용법:
        gov = TGOV1(T_m0=gen.T_m, dt=DT)
        gen.governor = gov
    """

    def __init__(self, T_m0, dt, params=None):
        p = dict(DEFAULT_TGOV1_PARAMS)
        if params is not None:
            p.update(params)

        self.R    = p['R']
        self.T1   = p['T1']
        self.T2   = p['T2']
        self.T3   = p['T3']
        self.Vmax = p['Vmax']
        self.Vmin = p['Vmin']
        self.Dt   = p['Dt']
        self.dt   = dt

        # 기준 토크 (ω_dev=0 정상상태 목표값)
        self.gref = float(T_m0)

        # 정상상태 초기화: ω_dev=0 → P1=P2=T_m0 → Pm=T_m0
        self.P1 = float(T_m0)
        self.P2 = float(T_m0)

    def step(self, omega_dev):
        """1스텝 진행 (Euler forward). omega_dev → Pm 반환."""
        dt = self.dt

        # 드룹 입력
        Pref = self.gref - omega_dev / self.R

        # 1차 지연 (T1)
        dP1 = (Pref - self.P1) / self.T1
        P1_new = float(np.clip(self.P1 + dP1 * dt, self.Vmin, self.Vmax))
        dP1_eff = (P1_new - self.P1) / dt   # 리미터 반영 실효 기울기

        # lead-lag 재열기 (T2/T3), 이전 P1 기준 (Euler forward)
        dP2 = (self.P1 + self.T2 * dP1_eff - self.P2) / self.T3
        P2_new = self.P2 + dP2 * dt

        Pm = P2_new - self.Dt * omega_dev

        self.P1 = P1_new
        self.P2 = P2_new
        return Pm


# ============================================================
# HYGOV: PSS/E 수력발전 조속기 모델 (5 상태변수)
# ============================================================
DEFAULT_HYGOV_PARAMS = {
    'R':     0.04,    # 영구 드룹 (permanent droop)
    'r':     1.0,     # 임시 드룹 (temporary droop)
    'Tr':    10.0,    # 리셋 시정수 [s]  (dashpot reset time)
    'Tf':    0.05,    # 필터 시정수 [s]  (pilot valve filter)
    'Tg':    10.0,    # 서보 시정수 [s]  (gate servomotor lag)
    'VELM':  0.001,   # 게이트 속도 제한 [pu/s]
    'GMAX':  1.0,     # 게이트 최대 위치 [pu]
    'GMIN':  0.0,     # 게이트 최소 위치 [pu]
    'TW':    1.0,     # 수관 기동 시정수 [s]  (water starting time)
    'At':    1.5,     # 터빈 이득 [pu]
    'Dturb': 0.0,     # 터빈 댐핑
    'qNL':   0.08,    # 무부하 유량 [pu]
}


class HYGOV:
    """HYGOV 수력발전 조속기 (PSS/E 호환, ParaEMT lib_numba.py 동일 구현).

    상태변수 (5개):
        xe  : 속도 오차 필터 출력  (Tf 지연)
        xc  : 게이트 지령           (VELM 속도 제한 + 게이트 리미터)
        xg  : 서보 지연 후 게이트  (Tg 지연)
        xq  : 수관 유량             (비선형 TW 동특성)
        Pm  : 기계 출력 [sys-base]  (대수 방정식, 다음 스텝 출력으로 보관)

    인터페이스:
        gov = HYGOV(T_m0_sys, mac_mva, sys_mva, dt)
        gen.governor = gov
        → gen.T_m = gov.step(gen.omega_dev)  (TGOV1과 동일)

    베이스 변환:
        HYGOV 내부 게이트/유량 변수는 무차원(0~1) 머신 베이스 기준.
        Pm_mac → Pm_sys = Pm_mac × (mac_mva / sys_mva) 로 변환 후 반환.
    """

    def __init__(self, T_m0_sys, mac_mva, sys_mva, dt, params=None):
        """
        Args:
            T_m0_sys: 초기 기계 토크 [pu, 시스템 베이스] (Power Flow 결과)
            mac_mva:  발전기 머신 베이스 MVA  (예: 555)
            sys_mva:  시스템 베이스 MVA       (예: 100)
            dt:       시뮬레이션 시간 스텝 [s]
            params:   파라미터 딕트 (None 이면 DEFAULT_HYGOV_PARAMS 사용)
        """
        p = dict(DEFAULT_HYGOV_PARAMS)
        if params is not None:
            p.update(params)

        self.R     = p['R']
        self.r     = p['r']
        self.Tr    = p['Tr']
        self.Tf    = p['Tf']
        self.Tg    = p['Tg']
        self.VELM  = p['VELM']
        self.GMAX  = p['GMAX']
        self.GMIN  = p['GMIN']
        self.TW    = p['TW']
        self.At    = p['At']
        self.Dturb = p['Dturb']
        self.qNL   = p['qNL']
        self.dt    = dt

        # Pm_sys = Pm_mac × _scale
        self._scale = mac_mva / sys_mva

        # 머신 베이스 초기 기계 출력
        T_m0_mac = T_m0_sys / self._scale

        # 정상상태 게이트/유량 (dX/dt=0, ω_dev=0):
        #   Pm_mac = (xq - qNL) * 1² * At  →  xq_ss = T_m0_mac/At + qNL
        xss = T_m0_mac / self.At + self.qNL

        # 상태 초기화
        self.xe   = 0.0
        self.xc   = xss
        self.xg   = xss
        self.xq   = xss
        self.Pm   = T_m0_sys          # sys-base 출력 (초기값)

        # gref = xc_ss × R  (ParaEMT Init_mac_gref 관례)
        self.gref = xss * self.R

    def step(self, omega_dev):
        """1스텝 진행 (Forward Euler, ParaEMT lib_numba.py 동일).

        Args:
            omega_dev: 속도 편차 Δω [pu]

        Returns:
            Pm_sys: 기계 출력 [pu, 시스템 베이스]
        """
        dt   = self.dt
        R, r, Tr, Tf = self.R, self.r, self.Tr, self.Tf
        Tg, TW       = self.Tg, self.TW
        At, qNL, Dturb = self.At, self.qNL, self.Dturb
        VELM, GMAX, GMIN = self.VELM, self.GMAX, self.GMIN

        xe, xc, xg, xq = self.xe, self.xc, self.xg, self.xq

        # --- 1. xe: 속도 오차 필터 (Tf 1차 지연) ---
        n1     = self.gref - (xc * R + omega_dev)
        xe_new = xe + (n1 - xe) / Tf * dt

        # --- 2. xc: 게이트 지령 (임시 드룹 + 서보 적분) ---
        dxe_dt = (xe_new - xe) / dt
        xc_new = xc + (dxe_dt * Tr + xe) / (r * Tr) * dt

        # 속도 제한 (±VELM [pu/s])
        dxc_dt = (xc_new - xc) / dt
        if dxc_dt > VELM:
            xc_new = xc + VELM * dt
        elif dxc_dt < -VELM:
            xc_new = xc - VELM * dt

        # 게이트 위치 제한
        if xc_new > GMAX:
            xc_new = GMAX
        if xc_new < GMIN:
            xc_new = GMIN

        # --- 3. xg: 서보 지연 (Tg) — OLD xc 사용 (ParaEMT 동일) ---
        xg_new = xg + (xc - xg) / Tg * dt

        # --- 4. xq: 수관 유량 (비선형 TW 동특성) — OLD xq, OLD xg 사용 ---
        xqxg   = xq / xg
        xq_new = xq + (1.0 - xqxg * xqxg) / TW * dt

        # --- 5. Pm: 대수 방정식 — OLD 상태 사용 (ParaEMT 동일) ---
        Pm_mac = (xq - qNL) * xqxg * xqxg * At - Dturb * omega_dev * xg

        # --- 상태 업데이트 ---
        self.xe = xe_new
        self.xc = xc_new
        self.xg = xg_new
        self.xq = xq_new
        self.Pm = Pm_mac * self._scale   # → sys-base

        return self.Pm


DEFAULT_SEXS_PARAMS = {
    'TA_TB':  0.0,   # TA/TB 비 (PSS/E SEXS 1번째 파라미터, lead-lag 선행 비)
    'TB':    10.0,   # 지연 시정수 [s]
    'K':    100.0,   # 여자기 이득
    'TE':     0.1,   # 여자기 시정수 [s]  (0이면 대수식)
    'Emin':   0.0,   # EFD 하한 [pu]
    'Emax':   3.0,   # EFD 상한 [pu]
    'vm_te':  0.1,   # Vt LP 필터 시상수 [s]  (ParaEMT 2gen.xlsx vm 시트, ke 행)
}


class SEXS:
    """SEXS 간략화 여자기 (PSS/E 호환, ParaEMT lib_numba.py 동일 구현).

    상태변수 (2개):
        v1  : 전압 오차 필터 출력  (TB 지연)
        EFD : 여자 전압 출력       (TE 지연 또는 대수식)

    인터페이스:
        exc = SEXS(E_fd0, Vt0, dt, params)
        gen.exciter = exc
        → gen.E_fd = exc.step(Vt)  (HYGOV와 동일 패턴)

    2gen.xlsx: TA/TB=0, TB=10, K=100, TE=0.1, Emin=0, Emax=3
    """

    def __init__(self, E_fd0, Vt0, dt, params=None):
        """
        Args:
            E_fd0: 초기 여자 전압 [pu]  (gen.initialize_states() 후 gen.E_fd)
            Vt0:   초기 단자 전압 크기 [pu]  (|V_term| at power flow)
            dt:    시뮬레이션 시간 스텝 [s]
            params: 파라미터 딕트 (None이면 DEFAULT_SEXS_PARAMS 사용)
        """
        p = dict(DEFAULT_SEXS_PARAMS)
        if params is not None:
            p.update(params)

        self.TA_TB = p['TA_TB']
        self.TB    = p['TB']
        self.K     = p['K']
        self.TE    = p['TE']
        self.Emin  = p['Emin']
        self.Emax  = p['Emax']
        self.vm_te = p['vm_te']
        self.dt    = dt

        # 정상상태 초기화:
        #   d(v1)/dt=0  →  v1_ss = vref − Vt   →  K×v1_ss = EFD_ss
        #   →  v1_ss = E_fd0/K,  vref = Vt0 + v1_ss
        self.v1   = E_fd0 / self.K
        self.EFD  = E_fd0
        self.vref = Vt0 + self.v1

        # Vt LP 필터 상태 (ParaEMT vtm, 초기값 = 정상상태 Vt)
        self.vtm  = Vt0

        # TA_TB≠0 용: 이전 스텝 입력 보관 (유한차분 미분)
        self._vref_prev = self.vref
        self._Vt_prev   = Vt0

    def step(self, Vt):
        """1스텝 진행 (Forward Euler, ParaEMT lib_numba.py 동일).

        Args:
            Vt: 단자 전압 크기 [pu]  (이전 스텝의 √(vd²+vq²))

        Returns:
            EFD: 여자 전압 [pu]
        """
        dt = self.dt
        TA_TB, TB, K, TE = self.TA_TB, self.TB, self.K, self.TE

        pv_v1  = self.v1
        pv_EFD = self.EFD

        # Vt LP 필터 (ParaEMT numba_BusMea vtm): 이전 스텝 필터 출력(vtm)을 SEXS 입력으로 사용
        vtm    = self.vtm
        nx_vtm = vtm + (Vt - vtm) / self.vm_te * dt

        # 1. v1: 전압 오차 필터 (필터된 vtm 기준)
        #   ParaEMT: nx_v1 = v1 + [(vref-vtm-v1)/TB + (TA/TB)×(dvref−dVtm)] × dt
        dvref = (self.vref - self._vref_prev) / dt
        dVt   = (vtm - self._Vt_prev) / dt
        nx_v1 = pv_v1 + ((self.vref - vtm - pv_v1) / TB
                         + TA_TB * (dvref - dVt)) * dt

        # 2. EFD: OLD v1 사용 (ParaEMT 동일, Forward Euler)
        if TE == 0.0:
            EFD = K * pv_v1
        else:
            EFD = pv_EFD + (K * pv_v1 - pv_EFD) / TE * dt

        # 리미터
        EFD = max(self.Emin, min(self.Emax, EFD))

        self._vref_prev = self.vref
        self._Vt_prev   = vtm      # 다음 스텝 미분 기준: 필터된 값
        self.vtm = nx_vtm
        self.v1  = nx_v1
        self.EFD = EFD
        return self.EFD


class Generator(BaseComponent):
    def __init__(self, bus, bus_manager, dt, params=None):
        """
        Args:
            bus:         연결될 모선 ID
            bus_manager: 공유 BusManager
            dt:          시뮬레이션 시간 스텝 [s]
            params:      파라미터 사전 (None 또는 부분 dict이면 DEFAULT_PARAMS와 병합)
        """
        self.bus = bus
        self._slot = bus_manager.register(bus)
        self._phases = bus_manager.phases
        self.dt = dt

        # 파라미터 병합
        p = dict(DEFAULT_PARAMS)
        if params is not None:
            p.update(params)
        self.params = p

        # 자주 쓰는 값을 속성으로 캐싱
        self.R_a      = p['R_a']
        self.X_d      = p['X_d']
        self.X_q      = p['X_q']
        self.X_d_p    = p['X_d_p']
        self.X_q_p    = p['X_q_p']
        self.X_d_pp   = p['X_d_pp']
        self.X_q_pp   = p['X_q_pp']    # round rotor 가정 (X''_q = X''_d)
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
        self.governor = None   # TGOV1/HYGOV 외부에서 gen.governor = ... 로 연결
        self.exciter  = None   # SEXS 등 외부에서 gen.exciter = SEXS(...) 로 연결
        self.Te       = 0.0   # 에어갭 토크 ψd·iq - ψq·id (sys-base pu), 스윙 방정식 내부용
        self.Pe       = 0.0   # 단자 전력 Pe = Te - Ra·I²  (sys-base pu), 비교 출력용
        self.tripped  = False  # True: 트립 중 (개방회로 자유회전, G 행렬에서 제외)
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
        # 이번 스텝 외삽으로 추정한 e''_abc(t)  (update_branch에서 채움, 플롯용)
        self.e_pp_abc_now  = np.zeros(n)

        # predict_state에서 외삽한 회전자 각도 (Park θ에 사용)
        self.theta_pred = 0.0

        # 현재 시뮬레이션 시각 (predict_state에서 갱신)
        self._t = 0.0

        # phasor 초기화용 (initialize_states에서 갱신, stamp_I_phasor에서 사용)
        self.V_term_p = 1.0 + 0.0j
        self.Q_op = p.get('Q_op', 0.0)
        self.P_terminal_op = p.get('T_m', 0.8)

    @property
    def psi_d_pp(self):
        """main_emt_3bus.py 하위 호환성을 위한 alias"""
        return self.psi_1d

    @property
    def psi_q_pp(self):
        """main_emt_3bus.py 하위 호환성을 위한 alias"""
        return self.psi_2q

    def _convert_to_ec_parameters(self):
        """표준 파라미터를 Kundur/ParaEMT 등가회로(Equivalent Circuit) 파라미터로 변환"""
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
        
        # 자속 갱신용 자체/상호 인덕턴스 pu
        self.L_ffd = self.L_ad * self.L_ad / (self.X_d - self.X_d_p)
        self.L_11d = self.L_ad * self.L_ad / (self.X_d - self.X_d_pp)
        self.L_11q = self.L_aq * self.L_aq / (self.X_q - self.X_q_p)
        self.L_22q = self.L_aq * self.L_aq / (self.X_q - self.X_q_pp)
        self.L_f1d = self.L_ad

    def _build_schur_matrices(self):
        """ParaEMT 방식의 Schur reduction 행렬 및 평균 저항 사전 계산"""
        dt = self.dt
        w0 = self.omega_0
        self.alpha_mac = 99.0 / 101.0  #수치 댐핑 계수
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
        """미리 축약된 Thevenin 등가 컨덕턴스 G_equiv를 G에 스탬핑.
        tripped=True이면 네트워크에서 제외 (rebuild_G 시 자동 반영).
        """
        if self.tripped:
            return
        n = self._phases
        i0 = self._slot * n
        for i in range(n):
            for j in range(n):
                G_matrix[i0 + i, i0 + j] += self.G_equiv[i, j]

    # ============================================================
    def predict_state(self, dt, t):
        """ParaEMT 방식의 2점 선형 외삽(predictX) 구현"""
        self._t = t
        
        # 외삽
        self.pd_omega_dev = 2.0 * self.pv_omega_dev_1 - self.pv_omega_dev_2
        self.pd_id = 2.0 * self.pv_id_1 - self.pv_id_2
        self.pd_iq = 2.0 * self.pv_iq_1 - self.pv_iq_2
        self.pd_EFD = 2.0 * self.pv_EFD_1 - self.pv_EFD_2
        self.pd_u_d = 2.0 * self.pv_u_d_1 - self.pv_u_d_2
        self.pd_u_q = 2.0 * self.pv_u_q_1 - self.pv_u_q_2
        
        # 각도 적분 (Trapezoidal)
        self.pd_delta = self.pv_delta_1 + dt * self.omega_0 * 0.5 * (self.pv_omega_dev_1 + self.pd_omega_dev)
        self.theta_pred = self.omega_0 * (t+dt) + self.pd_delta

    def stamp_history_current(self, I_total):
        """ParaEMT 방식의 R_av 보상 Norton 전류(updateIg) 주입 구현.
        tripped=True이면 이력 항은 계속 계산하되 네트워크 주입은 0으로 유지.
        """
        EFD2efd = self.R_fd / self.L_ad

        # 과거 전류 벡터 (tripped 중에는 id=iq=0 → pv_id_1/pv_iq_1이 0으로 수렴)
        pv_i_d_1 = np.array([-self.pv_id_1, self.pv_ifd_1, self.pv_i1d_1])
        pv_i_q_1 = np.array([-self.pv_iq_1, self.pv_i1q_1, self.pv_i2q_1])

        # History EMF (d-축) — 이력 계산은 tripped 여부와 무관하게 항상 실행
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

        if self.tripped:
            # 개방회로 자유회전: 네트워크에 전류 주입 안 함
            return

        id_src = self.ed_mod / self.R_av
        iq_src = self.eq_mod / self.R_av

        I_norton_abc = dq0_to_abc(np.array([id_src, iq_src, 0.0]), self.theta_pred)

        n = self._phases
        i0 = self._slot * n
        I_total[i0:i0+n] += I_norton_abc

    def update_branch(self, V_nodes):
        """ParaEMT 8차 상태 변수의 암시적/명시적 사다리꼴 적분(updateX) 및 pv_ 상태 변수 Shift.
        tripped=True이면 개방회로(id=iq=0): 네트워크 전압 의존 없이 자유회전.
        """
        n = self._phases
        i0 = self._slot * n

        if self.tripped:
            # 개방회로: 고정자 전류 = 0, 단자 전압은 이전 값 유지 (Pe=0)
            nx_ed = self.pv_ed_1
            nx_eq = self.pv_eq_1
            nx_id = 0.0
            nx_iq = 0.0
        else:
            # 1. 단자 전압을 Park 변환하여 ed, eq 추출 (theta_pred 사용)
            V_term_abc = V_nodes[i0:i0+n]
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
        nx_pe = nx_psyd * nx_iq - nx_psyq * nx_id          # 에어갭 토크 (스윙 방정식용)
        self.Te = nx_pe
        self.Pe = nx_pe - self.R_a * (nx_id**2 + nx_iq**2)  # 단자 전력 = Te - Ra·I²

        # 거버너 갱신: 이전 ω_dev → T_m(t+Δt)
        if self.governor is not None:
            self.T_m = self.governor.step(self.pv_omega_dev_1)

        # 여자기 갱신: 이전 단자 전압 √(vd²+vq²) → E_fd(t+Δt)
        if self.exciter is not None:
            Vt_prev = np.sqrt(self.pv_ed_1**2 + self.pv_eq_1**2)
            self.E_fd = self.exciter.step(Vt_prev)

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

        self.e_pp_abc_now = dq0_to_abc(np.array([-self.psi_2q, self.psi_1d, 0.0]), self.theta_pred)

    # ============================================================
    # 위상자 정상상태 (bumpless start)
    # ============================================================
    def stamp_Y_phasor(self, Y_matrix, omega):
        pass  # 발전기는 PQ 버스 — Y 행렬에 기여 없음 (전류원으로만 표현)

    def stamp_I_phasor(self, I_vector, omega):
        """PF 전류 주입: I_gen = (P - jQ) / conj(V)  (발전기 관례: S = V·I*)."""
        I_gen_ph = (self.P_terminal_op - 1j * self.Q_op) / np.conj(self.V_term_p)
        I_vector[self._slot] += I_gen_ph

    def update_phasor_voltage(self, V_phasor):
        """NR 반복 중 단자 전압 위상자만 갱신 (EMT 상태 보존)."""
        self.V_term_p = V_phasor[self._slot]

    def initialize_states(self, V_phasor, omega, dt):
        """위상자 V_phasor에서 회전자 상태 + 이력 역산.

        호출 전에 self.P_terminal_op, self.Q_op을 PF 역산값으로 설정해야 함.
        (sim.initialize() 후 main에서 네트워크 KCL로 계산 → gen.initialize_states 재호출)
        T_m, E_fd는 PF 결과에서 자동 역산되므로 수동 설정 불필요.
        """
        V_term_p = V_phasor[self._slot]
        self.V_term_p = V_term_p

        # P_op, Q_op: 외부(main)에서 네트워크 KCL로 계산된 값을 사용
        # NR 반복 중에는 __init__ 기본값(params) 사용 → 슬랙버스이므로 수렴에 무관
        Q_op = self.Q_op
        P_op = self.P_terminal_op

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
        
        # Field current & E_fd (PF에서 자동 역산)
        self.i_fd = (self.psi_d + (self.L_ad + self.L_l) * i_d_ss) / self.L_ad
        self.E_fd = self.i_fd * self.L_ad

        # T_m 자동 역산: Te_SS = ψd·iq - ψq·id (항상 PF와 일치, 수동 설정 불필요)
        self.T_m = self.psi_d * i_q_ss - self.psi_q * i_d_ss
        self.Te  = self.T_m   # 에어갭 토크 초기값 (스윙 방정식 기준)
        self.Pe  = self.Te - self.R_a * (i_d_ss**2 + i_q_ss**2)  # 단자 전력 초기값

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

