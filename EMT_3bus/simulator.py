# -*- coding: utf-8 -*-
"""
ParaEMTSimulator: EMT 메인 시뮬레이터.

라이프사이클:
    1) sim = ParaEMTSimulator(dt, t_end)
    2) sim.add(component)            # 컴포넌트 등록 (BusManager는 sim.bus_manager 공유)
    3) sim.build()                   # G 행렬 조립 + LU 분해
    4) sim.initialize(V_initial)     # bumpless start (선택)
    5) sim.run()                     # 시간 영역 루프

매 스텝 알고리즘:
    1) Predict & 이력 전류 stamping
    2) GV = I 풀이
    3) 분기 전류/이력 갱신
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix

from .supEMT import phasor_to_3phase_at, BusManager


class ParaEMTSimulator:
    def __init__(self, dt, t_end, phases=3):
        self.dt = dt
        self.t_end = t_end
        self.bus_manager = BusManager(phases=phases)
        self.components = []

        # build() 이후 채워짐
        self._G_matrix = None
        self._LU = None
        self._I_total = None
        self._V_nodes = None

        # 현재 시뮬레이션 시각 (predict_state로 컴포넌트에 전달)
        self._t = 0.0

        # 결과
        self.results = {'time': None, 'V': None}

    # ------------------------------------------------------------
    # 컴포넌트 등록 / G 행렬 빌드
    # ------------------------------------------------------------
    def add(self, component):
        """컴포넌트를 시뮬레이터에 등록한다."""
        self.components.append(component)

    def build(self):
        """모든 컴포넌트가 등록된 뒤 호출. G 행렬 조립 + LU 분해."""
        N = self.bus_manager.num_nodes
        if N == 0:
            raise RuntimeError("No buses registered. Add components before build().")

        G = np.zeros((N, N))
        for comp in self.components:
            comp.stamp_G(G)
        self._G_matrix = G

        # 희소 LU 분해
        G_sparse = csc_matrix(G)
        self._LU = spla.splu(G_sparse)

        self._I_total = np.zeros(N)
        self._V_nodes = np.zeros(N)

    def rebuild_G(self):
        """G 행렬 재조립 + 재 LU 분해.

        시뮬레이션 도중 토폴로지 변경 (예: 부하 step, 차단기 동작) 후 호출.
        분기 전류·콘덴서 전압 등 컴포넌트 내부 이력은 그대로 유지됨.
        """
        if self._LU is None:
            raise RuntimeError("rebuild_G called before initial build().")

        N = self.bus_manager.num_nodes
        G = np.zeros((N, N))
        for comp in self.components:
            comp.stamp_G(G)
        self._G_matrix = G
        G_sparse = csc_matrix(G)
        self._LU = spla.splu(G_sparse)

    # ------------------------------------------------------------
    # 초기화 (bumpless start)
    # ------------------------------------------------------------
    def initialize(self, omega=None, V_phasor=None, max_iter=20, tol=1e-8, verbose=False):
        """위상자 정상상태 해석으로 컴포넌트 내부 상태를 세팅 (bumpless start).

        반복(iterative) 풀이:
            1. Y matrix 빌드 (한 번만, 변하지 않음)
            2. 컴포넌트의 stamp_I_phasor + initialize_states를 사이클로 호출
               (능동 소자(발전기)의 e''_phasor가 회전자 상태에 의존하므로 수렴 필요)
            3. V_phasor 변화량 < tol 또는 max_iter 도달 시 종료

        Args:
            omega:    정상상태 각주파수 [rad/s]. None이면 2π·60.
            V_phasor: 사용자가 직접 모선 위상자를 주면 반복 없이 단일 패스.
            max_iter: 최대 반복 (선형 부하·전원 + 발전기로 보통 5-10회 수렴)
            tol:      수렴 판정 (max|ΔV_phasor|)
            verbose:  반복 진행 출력
        """
        if self._LU is None:
            raise RuntimeError("Call build() before initialize().")

        if omega is None:
            omega = 2.0 * np.pi * 60.0

        N_buses = self.bus_manager.num_buses
        n_phases = self.bus_manager.phases

        if V_phasor is None:
            # Y 행렬은 컴포넌트 파라미터에만 의존 → 한 번만 빌드
            Y = np.zeros((N_buses, N_buses), dtype=complex)
            for comp in self.components:
                comp.stamp_Y_phasor(Y, omega)

            # 반복 풀이
            V_phasor = np.zeros(N_buses, dtype=complex)
            for it in range(max_iter):
                # I_phasor는 능동 소자 상태에 의존 → 매 반복 재빌드
                I_ph = np.zeros(N_buses, dtype=complex)
                for comp in self.components:
                    comp.stamp_I_phasor(I_ph, omega)

                V_phasor_new = np.linalg.solve(Y, I_ph)

                # 모든 컴포넌트 상태를 새 V_phasor로 재초기화
                for comp in self.components:
                    comp.initialize_states(V_phasor_new, omega, self.dt)

                # 수렴 체크 (첫 반복은 비교 대상 없음)
                if it > 0:
                    diff = float(np.max(np.abs(V_phasor_new - V_phasor)))
                    if verbose:
                        print(f"  iter {it}: max|ΔV_phasor| = {diff:.2e}")
                    if diff < tol:
                        if verbose:
                            print(f"  Phasor iteration converged in {it+1} iters")
                        V_phasor = V_phasor_new
                        break
                V_phasor = V_phasor_new
            else:
                if verbose:
                    print(f"  Phasor iteration did not converge in {max_iter} iters "
                          f"(last diff = {diff:.2e})")
        else:
            # 사용자 지정 V_phasor → 단일 패스
            for comp in self.components:
                comp.initialize_states(V_phasor, omega, self.dt)

        # V_nodes는 t=0 시점 평가값으로
        for slot in range(N_buses):
            i0 = slot * n_phases
            self._V_nodes[i0:i0+n_phases] = phasor_to_3phase_at(
                V_phasor[slot], omega, 0.0, n_phases
            )

        self._t = 0.0
        self._V_phasor_init = V_phasor   # 디버깅/검증용 보관

    # ------------------------------------------------------------
    # 시간 스텝 단위
    # ------------------------------------------------------------
    def step(self):
        """현재 self._t에서 한 스텝의 EMT 풀이를 수행하고 self._t를 dt만큼 진행한다."""
        # 1) Predict & history currents (현재 시각 self._t를 컴포넌트에 전달)
        self._I_total.fill(0.0)
        for comp in self.components:
            comp.predict_state(self.dt, self._t)
            comp.stamp_history_current(self._I_total)

        # 2) Solve GV = I
        self._V_nodes = self._LU.solve(self._I_total)

        # 3) Update branch currents / state
        for comp in self.components:
            comp.update_branch(self._V_nodes)

        # 4) 다음 스텝을 위해 시각 진행
        self._t += self.dt

        return self._V_nodes

    # ------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------
    def run(self):
        """전체 시간 영역 시뮬레이션을 수행한다."""
        if self._LU is None:
            self.build()

        n_steps = int(np.ceil(self.t_end / self.dt))
        N = self.bus_manager.num_nodes

        time_arr  = np.zeros(n_steps)
        V_history = np.zeros((N, n_steps))

        print(f"Starting EMT Simulation: {n_steps} steps, dt={self.dt}, t_end={self.t_end}")
        self._t = 0.0   # run 시작 시 시각 리셋
        for k in range(n_steps):
            time_arr[k] = self._t   # step() 호출 전 시각 = k·dt
            self.step()              # solve + advance self._t
            V_history[:, k] = self._V_nodes
        print("Simulation Finished.")

        self.results['time'] = time_arr
        self.results['V'] = V_history
        return time_arr, V_history

    # ------------------------------------------------------------
    # 결과 조회 헬퍼
    # ------------------------------------------------------------
    def get_bus_voltage(self, bus_id, phase=0):
        """특정 모선/상의 전압 시계열을 (time, voltage)로 반환한다."""
        if self.results['V'] is None:
            raise RuntimeError("No results available. Call run() first.")
        slot = self.bus_manager.slot_of(bus_id)
        idx = slot * self.bus_manager.phases + phase
        return self.results['time'], self.results['V'][idx, :]
