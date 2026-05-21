# -*- coding: utf-8 -*-
"""
EMT_3bus 시뮬레이터 — 시나리오 분기 (전원 종류 + 외란 종류).

설정:
    SOURCE_TYPE: 'gen' | 'vs'
        'gen' — Bus 2에 GENROU 6th 동기기 (자속, 회전자각 등 동특성)
        'vs'  — Bus 2에 VoltageSource (해석적 사인파, 동특성 없음)

    SCENARIO   : 'load_step' | 'line_trip'
        'load_step' — t=T_EVENT에 부하 P,Q 변경
        'line_trip' — t=T_EVENT에 라인 2-3 트립 (3상 개방)
"""

import numpy as np
import matplotlib.pyplot as plt

from EMT_3bus import (
    ParaEMTSimulator,
    Generator,
    Load,
    VoltageSource,
    build_3bus_lines,
)


# ============================================================
# 시나리오 / 시뮬레이션 설정 (자주 바꾸는 값들)
# ============================================================
SOURCE_TYPE = 'gen'         # 'gen' | 'vs'
SCENARIO    = 'load_step'   # 'load_step' | 'line_trip'
T_EVENT     = 1
DT          = 50e-6
T_END       = 5.0

# load_step 시 새 운전점 (10× 시나리오)
LOAD_NEW    = {'P': 8.0, 'Q': 4.0}

# Bus 2 VoltageSource 파라미터 (gen 정상상태 V_bus2 ≈ 1.010 ∠ 0.83° 매칭)
VS2_PARAMS  = {'V_mag': 1.010, 'V_angle_deg': 0.83, 'R_internal': 0.005}

# Bus 2 Generator 파라미터
GEN_PARAMS  = {'T_m': 0.8, 'E_fd': 1.0, 'Q_op': 0.2, 'D': 2.0}


# ============================================================
def build_simulator():
    """시뮬레이터 + 컴포넌트 등록. (sim, gen_or_None, line_2_3, load) 반환."""
    sim = ParaEMTSimulator(dt=DT, t_end=T_END, phases=3)

    # 라인 (line 2-3 객체 참조 보존)
    lines = build_3bus_lines(
        dt=sim.dt, bus_manager=sim.bus_manager,
        damping_L=0.05, damping_C=0.05,
    )
    line_2_3 = next(l for l in lines if (l.bus_from, l.bus_to) == (2, 3))
    for line in lines:
        sim.add(line)

    # Bus 1: slack VoltageSource
    sim.add(VoltageSource(
        bus=1, bus_manager=sim.bus_manager,
        V_mag=1.0, V_angle_deg=0.0, f_sys=60.0, R_internal=0.001,
    ))

    # Bus 2: SOURCE_TYPE 분기
    gen = None
    if SOURCE_TYPE == 'gen':
        gen = Generator(
            bus=2, bus_manager=sim.bus_manager, dt=sim.dt,
            params=GEN_PARAMS,
        )
        sim.add(gen)
    elif SOURCE_TYPE == 'vs':
        sim.add(VoltageSource(
            bus=2, bus_manager=sim.bus_manager,
            f_sys=60.0,
            **VS2_PARAMS,
        ))
    else:
        raise ValueError(f"Unknown SOURCE_TYPE: {SOURCE_TYPE}")

    # Bus 3: 부하
    load = Load(
        bus=3,
        params={'P': 0.8, 'Q': 0.4, 'V_nom': 1.0, 'f_sys': 60.0},
        dt=sim.dt, bus_manager=sim.bus_manager,
    )
    sim.add(load)

    # G 행렬 조립 + LU 분해
    sim.build()

    return sim, gen, line_2_3, load


# ============================================================
def apply_event(sim, line_2_3, load):
    """SCENARIO에 따라 외란 적용. (G 행렬 재구성 포함)"""
    if SCENARIO == 'load_step':
        load.set_operating_point(**LOAD_NEW)
        print(f"  >>> Load step: P→{LOAD_NEW['P']}, Q→{LOAD_NEW['Q']}")
    elif SCENARIO == 'line_trip':
        line_2_3.set_active(False)
        print(f"  >>> Line 2-3 tripped (3상 개방)")
    else:
        raise ValueError(f"Unknown SCENARIO: {SCENARIO}")
    sim.rebuild_G()


# ============================================================
def run_simulation(sim, gen, line_2_3, load):
    """수동 step loop. 시간 영역 결과 dict 반환.

    gen이 None이면 회전자 관련 기록은 빈 채로 둠.
    """
    n_steps = int(T_END / DT)
    out = {
        'time': np.zeros(n_steps),
        'V':    np.zeros((9, n_steps)),
    }
    if gen is not None:
        out.update({
            'delta':    np.zeros(n_steps),
            'omega':    np.zeros(n_steps),
            'psi_dd':   np.zeros(n_steps),
            'psi_qq':   np.zeros(n_steps),
            'e_pp':     np.zeros((3, n_steps)),
            'Te':       np.zeros(n_steps),
        })

    event_done = False
    print(f"Running {n_steps} steps...")
    for k in range(n_steps):
        out['time'][k] = sim._t

        if (not event_done) and (sim._t >= T_EVENT):
            print(f"  Event '{SCENARIO}' at t={sim._t:.4f}s")
            apply_event(sim, line_2_3, load)
            event_done = True

        sim.step()
        out['V'][:, k] = sim._V_nodes

        if gen is not None:
            out['delta'][k]   = gen.delta
            out['omega'][k]   = gen.omega_dev
            out['psi_dd'][k]  = gen.psi_d_pp
            out['psi_qq'][k]  = gen.psi_q_pp
            out['e_pp'][:, k] = gen.e_pp_abc_now
            out['Te'][k]      = gen.psi_d_pp * gen.i_q_prev - gen.psi_q_pp * gen.i_d_prev
    print("Done.\n")
    return out


# ============================================================
def plot_gen_scenario(sim, gen, data):
    """발전기 시나리오: 회전자 상태 + V_abc 전체 구간 (2x3)."""
    t_ms = data['time'] * 1000
    t_event_ms = T_EVENT * 1000

    _, axes = plt.subplots(2, 3, figsize=(16, 8))

    # (0,0) δ
    ax = axes[0, 0]
    ax.plot(t_ms, np.degrees(data['delta']), 'b-', lw=1.0)
    ax.axvline(t_event_ms, color='r', ls='--', lw=0.8, label='event')
    ax.set_title('Rotor angle δ [deg]')
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('δ [deg]')
    ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.4)

    # (0,1) Δω
    ax = axes[0, 1]
    ax.plot(t_ms, data['omega'] * 100, 'r-', lw=1.0)
    ax.axvline(t_event_ms, color='r', ls='--', lw=0.8)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_title('Speed deviation Δω [% pu]')
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('Δω [% pu]')
    ax.grid(True, alpha=0.4)

    # (0,2) T_e vs T_m
    ax = axes[0, 2]
    ax.plot(t_ms, data['Te'], 'g-', lw=1.0, label='T_e')
    ax.axhline(gen.T_m, color='k', ls=':', lw=1.0, label=f'T_m={gen.T_m}')
    ax.axvline(t_event_ms, color='r', ls='--', lw=0.8)
    ax.set_title('Electrical torque T_e vs T_m')
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('T [pu]')
    ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.4)

    # (1,0) ψ''
    ax = axes[1, 0]
    ax.plot(t_ms, data['psi_dd'], 'b-', lw=1.0, label="ψ''_d")
    ax.plot(t_ms, data['psi_qq'], 'r-', lw=1.0, label="ψ''_q")
    ax.axvline(t_event_ms, color='gray', ls='--', lw=0.8)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_title("Sub-transient fluxes ψ''_d, ψ''_q")
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('ψ [pu]')
    ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.4)

    # (1,1) Bus 3 V_abc (full)
    _plot_v_abc(axes[1, 1], sim, 3, data['V'], t_ms, t_event_ms,
                title='Bus 3 (load) V_abc')

    # (1,2) Bus 2 V_abc (full)
    _plot_v_abc(axes[1, 2], sim, 2, data['V'], t_ms, t_event_ms,
                title='Bus 2 (gen) V_abc')

    plt.suptitle(
        f'[GEN scenario] event={SCENARIO} at t={T_EVENT}s (H={gen.H}, D={gen.D})',
        y=1.00, fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


def plot_vs_scenario(sim, data):
    """전압원 시나리오: 3 모선 phase A + V_abc 전체 구간 (2x3)."""
    t_ms = data['time'] * 1000
    t_event_ms = T_EVENT * 1000

    _, axes = plt.subplots(2, 3, figsize=(16, 8))
    bus_names = {1: 'Bus 1 (slack)', 2: 'Bus 2 (VS)', 3: 'Bus 3 (load)'}

    # 상단: 각 모선의 phase A 전 구간 — 60Hz 사인파의 envelope 변화 보임
    for col, bus_id in enumerate([1, 2, 3]):
        ax = axes[0, col]
        slot = sim.bus_manager.slot_of(bus_id)
        ax.plot(t_ms, data['V'][slot*3+0, :], 'tab:red', lw=0.4)
        ax.axvline(t_event_ms, color='gray', ls='--', lw=0.8, label='event')
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{bus_names[bus_id]} — Phase A (full)')
        ax.set_xlabel('Time [ms]'); ax.set_ylabel('V [pu]')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.4)

    # 하단: 3상 V_abc 전체 구간
    for col, bus_id in enumerate([1, 2, 3]):
        _plot_v_abc(axes[1, col], sim, bus_id, data['V'], t_ms, t_event_ms,
                    title=f'{bus_names[bus_id]} V_abc')

    plt.suptitle(
        f'[VS scenario] event={SCENARIO} at t={T_EVENT}s '
        f'(Bus 2 VS: V={VS2_PARAMS["V_mag"]} ∠ {VS2_PARAMS["V_angle_deg"]}°)',
        y=1.00, fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.show()


def _plot_v_abc(ax, sim, bus_id, V_history, t_ms, t_event_ms, title):
    """공통 V_abc plot helper (전체 구간)."""
    slot = sim.bus_manager.slot_of(bus_id)
    for ph, lab, c in zip(range(3), ['A','B','C'], ['tab:red','tab:green','tab:blue']):
        ax.plot(t_ms, V_history[slot*3+ph, :],
                label=f'Phase {lab}', color=c, lw=0.6)
    ax.axvline(t_event_ms, color='gray', ls='--', lw=0.8)
    ax.set_title(title)
    ax.set_xlabel('Time [ms]'); ax.set_ylabel('V [pu]')
    ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.4)


# ============================================================
def main():
    sim, gen, line_2_3, load = build_simulator()

    print(sim.bus_manager)
    print(f"\n[SOURCE_TYPE: {SOURCE_TYPE}, SCENARIO: {SCENARIO}]")
    print("Initializing (iterative phasor)...")
    sim.initialize(verbose=True)

    if gen is not None:
        T_e_init = gen.psi_d_pp*gen.i_q_prev - gen.psi_q_pp*gen.i_d_prev
        print(f"\nGenerator (pre-event):")
        print(f"  delta_0    = {np.degrees(gen.delta):+.2f} deg")
        print(f"  E_fd_impl  = {gen.E_fd:+.4f}")
        print(f"  T_e_init   = {T_e_init:+.4f}")
    print(f"  V_phasor   = {sim._V_phasor_init.round(4)}")
    print()

    data = run_simulation(sim, gen, line_2_3, load)

    if SOURCE_TYPE == 'gen':
        plot_gen_scenario(sim, gen, data)
    else:
        plot_vs_scenario(sim, data)


if __name__ == "__main__":
    main()
