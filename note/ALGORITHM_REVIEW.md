# 3-bus EMT 시뮬레이터 — 알고리즘 정리

지금까지 만든 코드를 **알고리즘 흐름 중심**으로 한 번에 쭉 정리.

## 전체 구조 한눈에

```
EMT_TEST/
├── main_emt_3bus.py            ─ 진입점 (Bus 2 = Generator 고정, .mat 저장 + plot)
├── _archive/                   ─ 백업 (옛 VS 모드 main 등)
│   └── main_emt_3bus_VS.py
└── EMT_3bus/                   ─ 패키지
    ├── __init__.py             ─ public API 재노출
    ├── supEMT.py               ─ 인프라/유틸 통합
    │                              · BusManager (모선 ID ↔ 0-indexed 슬롯)
    │                              · Park 변환 (abc ↔ dq0, Kundur, amplitude-invariant 2/3)
    │                              · BaseComponent (7개 메서드 컨트랙트)
    │                              · phasor_to_3phase_at (위상자 → 시간 인스턴스)
    ├── voltage_source.py       ─ 이상 전원 (해석적, 이산화 없음)
    ├── transmission_line.py    ─ pi-model + 사다리꼴 (damping_L, damping_C)
    ├── load.py                 ─ Constant-Z (R / RL / RC), RL에 damping_L 옵션
    ├── generator.py            ─ GENROU 6th order
    └── simulator.py            ─ 솔버 (build → init → step → run)
```

> 과거 분리됐던 `base_component.py`, `bus_manager.py`, `park.py`는 모두 `supEMT.py` 한 파일로
> 통합되어 있다. 도메인 컴포넌트(Generator, Load, …)는 모두 `from .supEMT import …`로
> 인터페이스/유틸을 가져온다.

## 1. EMT 한 스텝의 알고리즘 (핵심)

매 시간 스텝 $\Delta t$ 마다 일어나는 일을 도식화하면:

```
─── step k 시작 ─── 시각 t_k = k·Δt ───────────────
│
│  ① predict_state(dt, t_k)               ◀ 솔브 직전 외삽
│     ├ VoltageSource: self._t = t_k 캐싱
│     ├ Generator: δ_pred = 2·δ(t-Δt) − δ(t-2Δt)   (2점 선형 외삽, 1차 정확도)
│     │                                              ※ history 2점만 있으면 derivative
│     │                                                 정보 없이 외삽 가능. 정상상태에선
│     │                                                 두 점이 같아 δ_pred = δ_ss 자동.
│     │                                              ※ initialize_states에서 두 슬롯 모두
│     │                                                 δ_ss로 셋업되어 첫 스텝 fallback 불필요
│     │            θ_pred = ω_0·t_k + δ_pred
│     │            e''_d = -ψ''_q              (ψ''는 외삽 없이 현재값 그대로 사용:
│     │            e''_q = +ψ''_d               T''_d0=30ms ≫ Δt=50µs → 무시 가능)
│     │            e''_abc(t_k) = inv_Park(e''_dq, θ_pred)
│     └ 라인/부하: no-op (선형 시불변)
│
│  ② stamp_history_current(I_total)       ◀ 우변 I 벡터 조립
│     ├ VoltageSource:  +V_phase(t_k)/R_int  (Norton 전원)
│     ├ Line R-L 직렬:  ±I_hist^RL  (from→to 방향)
│     ├ Line R-C 션트:  -I_hist^RC  (각 끝)
│     ├ Load:           -I_hist (RL 또는 RC 따라)
│     └ Generator:      +G_gen·(e''(t)+e''(t-Δt)) - I_hist^RL
│
│  ③ V_nodes = LU.solve(I_total)          ◀ G·V = I 풀이
│     (G는 build()에서 미리 LU 분해됨 — 토폴로지 안 변하면 재분해 X)
│
│  ④ update_branch(V_nodes)               ◀ 솔브 후 상태 갱신
│     ├ Line: i_series, i_shunt, v_C_*, v_*_prev 모두 trapezoidal 한 스텝 진행
│     ├ Load: i_branch, v_C 등 갱신
│     ├ VoltageSource: 메모리 없음 (no-op)
│     └ Generator: 큰 일 → 아래 ⑤
│
│  ⑤ Generator의 update_branch (가장 복잡):
│     a. V_term_abc → Park(θ_pred) → v_d, v_q
│     b. Algebraic stator로 i_d, i_q 산출 (Cramer)
│        v_d = -R_a·i_d + X''_q·i_q - ψ''_q
│        v_q = -R_a·i_q - X''_d·i_d + ψ''_d
│     c. GENROU 4 electrical ODE 미분값 (E'_q, E'_d, ψ''_d, ψ''_q)
│     d. T_e = ψ''_d·i_q - ψ''_q·i_d
│     e. Swing: dδ/dt = ω_0·Δω, dΔω/dt = (T_m - T_e - D·Δω)/(2H)
│     f. 명시적 Euler로 6 상태 변수 한 스텝 적분
│        ※ δ 적분 직전 history shift: delta_prev2 ← delta (선형 외삽 history용)
│     g. e_pp_abc_prev ← e_pp_abc_now (다음 스텝 history용)
│
│  ⑥ self._t += Δt                        ◀ 시각 진행
│
─── step k 종료 ──────────────────────────────────
```

**핵심 통찰**:
- ①과 ②는 "**미래 정보**" (t_k의 V를 모르는 상태에서 t_k에 대한 추정)로 G·V=I 풀이를 가능하게 함
- ④, ⑤는 "**진짜 상태**"를 솔브된 V로 갱신
- 외삽(①)이 살짝 부정확해도 G에 발전기 임피던스 ($R_a + L''_d$)가 들어가 있어 implicit coupling이 자기 보정

## 2. Bumpless start 알고리즘 (Stage 6)

`sim.initialize()` 의 반복 fixed-point 풀이 (Picard iteration, NOT 뉴턴-랩슨):

```
Y_phasor = ∑ comp.stamp_Y_phasor(ω)        ◀ 한 번만 빌드, 절대 안 바뀜
                                              (파라미터에만 의존)

for it in range(max_iter):
    I_phasor = 0
    for each component:
        I_phasor += comp.stamp_I_phasor(ω)  ◀ "읽기": 현재 회전자 상태로 e''_phasor 계산
    
    V_phasor_new = solve(Y, I_phasor)        ◀ 선형 1-shot 풀이
    
    for each component:
        comp.initialize_states(V_phasor_new, ω, dt)  ◀ "쓰기": 새 V로 회전자 상태 갱신
    
    if it > 0 and max|ΔV_phasor| < tol:     ◀ 첫 iter는 비교 대상 없음
        break

# t=0 시점 V_nodes 세팅 + sim._t = 0
```

**왜 반복?** 발전기의 e''_phasor는 회전자 상태(ψ''_d, ψ''_q, δ)에 의존, 회전자 상태는 V_phasor에 의존, V_phasor는 e''_phasor에 의존 → 닭과 달걀.

**시차 굴림 (한 박자 늦은 Picard 갱신)**:
- iter 0: ψ''=δ=0 → I_gen=0 → 부하/슬랙만으로 V 1차 추정 → 그 V로 회전자 상태 *첫 계산*
- iter 1: iter 0의 회전자 상태로 I_gen 계산 → V 재추정 → 회전자 상태 갱신
- iter ≥1: |V_new − V_old| < tol 검사

**왜 NR이 아니어도 되나?** 부하 Constant-Z + 발전기 PQ 모드 (T_m, Q_op 지정) + slack 강함이라는
세 가지 단순화 덕분. 비선형성은 **발전기 자기일관성 한 곳뿐** → fixed-point 수렴이 빠름.
Constant-PQ 부하 / PV 모선이 도입되면 NR이 자연스러움.

**왜 수렴?** Slack이 강해서 발전기의 V 영향력 작음 → 수축이 빠르고 보통 5–10 iter면 ε=1e-9.

### 발전기 backsolve 핵심 (initialize_states)

```
V_term_p = V_phasor[gen_bus]
P_op = T_m,  Q_op = params['Q_op']

I_gen_p = (P_op - jQ_op) / V_term_p*

E_q_fic = V_term_p + (R_a + j·X_q)·I_gen_p
δ_ss = ∠E_q_fic - π/2          ◀ 부하각 도출

V_dq = V_term_p · e^(-jδ_ss),    I_dq = I_gen_p · e^(-jδ_ss)

# stator algebraic에서 ψ'' 직접 backsolve (자동 일관성)
ψ''_q = -R_a·i_d + X''_q·i_q - v_d
ψ''_d = v_q + R_a·i_q + X''_d·i_d

# rotor ODE 정상상태 조건에서 E'_q, E'_d, E_fd backsolve
E'_q = ψ''_d + (X'_d - X_l)·i_d
E_fd = E'_q + (X_d - X'_d)·i_d   ◀ 사용자 입력 OVERRIDE!
E'_d = -ψ''_q - (X'_q - X_l)·i_q

# t=-Δt 시점 인스턴스 값으로 모든 이력 세팅
v_term_prev   = phasor_to_3phase_at(V_term_p,  ω, -Δt)
i_term_prev   = phasor_to_3phase_at(-I_gen_p,  ω, -Δt)   ◀ 부호 반전 (gen→load conv.)
e_pp_abc_prev = phasor_to_3phase_at(e_pp_phasor, ω, -Δt)
```

**왜 E_fd override?**
- Simple GENROU의 ψ''_d ODE는 $X'_d - X_l$ 사용
- 물리적으로는 $X_d - X''_d$ 가 맞음
- $X''_d \neq X_l$ 일 때 두 식이 어긋남 → stator algebraic이 만족 안 됨
- **자동 보정**: 사용자 E_fd 무시하고 운전점에 맞는 값으로 강제

## 3. 컴포넌트별 모델 요약

### TransmissionLine (pi-model)

```
bus_from ──[R-L 직렬 || R_dL]── bus_to
   │                              │
[R_dC + C/2]                  [R_dC + C/2]   ◀ R_dC=0이면 순수 C
   │                              │
  ground                        ground

Trapezoidal:
  G_series = 1/(R + 2L/Δt) + 1/R_dL  ◀ 인덕터 병렬 댐핑
  G_shunt  = C/((1+damping_C)·Δt)    ◀ 콘덴서 직렬 댐핑
  α        = (2L/Δt - R)/(2L/Δt + R)
  
상태 (3상):
  i_series_prev (i_RL — 인덕터 분기만, 댐핑 R 분 제외)
  i_shunt_from/to_prev
  v_from/to_prev (모선 V 이력, R-L history용)
  v_C_from/to_prev (콘덴서 단자 V, R-C history용)

set_active(False) 시: 모든 stamping no-op (트립)
```

### Load (Constant-Z, kind ∈ {R, RL, RC})

```
P, Q, V_nom 입력 → Z = R + jX = V_nom²/(P-jQ) 산출
X 부호로 kind 분기:
  X > 0  → 'RL' (유도성)  : trapezoidal R-L (라인과 동형)
                            params['damping_L']로 인덕터 병렬 R_dL 옵션
  X < 0  → 'RC' (용량성)  : trapezoidal R-C 직렬 (v_C 별도 추적)
                            (R 자체가 댐핑 역할 → 별도 옵션 없음)
  X = 0  → 'R'            : 단순 G_eq = 1/R, 이력 없음

set_operating_point(P, Q): 계수 재계산 (kind 변경 불가, 이력 보존)
```

### VoltageSource

```
v_phase(t) = V_mag · cos(ω_0·t + φ + offset_phase)
  offset = [0, -2π/3, +2π/3] (3상)

Norton 등가:
  G_int = 1/R_internal
  I_norton(t) = v_phase(t) · G_int  ◀ 매 스텝 함수값 샘플링

이산화 없음 (미분 방정식 미보유)
```

### Generator (GENROU 6th)

```
상태 변수 6개:
  E'_q, E'_d        (transient EMF — T'_d0=8s, T'_q0=1s)
  ψ''_d, ψ''_q      (subtransient flux — T''_d0=30ms, T''_q0=70ms)
  δ, Δω             (rotor angle, speed deviation)

단자 인터페이스: R-L 직렬 (R_a + L''_d) + 시변 EMF e''_abc(t)
  → 라인의 R-L과 동형 (재사용)
  → 단, EMF 부분이 매 스텝 회전자 상태에서 새로 계산됨

GENROU 4 ODE (simple form):
  T'_d0  Ė'_q   = E_fd - E'_q - (X_d - X'_d)·i_d
  T''_d0 ψ̇''_d = E'_q - ψ''_d - (X'_d - X_l)·i_d
  T'_q0  Ė'_d   = -E'_d + (X_q - X'_q)·i_q
  T''_q0 ψ̇''_q = -E'_d - ψ''_q - (X'_q - X_l)·i_q

Swing:
  δ̇    = ω_0·Δω
  2H Δω̇ = T_m - T_e - D·Δω,   T_e = ψ''_d·i_q - ψ''_q·i_d

적분: 명시적 Euler (slow rotor + small dt → 안정)
```

## 4. 핵심 컨벤션 정리

| 항목 | 선택 | 이유 |
|---|---|---|
| Park 변환 | Kundur, **2/3 amplitude-invariant** | dq peak가 abc peak와 일치 |
| 위상자 | **Peak amplitude** + 3-phase p.u. | $S = V \cdot I^*$ 그대로 (RMS 변환 X) |
| 전류 부호 | i_term = bus→gen INTO (load 방향) | 라인·부하와 동형 |
| | i_d, i_q = gen→bus OUT (machine 방향) | 발전기 표준 |
| 회전자 각 | $\theta_{Park} = \omega_0 t + \delta$ | absolute 각도 (절대 stator 기준) |
| 시간 인덱스 | step k에서 self._t = k·Δt | step()이 끝날 때 진행 |

## 5. 외란 / 토폴로지 변경 메커니즘

```
시점 t = T_EVENT 에서:

[load step]         load.set_operating_point(P, Q)
                    → kind 그대로, 계수 재계산
                    → sim.rebuild_G() 필수

[line trip]         line.set_active(False)
                    → 모든 stamping no-op
                    → sim.rebuild_G() 필수

[reclose]           line.set_active(True)
                    → 다시 활성, 이력은 그대로 (재투입)

rebuild_G():        G ← 0; for comp: comp.stamp_G(G)
                    LU ← splu(csc_matrix(G))
                    (기존 이력 보존, G만 재구성)
```

## 6. 알려진 한계 / 약점

| | |
|---|---|
| **Simple GENROU의 X''_d ≠ X_l 비일치** | E_fd 자동 override로 우회 (Stage 6) — 본격 대응은 γ-corrected GENROU 필요 |
| **Stator transient 무시** | 60 Hz 근처는 정확, 수 kHz는 부정확 (full Park machine 필요) |
| **AVR/Governor 없음** | E_fd, T_m이 상수 → 외란 후 V/ω 자동 회복 안 됨 |
| **명시적 Euler** | rotor 상태 적분에 작은 누적 오차 (10초당 ~0.1°) — 사다리꼴 implicit으로 개선 가능 |
| **Constant Z 부하만** | NR PF 비선형 부하 미구현 |
| **3상 평형만** | 상호 인덕턴스, 단상 고장 미지원 |
| **이벤트 시 numerical chatter** | 라인 트립 시 TRV 스파이크 — 차단기 모델(arc, snubber) + CDA 필요 |
| **Saturation 없음** | 정상 운전 영역 OK, 큰 외란 시 부정확 |

## 7. 한 스텝 데이터 흐름 다이어그램

```
시간 t_k                                                  
  │                                                       
  ▼                                                       
┌─────────────────────────────────────────────────────┐  
│ predict_state(Δt, t_k)                              │  
│  ├ Gen: δ_pred 외삽 → e''_abc_now 계산              │  
│  └ VS:  self._t = t_k 캐싱                          │  
└─────────────────────────────────────────────────────┘  
  │                                                       
  ▼  I_total = 0                                          
┌─────────────────────────────────────────────────────┐  
│ stamp_history_current(I_total)  (모든 컴포넌트)     │  
│  Line, Load:    ±I_hist (이력)                      │  
│  VS:            +V(t_k)·G_int (능동)                │  
│  Gen:           +G·(e''(t_k)+e''(t_k-Δt)) - I_hist  │  
└─────────────────────────────────────────────────────┘  
  │                                                       
  ▼                                                       
┌─────────────────────────────────────────────────────┐  
│  V_nodes(t_k) = LU.solve(I_total)                   │  
└─────────────────────────────────────────────────────┘  
  │                                                       
  ▼                                                       
┌─────────────────────────────────────────────────────┐  
│ update_branch(V_nodes)  (모든 컴포넌트)             │  
│  Line: trapezoidal로 i_*, v_C_* 갱신               │  
│  Load: trapezoidal로 i_branch, v_C 갱신            │  
│  Gen:  Park → algebraic stator → 6 ODE Euler 적분  │  
└─────────────────────────────────────────────────────┘  
  │                                                       
  ▼  self._t += Δt                                        
시간 t_{k+1}                                              
```

## 8. main_emt_3bus.py 진입점 흐름

현재 구성 — **Bus 2 = Generator 고정** (VS 모드는 `_archive/main_emt_3bus_VS.py`로 백업).

```
[모듈 상단 상수]
    SCENARIO   = 'load_step' | 'line_trip'
    T_EVENT    = 4 s
    DT         = 50 µs
    T_END      = 10 s
    LOAD_NEW   = {P: 8.0, Q: 4.0}              ◀ load_step 시 새 운전점 (10× 시나리오)
    GEN_PARAMS = {T_m: 0.8, E_fd: 1.0, Q_op: 0.2, D: 0.0}
    SAVE_MAT, MAT_FILENAME, MAT_STRIDE         ◀ .mat 저장 옵션

[main 흐름]
    sim, gen, line_2_3, load = build_simulator()
        ├ 라인 3개 (damping_L=0.05, damping_C=0.05)
        ├ Bus 1: VoltageSource (slack, R_int=0.001)
        ├ Bus 2: Generator (GEN_PARAMS)
        ├ Bus 3: Load (P=0.8, Q=0.4)
        └ sim.build()      ◀ G + LU 분해

    sim.initialize(verbose=True)   ◀ 위상자 fixed-point bumpless start
                                     verbose: 각 iter의 max|ΔV| 출력

    data = run_simulation(sim, gen, line_2_3, load)
        # 수동 step loop:
        for k in range(n_steps):
            time[k] = sim._t
            if (not event_done) and (sim._t >= T_EVENT):
                apply_event(sim, line_2_3, load)   ◀ SCENARIO에 따라 외란
                event_done = True
            sim.step()
            # 모든 모선 V, 회전자 상태(δ, Δω, ψ''_d, ψ''_q, e''_abc, T_e) 기록

    if SAVE_MAT:
        save_results_mat(data)         ◀ scipy.io.savemat (MATLAB 호환)

    plot_gen_scenario(sim, gen, data)  ◀ 2×3 subplot:
                                          (0,0) δ        (0,1) Δω       (0,2) T_e vs T_m
                                          (1,0) ψ''_d/q  (1,1) Bus 3 V  (1,2) Bus 2 V
```

**외란 처리** (`apply_event`):

| SCENARIO | 동작 |
|---|---|
| `'load_step'` | `load.set_operating_point(**LOAD_NEW)` + `sim.rebuild_G()` |
| `'line_trip'` | `line_2_3.set_active(False)` + `sim.rebuild_G()` |

`rebuild_G`는 G 재조립 + 재 LU 분해. 분기 전류·v_C 등 컴포넌트 내부 이력은 보존.

**MATLAB 측에서 후처리** (선택):

```matlab
load('emt_results.mat');
plot(time, rad2deg(delta));
plot(time, V_bus2');   % 3 × n 행렬, 각 행이 phase a/b/c
```

## 핵심 한 줄 요약

> **각 컴포넌트가 자기 사다리꼴 등가 모델 알고**, simulator는 **G에 stamping → LU 분해 → 매 스텝 RHS 갱신 → solve**만 반복. 발전기는 **predict (외삽) + update (적분)** 으로 비선형 동특성 처리. Bumpless는 **위상자 fixed-point 반복**으로 자기 일관성 확보.
