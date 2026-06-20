# 3-bus EMT Simulation — Project State

3 모선 EMT 시뮬레이터를 객체별로 분리해 점진적으로 구축. PSS/E 스타일 GENROU 6th 동기기까지 구현 완료.
ParaEMT 비교 가능한 baseline을 목표로 함.

## 디렉터리 구조

```
EMT_TEST/
├── main_emt_3bus.py            진입점 (실행: `py main_emt_3bus.py`)
├── CLAUDE.md                   이 파일 — 프로젝트 상태, 진행 사항
├── ALGORITHM_REVIEW.md         코드 알고리즘 / 데이터 흐름 정리
├── GENROU_STUDY.md             동기기 모델 학습 자료 (웹 Claude 채팅 컨텍스트용)
├── EMT_3bus/                   패키지
│   ├── __init__.py
│   ├── base_component.py       통일 인터페이스 + phasor_to_3phase_at helper
│   ├── bus_manager.py          ID → 0-indexed 슬롯 매핑 (14-bus 등 확장 대비)
│   ├── park.py                 abc ↔ dq0 변환 (Kundur 컨벤션, amplitude-invariant 2/3)
│   ├── voltage_source.py       이상 전압원 + R_internal (Norton 등가)
│   ├── transmission_line.py    pi-model + trapezoidal + set_active() 트립 메커니즘
│   ├── load.py                 Constant-Z (R/RL/RC) + set_operating_point()
│   ├── generator.py            GENROU 6th (simple form, Stage 1-6)
│   └── simulator.py            ParaEMTSimulator (build/initialize/step/run/rebuild_G)
├── 3_busEMT.py                 옛 절차형 참조 (덮어쓰지 말 것)
└── test_EMT/                   별도 실험 폴더
```

`EMT_3bus/` 안의 모듈은 직접 실행 ❌ (상대 임포트). 실행은 `main_emt_3bus.py`로.

## 컴포넌트 인터페이스 (BaseComponent)

매 EMT 스텝 단위:
```
predict_state(dt, t)         → 솔브 직전. t 캐싱, 비선형/시변 컴포넌트 외삽
stamp_G(G_matrix)            → G에 등가 컨덕턴스 누적
stamp_history_current(I)     → 우변 I에 이력/EMF Norton 주입
update_branch(V_nodes)       → 솔브 결과로 분기 전류·내부 상태 갱신
```

위상자 정상상태 (bumpless start):
```
stamp_Y_phasor(Y, ω)         → 60Hz 복소 admittance 누적
stamp_I_phasor(I_ph, ω)      → Norton 위상자 주입 (능동 소자)
initialize_states(V_phasor, ω, dt)  → 위상자에서 t=-Δt 인스턴스 값 backsolve
```

## 핵심 컨벤션

| 항목 | 선택 |
|---|---|
| Park 변환 | Kundur (amplitude-invariant 2/3 scaling), `V_dq = V_phasor·e^(-jδ)` |
| 위상자 | **Peak amplitude**, 3-phase p.u.로 `S = V·I*` (RMS 변환 없이) |
| 전류 부호 | i_term: 모선→발전기 INTO (load convention). i_d/i_q (gen convention, OUT of stator) |
| Stator | sub-transient algebraic (no stator transient): v_d = -R_a·i_d + X''_q·i_q - ψ''_q, v_q = -R_a·i_q - X''_d·i_d + ψ''_d |
| GENROU | **Simple form** (no γ-correction, no saturation), explicit Euler 6 ODE |
| 회전자 각 | θ_Park = ω_0·t + δ |
| 전기 토크 | T_e = ψ''_d·i_q - ψ''_q·i_d (round rotor 가정 X''_q = X''_d) |

## 알려진 결함/제약

1. **Simple GENROU는 X''_d = X_l 일 때만 stator algebraic과 ODE가 자동 일치**. Kundur 4.1 디폴트는 X''_d=0.23 ≠ X_l=0.15. 
   → Stage 6에서 `initialize_states`가 ψ''를 stator algebraic으로 backsolve하고 E_fd를 자동 override해서 일관성 확보. **사용자 입력 E_fd값은 무시됨**.
2. **stator transient 무시** (표준 GENROU). 60Hz 근처는 정확하지만 수 kHz 트랜지언트는 부정확.
3. **Saturation off**, **AVR/Governor 없음** — E_fd, T_m 상수.
4. ~~**명시적 Euler** 적분 (rotor 6 ODE). 작은 잔류 drift 가능 (2초당 ~1e-3 수준).~~ → **G_equiv 방식으로 해결** (아래 참고). drift 1.2e-5 rad 이하.
5. **부하는 Constant Z만**. Constant PQ나 NR PF는 미구현.

## Bumpless Start — G_equiv 방식 (2026-06-08 개선 완료)

**문제**: phasor NR이 복소수 Y_gen = 1/(Ra + jX''_d) 을 사용했으나, EMT 스텝은 Schur-reduced 실수 G_av = 1/R_av 를 사용 → 두 등가회로 불일치 → pre-event drift 2.5e-3 rad.

**해결**: `stamp_Y_phasor` / `stamp_I_phasor`를 EMT G 행렬과 동일한 G_av 기반으로 변경.

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| `stamp_Y_phasor` | `Y_gen = 1/(Ra+jX''_d)` (복소) | `G_av = 1/R_av` (실수, alpha_mac 포함) |
| `stamp_I_phasor` | `Y_gen × e''_phasor` | `I_gen + G_av × V` (G_av 항 소거) |
| P_op 계산 | `P_op = T_m` | `P_terminal = T_m - Ra·\|I_gen\|²` (Ra·I² 보정) |

**결과**:

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| pre-event δ drift | 2.5e-3 rad/s | **1.2e-5 rad/s** |
| T_e_init | 0.803 (T_m 초과) | **0.8000** (T_m 정확 일치) |
| NR 수렴 | 5회 | **3회** |

**핵심 원리**: G_av(Schur) = (Rd_red + Rq_red)/2 ≈ 24.18 pu → G_av ≈ 0.041 pu. Phasor NR의 Y_gen(복소 −j4.35)과 전혀 다른 값이었음. alpha_mac = 99/101 (수치 댐핑)이 포함된 실수 컨덕턴스를 사용해야 EMT와 일관성 확보.

---

## Stage 진행 (모두 완료, 검증됨)

| Stage | 기능 | 검증 포인트 |
|---|---|---|
| 1 | `park.py` abc↔dq0 | 평형 60Hz → DC, 라운드트립 ε 수준 |
| 2 | Generator 골격 (파라미터/상태 변수) | sim.add OK |
| 3 | stamp_G + 단자 R-L (passive shunt) | G 대각 변화량 = G_gen_pure |
| 4 | rotor 6 ODE 명시적 Euler | 수치 안정 (NaN/Inf 없음) |
| 5 | predict 외삽 + EMF Norton | e''_abc 활성, T_e가 T_m에 가까이 |
| 6 | bumpless (위상자 → 회전자 backsolve, 반복 수렴) | drift 6 차수 감소, T_e=T_m |
| **A** | main_emt_3bus.py: gen 추가 + 6패널 플롯 | 깨끗한 60Hz 사인파, ψ''/δ 정상 일정 |
| **B** | 외란 응답: 부하 step + swing 시각화 | f_n=1.78 Hz (이론 1.77 Hz와 0.5% 일치), ζ=0.135 (댐퍼 권선 효과) |

**main_emt_3bus.py 현재 구성** (PSCAD 비교 기준):
- bus 1 = VoltageSource (slack, V_mag=1.0, V_angle=0°, R_int=0.00104, **cos(ωt) 기준**)
- bus 2 = Generator (T_m=0.8, E_fd=2.0, Q_op=0.2, D=0.0)
- bus 3 = Load (P=0.8, Q=0.4, Constant-Z)
- `SCENARIO`: `'load_step'` / `'line_trip'` / `'load_step_temp'` / `'line_trip_temp'`
- T_END=20s, T_EVENT=10s, DT=50µs
- `TEMP_DURATION = 20×DT` (temp 시나리오 복귀 시간)

**main 구조**:
```
build_simulator()  → sim, gen, line_2_3, load  (+ sim.build() 자동)
apply_event()      → SCENARIO에 따라 외란 적용 + rebuild_G
clear_event()      → temp 시나리오 복귀 + rebuild_G
run_simulation()   → 수동 step loop, dict 반환
plot_gen_scenario() → 2×3 서브플롯 (Δδ / ω / Te / ψ'' / V_abc×2)
save_results_mat() → .mat 저장 (delta_dev 포함)
```

**플롯 컨벤션** (PSCAD 비교 기준, 2026-06-08 정리):
- (0,0) **Δδ = δ(t) − δ(0)** [rad] — PSCAD rotor angle과 직접 비교 가능
- (0,1) **ω** [rad/s] = (omega_dev + 1) × 2π×60
- (0,2) Te [pu] vs T_m
- (1,0) ψ''_d, ψ''_q [pu]
- (1,1) Bus 3 V_abc, (1,2) Bus 2 V_abc
- **시간축**: s (초), **y축**: plain 숫자 (scientific notation 제거)

**외란 처리 메커니즘**:
- `Load.set_operating_point(P, Q)` — 운전점 변경 (이력 보존, 계수만 재계산, kind 변경 불가)
- `TransmissionLine.set_active(bool)` — 라인 트립/재투입 토글 (모든 stamping no-op)
- `Simulator.rebuild_G()` — G 재조립 + 재 LU 분해
- 사용 패턴: 수동 sim.step() loop에서 t≥t_event 시점에 두 메서드 호출

**시나리오 비교** (T_EVENT=0.5s, T_END=5s):

| | Load step (P 0.8→1.2) | **Line trip (line 2-3)** |
|---|---|---|
| Bus 2 V 스파이크 | 1.010 → 1.005 (−0.4%) | **1.010 → 1.424** (+42% TRV) |
| Bus 3 V 스파이크 | 0.990 → 0.980 (−1%) | **0.990 → 1.361** (+37%) |
| δ swing range | 0.4° | **2.12°** (5배) |
| 정착 후 δ | −42.7° | −41.95° |

→ Line trip은 토폴로지 급변으로 **차단기 개방 TRV** (전형적 EMT 트랜지언트) 발생. 보호 협조 연구의 핵심.

**Swing 응답 검증 결과** (50% 부하 step):
- 측정 f_n = 1.78 Hz (이론 1.77 Hz와 0.5% 일치 ✓)
- 핵심: **swing 시간 스케일에서는 X'_d (transient)** 가 유효 임피던스 — sub-transient는 이미 감쇠, transient는 거의 동결 ($T'_{d0}$=8s ≫ 1s)
  - K_s ≈ E'_q·V/X'_d · cos(δ_load) ≈ 0.97·1.01/0.30·0.7 ≈ 2.29
  - ω_n = √(ω_0·K_s/(2H)) = √(377·2.29/7) ≈ 11.1 rad/s
- 댐핑 ratio ζ = 0.135 — D=2만으론 0.013 → **댐퍼 권선** ($T''_{d0}, T''_{q0}$)이 10배 추가 댐핑
- δ swing 작음 (0.4°) — 부하가 bus 3, gen이 bus 2라 slack(bus 1)이 추가 부하 대부분 흡수

## 자주 혼동되는 개념 (정리)

**Q. ψ''_d, ψ''_q가 외란 없을 때 일정한 이유?**
- 라운드 로터나 AVR 부재 때문 ❌
- **dq 좌표 + 정상상태** 때문 ✓
- dq 프레임은 회전자와 동승하므로 abc의 60Hz 회전이 자동 상쇄 → 정상상태 = DC
- 외란(부하 step, 단락, T_m 변화)이 들어오면 dq 프레임에서도 흔들림 보임

**Q. AVR/Gov가 있으면 ψ''_d, ψ''_q가 변하나?**
- 외란이 없으면 ❌ — 이미 균형이라 컨트롤러도 가만히 있음
- 외란이 있을 때 ✓ — 컨트롤러가 E_fd, T_m을 시간에 따라 바꾸므로 ψ도 따라 변함

**Q. Constant-Z 부하에서 진정한 voltage collapse 발생?**
- ❌ — V 떨어지면 I = V/Z도 비례 감소 → 자기 안정화 → V→0 점근하지만 절대 0 아님
- 실제 collapse 보려면 **Constant-PQ 부하** 필요 (V 떨어지면 I 증가 → positive feedback → 발산)
- 우리 10× load 시나리오에서 |V_bus3|=0.83 (정상 grid면 alarm/UVLS 영역)이지만 시뮬엔 안정 정착

**Q. 작은 외란에서 V 거의 안 변하는 이유?**
- Slack 강도 (R_int=0.001) + Z_line/Z_load 비율 작음 (~5%) → voltage divider 효과 미미
- 부하 10× step 시 비율 ~45%로 커지며 V 16% 강하 발생 (voltage divider 가시화)

## 핵심 수식 (참고용)

**Trapezoidal R-L Norton** (라인·부하·발전기 단자에서 공통):
- $G = 1/(R + 2L/\Delta t)$, $\alpha = (2L/\Delta t - R)/(2L/\Delta t + R)$
- $i(t) = G \cdot v(t) + I_\text{hist}$,  $I_\text{hist} = G \cdot v(t-\Delta t) + \alpha \cdot i(t-\Delta t)$

**GENROU 6 ODE (simple form)**:
- $T'_{d0}\dot E'_q = E_{fd} - E'_q - (X_d - X'_d) i_d$
- $T''_{d0}\dot \psi''_d = E'_q - \psi''_d - (X'_d - X_l) i_d$
- $T'_{q0}\dot E'_d = -E'_d + (X_q - X'_q) i_q$
- $T''_{q0}\dot \psi''_q = -E'_d - \psi''_q - (X'_q - X_l) i_q$
- $\dot \delta = \omega_0 \Delta\omega$
- $2H\dot{\Delta\omega} = T_m - T_e - D \Delta\omega$

**수치 댐핑** (이미 구현):
- $R_d^L$ ‖ inductor: `damping_L = α` → $R_d = (2L/\Delta t)/\alpha$
- $R_d^C$ + capacitor 직렬: `damping_C = β` → $R_d = \beta \cdot \Delta t/C$
- 권장 0~0.1, 기본 0 (꺼짐)

## PSCAD 비교 — 각도 컨벤션 (2026-06-08 분석 완료)

**핵심 차이**:

| 항목 | 우리 코드 | PSCAD |
|---|---|---|
| rotor angle 출력 | 절대값 δ = −0.7602 rad | **Δδ = δ(t) − δ(0)** (항상 0 시작) |
| 전압 파형 기준 | **cos(ωt)** (V_a(0)=1.0, 피크 시작) | **sin(ωt)** (V_a(0)=0, 영교차 시작) |
| V_abc 위상차 | — | **90° = 4.17 ms** 차이 |

**PSCAD Wang 공식**: `Wang = θ_V_terminal + δ_0 + ∫(ω_pu − 1) dt`
- PSCAD가 플롯에 출력하는 "rotor angle" = 마지막 항 `∫(ω_pu − 1) dt` = Δδ
- cos/sin 기준 차이로 절대 δ값은 π/2 offset 있음, **Δδ 비교에서는 소거됨**

**비교 방법**:
- δ 비교: `delta_dev = delta - delta[0]` (코드에 적용됨, mat 저장에도 포함)
- V_abc 비교: PSCAD 파형을 4.17ms (= T/4) 시프트하거나 우리 코드 V_angle_deg = −90° 설정

**Rotor Mechanical Angle** (PSCAD 별도 출력): 0~2π sawtooth — 물리적 로터 위치, swing 비교와 무관.

---

## PSCAD 환산 (Kundur 555 MVA, 24 kV base)

| 기준 | 값 |
|---|---|
| $S_{base}$ | 555 MVA (3-phase) |
| $V_{LL,base}$ | 24 kV (RMS) |
| $Z_{base}$ | 1.0378 Ω |
| $V_{LN,peak,base}$ | 19.596 kV |
| $I_{base}$ (line RMS) | 13,349 A |

**환산식**: $X_\text{actual}[\Omega]=X_{pu} \cdot Z_{base}$, $L[H]=X_\text{actual}/\omega$, $C[F]=B_{pu}/(\omega \cdot Z_{base})$

**Line 1-2** (R=0.010, X=0.05, B=0.10 pu): R=10.4 mΩ, L=0.138 mH, C_total=255 µF (C/2=128 µF 양 끝)
**Line 2-3** (R=0.020, X=0.06, B=0.12 pu): R=20.8 mΩ, L=0.165 mH, C_total=307 µF
**Line 3-1** (R=0.015, X=0.04, B=0.08 pu): R=15.6 mΩ, L=0.110 mH, C_total=204 µF

**부하 (P=0.8, Q=0.4 pu)**: P=444 MW, Q=222 MVAr; R=1.038 Ω, L=1.376 mH (3상 Y-equivalent per-phase)
**전압원 (Bus 1)**: V_LL=24 kV RMS, R_int=1.04 mΩ
**발전기**: PSCAD 동기기 모델은 pu 값 그대로 입력 (자체 pu 시스템). $E_{fd}$=2.04 pu (bumpless override 값) 강제.

자세한 PSCAD 회로 구성 정보 → 다음 세션에서 PSCAD 비교 시 참조.

## 다음 가능 방향 (TODO)

| | |
|---|---|
| ~~A~~ | ~~main_emt_3bus.py에 발전기 + 시각화로 깨끗한 60Hz 파형 확인~~ ✓ 완료 |
| ~~B~~ | ~~외란 응답 (부하 step) → swing 응답~~ ✓ 완료 (이론과 0.5% 일치) |
| ~~B+~~ | ~~SOURCE_TYPE 분기 + 10× 부하 시나리오~~ ✓ 완료 |
| ~~Bumpless~~ | ~~G_equiv 방식 bumpless start~~ ✓ 완료 (drift 2.5e-3 → 1.2e-5, T_e=T_m) |
| ~~PSCAD 컨벤션~~ | ~~각도/각속도 컨벤션 분석~~ ✓ 완료 (Δδ 기준, cos/sin 정리) |
| **PSCAD 비교** | **진행 중** — load_step / line_trip 파형 수치 비교 (Δδ, ω, Te) |
| **C** | AVR (IEEET1) / Governor (TGOV1) — 폐루프 제어, V/ω 회복 |
| D | γ-corrected GENROU (PSS/E exact, E_fd override 안 해도 됨) |
| E | Constant PQ 부하 + NR PF 모듈 → 진정한 voltage collapse 시뮬 가능 |
| F | Multi-machine (SMIB → multi) |

**다음 세션 시작 시 추천 진입점**:
1. **PSCAD 파형 수치 비교** — Δδ / ω / Te 정량적 일치 확인 (load_step, line_trip)
2. **C (AVR/Gov)** — IEEET1 + TGOV1 도입
3. **E (Constant PQ)** — voltage collapse, NR PF 도입

**부속 문서**:
- [`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md) — 알고리즘 흐름, 데이터 흐름, 한 스텝 도식
- [`GENROU_STUDY.md`](GENROU_STUDY.md) — 동기기 모델 학습 자료 (웹 Claude 채팅 컨텍스트)

## 실행 / 테스트

```powershell
PS C:\study\PSL\ParaEMT\ParaEMT_public-main\EMT_TEST> py main_emt_3bus.py
```

스모크 테스트 패턴 (각 Stage 끝나고 사용):
```python
import sys; sys.path.insert(0, 'c:/study/PSL/ParaEMT/ParaEMT_public-main/EMT_TEST')
for m in list(sys.modules):
    if m.startswith('EMT_3bus'):
        del sys.modules[m]
from EMT_3bus import ...
```
(stale `__pycache__` 회피)
