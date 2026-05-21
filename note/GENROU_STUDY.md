# GENROU 6th Order 동기기 모델 — 학습 자료

이 문서는 3-bus EMT 시뮬레이터 프로젝트에서 사용하는 동기기 모델의 핵심 정리입니다.
웹 Claude 채팅에서 이어서 대화하거나 깊이 있는 질문을 할 때 컨텍스트로 사용할 수 있습니다.

---

## 프로젝트 컨텍스트

3 모선 EMT 시뮬레이터를 객체별로 분리하여 점진적으로 구축한 프로젝트.
Park 변환, 사다리꼴 이산화, GENROU 6th order 동기기, bumpless start까지 구현 완료.
ParaEMT 비교 가능한 baseline을 목표로 함.

**구성**:
- bus 1: 무한 모선 전압원 (slack)
- bus 2: GENROU 6th 동기기
- bus 3: Constant-Z 부하

**검증 결과**:
- 부하 step 응답에서 자연 진동수 1.78 Hz 측정 (이론값 1.77 Hz와 0.5% 일치)
- Bumpless start로 첫 스텝부터 정상상태 사인파 (잔차 ~1e-6 수준)
- 댐핑 비율 ζ=0.135 (D=2 + 댐퍼 권선 효과 10배 추가)

---

## 큰 그림: 동기기란 무엇인가

```
        ┌─────────────────────┐
        │      ROTOR (회전자)  │  ─── 회전 (ω ≈ ω_0)
        │  [field winding fd] │  ←── 직류 E_fd 인가
        │  [damper kd, kq]    │  ←── 단락 (induced current만)
        └─────────────────────┘
                    │ 자속 결합 (mutual inductance)
        ┌─────────────────────┐
        │     STATOR (고정자)  │  ─── 정지
        │  [a, b, c 권선]     │  ←─→ 외부 3상 네트워크
        └─────────────────────┘
```

**동작 원리**: 회전자의 직류 자속이 회전하면서 stator 권선에 60Hz EMF 유도 → 외부 네트워크로 전력 송출.

---

## 좌표계 두 개 — abc vs dq

| | abc 좌표 (정지) | **dq 좌표 (회전)** |
|---|---|---|
| 위치 | 외부 네트워크 (line, load) | 회전자 내부 |
| 정상상태 | 60Hz 사인파 | **DC 일정값** |
| 변환 | Park (forward) | Park inverse |

**Park 각도**: $\theta = \omega_0 t + \delta$
- $\omega_0 t$: 동기 회전 (60Hz)
- $\delta$: 회전자가 동기 기준에서 얼마나 앞서/뒤져 있는지

→ **동기기 식의 거의 모든 부분은 dq 좌표에 살고**, 외부 인터페이스만 abc 변환.

**사용 컨벤션**: Kundur amplitude-invariant (2/3 scaling), $V_{dq} = V_{phasor} \cdot e^{-j\delta}$

---

## 3가지 시간 척도 (← 핵심 통찰)

GENROU의 식들은 서로 매우 다른 속도로 변하는 변수들을 다룹니다:

| 척도 | 시상수 | 변수 | 물리적 의미 |
|---|---|---|---|
| **Synchronous** | – (정상상태) | $X_d, X_q$ | 모든 transient 사라진 후 |
| **Transient** | $T'_{d0} \approx 8\,\text{s}$ | $E'_q, E'_d, X'_d, X'_q$ | 계자/q축 댐퍼 권선 효과 |
| **Sub-transient** | $T''_{d0} \approx 30\,\text{ms}$ | $\psi''_d, \psi''_q, X''_d, X''_q$ | 빠른 댐퍼 권선 (D, Q 댐퍼) |

→ 외란 후 시간 흐름:
```
0~수 ms          수 ms~수십 ms       수 초~수십 초
   │                 │                    │
sub-transient    transient            synchronous
빠르게 감쇠       천천히 감쇠           정상 도달
ψ"_d, ψ"_q       E'_q, E'_d            δ, V 안정
```

---

## 6개 상태 변수

### 전기적 (4개) — 회전자 자속/EMF 관련

| 변수 | 의미 | ODE 시상수 |
|---|---|---|
| $E'_q$ | q축 transient EMF (계자 자속 변환) | $T'_{d0}$ |
| $E'_d$ | d축 transient EMF (q축 첫 댐퍼) | $T'_{q0}$ |
| $\psi''_d$ | d축 sub-transient 자속 (느린 댐퍼) | $T''_{d0}$ |
| $\psi''_q$ | q축 sub-transient 자속 | $T''_{q0}$ |

> "프라임 (')은 transient, 더블프라임 ('')은 sub-transient" — 작은 따옴표 = 큰 시상수, 두 따옴표 = 짧은 시상수.

### 기계적 (2개) — 회전자 운동

| 변수 | 의미 | 단위 |
|---|---|---|
| $\delta$ | 회전자 각 (동기 기준 좌표계에서) | rad |
| $\Delta\omega$ | 속도 편차 ($\omega - \omega_0$) | pu |

---

## 식의 구조 — 4 + 2

### 회전자 4 electrical ODE (Simple GENROU)

$$T'_{d0} \dot E'_q = E_{fd} - E'_q - (X_d - X'_d)\, i_d$$

$$T''_{d0} \dot \psi''_d = E'_q - \psi''_d - (X'_d - X_l)\, i_d$$

$$T'_{q0} \dot E'_d = -E'_d + (X_q - X'_q)\, i_q$$

$$T''_{q0} \dot \psi''_q = -E'_d - \psi''_q - (X'_q - X_l)\, i_q$$

**해석**:
- d축: 외부 입력 $E_{fd}$ (계자 전압) → $E'_q$ (slow) → $\psi''_d$ (fast)
- q축: $E_{fd}$ 없음 (회전자에 q축 계자 없음) → $E'_d$ → $\psi''_q$
- 모두 $-i_d$ 또는 $-i_q$ 가 자속 감소 효과 (**armature reaction** = stator 전류가 회전자 자속을 줄임)

### Mechanical 2 ODE (Swing eq.)

$$\dot \delta = \omega_0 \cdot \Delta\omega$$

$$2H \cdot \dot{\Delta\omega} = T_m - T_e - D \cdot \Delta\omega$$

여기서:
- $T_m$: 외부 입력 (mechanical torque, governor 출력)
- $T_e = \psi''_d \cdot i_q - \psi''_q \cdot i_d$ (전기 토크, round rotor 가정)
- $H$: 관성 상수 (단위: s)
- $D$: 댐핑 계수

**해석**: $T_m > T_e$ 면 가속, $T_m < T_e$ 면 감속. $\delta$ 는 그 가속의 적분.

---

## Stator algebraic — 외부와 만나는 부분

GENROU 표준은 **stator transient 무시** ($d\psi_d/dt = d\psi_q/dt = 0$). dq에서 (Kundur 컨벤션, generator current OUT of stator):

$$v_d = -R_a i_d + X''_q i_q - \psi''_q$$

$$v_q = -R_a i_q - X''_d i_d + \psi''_d$$

**해석**:
- $-R_a i$: 저항 강하
- $X'' i$: 빠른 좌표 변환 (속도 EMF, $\omega \times \psi$)
- $\psi''$ 항: 회전자에서 stator로 유도된 sub-transient EMF

이게 **algebraic** (미분 없음)이라 dq에서는 즉각적 관계.

이를 stator 전류 $i_d, i_q$ 에 대해 풀면 (Cramer's rule):

$$i_d = \frac{R_a(-v_d - \psi''_q) + X''_q(\psi''_d - v_q)}{R_a^2 + X''_d X''_q}$$

$$i_q = \frac{R_a(\psi''_d - v_q) + X''_d(v_d + \psi''_q)}{R_a^2 + X''_d X''_q}$$

---

## EMT 인터페이스 — Thevenin 등가

stator 식을 정리하면 단자에서 본 등가 회로:

$$v^{\text{term}}_{abc}(t) = R_a \cdot i_{abc}(t) + L''_d \cdot \frac{di_{abc}}{dt} + e''_{abc}(t)$$

```
    네트워크 ←  i_abc
       │
       │
    ┌──┴──┐
    │ R_a │
    └──┬──┘
       │
    ┌──┴──┐
    │L''_d│
    └──┬──┘
       │     +
    ┌──┴──┐ ─ e''_abc(t)
    │     │ +
    └──┬──┘
       │
     ground
```

→ **R-L 직렬 + 시변 EMF**. 라인의 R-L 사다리꼴 이산화를 그대로 재사용 가능 (코드 디자인의 핵심).

여기서 $e''_{abc}(t)$ 는:
- dq에서 $e''_d = -\psi''_q$, $e''_q = +\psi''_d$ (round rotor)
- inverse Park로 abc 변환: $e''_{abc}(t) = \text{InvPark}(e''_{dq}, \theta)$

---

## 부호 규약 정리 (헷갈리기 쉬운 부분)

| 변수 | 방향 |
|---|---|
| $i_{term}$ (네트워크 코드) | 모선 → 발전기 INTO (load 방향) |
| $i_d, i_q$ (machine 코드) | 발전기 → 모선 OUT (gen 방향) |
| $i_{term} = -i_{gen}$ | 둘 사이 부호 반전 |

**결과적으로**:
- gen이 P를 네트워크에 공급 → $i_q > 0$ (machine), $i_{term}$의 q성분 < 0
- gen이 over-excited (Q 공급) → $i_d > 0$ (machine)

---

## 한 스텝의 데이터 흐름 (회전자 관점)

```
[솔브 직전] predict_state(dt, t):
   1. δ_pred = δ + Δt·ω_0·Δω  (선형 외삽)
   2. e''_dq = (-ψ''_q, +ψ''_d)  (현재 자속 사용, ω≈1 가정)
   3. e''_abc(t) = InvPark(e''_dq, θ_pred)
   4. 이걸로 Norton 전류원 (G_gen·e''_abc) → 네트워크에 주입

[네트워크 풀이] G·V = I → V_term_abc 결정

[솔브 직후] update_branch(V_nodes):
   1. V_term_abc → Park(θ_pred) → v_d, v_q
   2. Algebraic stator 풀이 → i_d, i_q (Cramer)
   3. 4 rotor ODE 우변 계산 (E_fd, T_m, ψ, E', i_d, i_q에서)
   4. Swing eq. 우변 계산 (T_e, T_m, Δω)
   5. 명시적 Euler로 6 상태 한 스텝 적분
   6. e''_abc_prev ← e''_abc_now (다음 스텝 history용)
```

---

## 시간 척도 분리의 의미 — 외란 응답 직관

**부하 step 발생 후**:

```
 0 ─────────── 30ms ──────────── 100ms ────── 1s ────── 8s
 │              │                  │           │         │
 즉시 응답      sub-transient    swing      transient   synchronous
                 감쇠 끝          peak      감쇠 시작     도달
                 ψ"_d 안정       δ swing      E'_q 변화   완전 정상
```

→ 우리 swing 응답 (1.78 Hz, 약 560 ms 주기) 이 **transient 시간 척도**에 해당. swing 동안 ψ"는 빠르게 따라가지만 E'는 거의 동결.

이게 swing 분석의 핵심 통찰: **swing 시간 척도에서는 X'_d (transient)** 가 유효 임피던스 — sub-transient는 이미 감쇠, transient는 거의 동결 ($T'_{d0}$=8s ≫ 1s).

---

## Bumpless start (정상상태 역산)

`sim.initialize()`에서 위상자 fixed-point 반복으로 회전자 6개 상태를 정상상태 값으로 자동 세팅:

```
V_term_p = V_phasor[gen_bus]
P_op = T_m,  Q_op = params['Q_op']

I_gen_p = (P_op - jQ_op) / V_term_p*

E_q_fic = V_term_p + (R_a + j·X_q)·I_gen_p
δ_ss = ∠E_q_fic - π/2          ← 부하각 도출

V_dq = V_term_p · e^(-jδ_ss),    I_dq = I_gen_p · e^(-jδ_ss)

# stator algebraic에서 ψ'' 직접 backsolve (자동 일관성)
ψ''_q = -R_a·i_d + X''_q·i_q - v_d
ψ''_d = v_q + R_a·i_q + X''_d·i_d

# rotor ODE 정상상태 조건에서 E'_q, E'_d, E_fd backsolve
E'_q = ψ''_d + (X'_d - X_l)·i_d
E_fd = E'_q + (X_d - X'_d)·i_d   ← 사용자 입력 OVERRIDE
E'_d = -ψ''_q - (X'_q - X_l)·i_q
```

**왜 E_fd override?**
- Simple GENROU의 $\psi''_d$ ODE는 $X'_d - X_l$ 사용
- 물리적으로는 $X_d - X''_d$ 가 맞음
- $X''_d \neq X_l$ 일 때 두 식이 어긋남 → stator algebraic이 만족 안 됨
- **자동 보정**: 사용자 E_fd 무시하고 운전점에 맞는 값으로 강제

---

## 외부 입력 (현재 상수)

| 변수 | 정상 출처 | 우리 코드 |
|---|---|---|
| $E_{fd}$ | AVR (Automatic Voltage Regulator) | 상수 (bumpless에서 override) |
| $T_m$ | Governor (mechanical) | 상수 |

---

## Simple GENROU의 단순화

| 단순화 | 실제 GENROU | 영향 |
|---|---|---|
| **No γ-correction** | $\gamma_d = (X'_d - X''_d)/(X'_d - X_l)^2$ 가 ψ ODE에 추가 | $X''_d \neq X_l$ 일 때 정상상태 부정확 → E_fd auto-override로 대처 |
| **Stator algebraic only** | $d\psi_d/dt, d\psi_q/dt$ 항 추가 (8th order) | 60 Hz 근처 OK, 수 kHz 부정확 |
| **No saturation** | $S(1.0), S(1.2)$ 항 | 정상 영역 OK, 큰 외란 시 부정확 |
| **Round rotor** | $X''_d \neq X''_q$ (살리언트) | T_e cross term 무시 |
| **Explicit Euler 적분** | Trapezoidal implicit | 작은 누적 오차 (10초당 ~0.1°) |

---

## 사용 파라미터 (Kundur Example 4.1, 555 MVA round-rotor)

| 그룹 | 파라미터 | 값 |
|---|---|---|
| 저항 | $R_a$ | 0.003 p.u. |
| Synchronous | $X_d, X_q$ | 1.81, 1.76 |
| Transient | $X'_d, X'_q$ | 0.30, 0.65 |
| Subtransient | $X''_d (=X''_q)$ | 0.23 |
| Leakage | $X_l$ | 0.15 |
| 시상수 | $T'_{d0}, T'_{q0}$ | 8.0 s, 1.0 s |
|       | $T''_{d0}, T''_{q0}$ | 0.03 s, 0.07 s |
| 기계 | $H$ | 3.5 s |
| 댐핑 | $D$ | 0.0 (외란 응답 시 2.0) |

---

## 검증 결과 (방향 B: 외란 응답)

**부하 50% step 시나리오** (P 0.8→1.2):
- 자연 진동수 측정 = 1.78 Hz
- 이론 $\omega_n = \sqrt{\omega_0 K_s/(2H)}$ 에서:
  - $K_s = E'_q V/X'_d \cdot \cos(\delta_{load}) \approx 0.97 \cdot 1.01/0.30 \cdot 0.7 \approx 2.29$
  - $\omega_n = \sqrt{377 \cdot 2.29/7} \approx 11.1$ rad/s → $f_n \approx 1.77$ Hz
- **0.5% 이내 일치** — GENROU의 sub-transient + transient 시간 분리가 정확히 작동

**댐핑 비율**:
- $\zeta_{theory} (D만)$ = $D/(2\sqrt{2H \omega_0 K_s}) = 0.013$
- 측정 ζ = 0.135 → **10배 큰 댐핑** ← 댐퍼 권선 효과 ($T''_{d0}, T''_{q0}$ 통한 추가 dissipation)

---

## 한 페이지 cheat sheet

```
┌──────────────────────────────────────────────────────┐
│ GENROU 6th order — One-page summary                  │
├──────────────────────────────────────────────────────┤
│ 상태: δ, Δω, E'_q, E'_d, ψ''_d, ψ''_q                 │
│ 입력: E_fd (AVR), T_m (Gov)                          │
│ 출력: v_d, v_q ↔ i_d, i_q (algebraic stator)         │
│                                                      │
│ Stator (algebraic, ω≈1, Kundur):                     │
│   v_d = -R_a·i_d + X''_q·i_q - ψ''_q                 │
│   v_q = -R_a·i_q - X''_d·i_d + ψ''_d                 │
│                                                      │
│ Rotor (4 ODE):                                       │
│   T'_d0 Ė'_q   = E_fd - E'_q - (X_d-X'_d)·i_d       │
│   T''_d0 ψ̇''_d = E'_q - ψ''_d - (X'_d-X_l)·i_d      │
│   T'_q0 Ė'_d   = -E'_d + (X_q-X'_q)·i_q             │
│   T''_q0 ψ̇''_q = -E'_d - ψ''_q - (X'_q-X_l)·i_q     │
│                                                      │
│ Swing (2 ODE):                                       │
│   δ̇ = ω_0·Δω                                        │
│   2H·Δω̇ = T_m - T_e - D·Δω                          │
│   T_e = ψ''_d·i_q - ψ''_q·i_d                        │
│                                                      │
│ External: e''_abc = InvPark(-ψ''_q, +ψ''_d, 0; θ)    │
│           Z_terminal = R_a + jωL''_d                  │
│                                                      │
│ Time scales:                                         │
│   sub-transient ~30 ms (ψ''_d, ψ''_q)                │
│   transient     ~8 s   (E'_q, E'_d)                  │
│   swing         ~1 s   (δ — between sub & trans)     │
└──────────────────────────────────────────────────────┘
```

---

## 학습 추천 순서

1. **회로 이해** (15분): rotor와 stator의 자속 결합, 왜 dq를 쓰는지
2. **Park 변환** (30분): forward/inverse, 정상상태에서 DC 되는 이유
3. **Synchronous (정상상태) 식** (30분): $V_q = E_{fd} - X_d \cdot i_d$ 같은 단순 형태부터
4. **Transient 추가** (1시간): $E'_q$ ODE, $X'_d$ 도입, $T'_{d0}$ 시상수
5. **Sub-transient 추가** (1시간): $\psi''_d$ ODE, $X''_d$, $T''_{d0}$
6. **Swing eq.** (30분): 관성, 댐핑, $T_e$ 표현
7. **EMT 인터페이스** (1시간): 단자 Thevenin 등가, sub-transient 인덕턴스 역할

**추천 교재**:
- **Kundur "Power System Stability and Control"** Ch. 3, 4 (가장 표준)
- **Sauer-Pai "Power System Dynamics and Stability"** (식 정리 깔끔)
- **Anderson-Fouad "Power System Control and Stability"** (실용 지향)

---

## 깊이 있는 질문 (다음 학습 토픽)

이 모델로 더 공부하고 싶은 주제들:

1. **γ-correction 유도**: 왜 simple GENROU가 $X''_d \neq X_l$ 일 때 부정확한지, PSS/E exact form은 어떻게 보정하는지
2. **Stator transient 포함 (8th order)**: $d\psi_d/dt, d\psi_q/dt$ 항을 다시 살리면 어떻게 되는지, EMT에서 왜 필요한지
3. **Saturation 모델링**: $S(1.0), S(1.2)$ 가 식 어디에 들어가는지, 비선형 자속의 영향
4. **Salient pole rotor** ($X''_d \neq X''_q$): cross term이 swing equation에 어떻게 나타나는지
5. **AVR/Governor 추가**: IEEET1, TGOV1 등 표준 모델, 폐루프 응답
6. **Single-phase fault, 비대칭 운전**: 0-축 활성화, 음성분 전류 분석
7. **Multi-machine 시스템**: 여러 발전기 간 inter-area oscillation, 모드 분석
8. **Park 변환의 다양한 컨벤션**: power-invariant vs amplitude-invariant, IEEE vs Kundur

---

## 코드 위치 (참고)

이 모델의 실제 구현은 다음 파일에 있습니다:
- `EMT_TEST/EMT_3bus/generator.py` — Generator 클래스
- `EMT_TEST/EMT_3bus/park.py` — Park 변환 유틸리티
- `EMT_TEST/EMT_3bus/simulator.py` — 반복 phasor solve

각 식이 코드의 어디에 있는지 1:1 매핑 가능합니다.
