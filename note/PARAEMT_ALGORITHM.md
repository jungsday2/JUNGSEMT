# ParaEMT — 전체 알고리즘 정리

NREL이 공개한 **ParaEMT** (Parallelizable EMT Simulator)의 EMT 시뮬레이션 알고리즘을 정리한 문서.
참조 폴더: [`../ParaEMT_public-main/`](../ParaEMT_public-main/)

> **범위 안내**
> - **EMT 기본 틀과 알고리즘**에 집중. BBD (Bordered Block Diagonal) 병렬 솔버는 한 섹션으로만 요약.
> - **`Lib_BW.py`** (3469줄) 와 **`lib_numba.py`** (2050줄)의 모든 클래스/메서드/함수를 매우 상세히 enumerate.
> - 우리 [`EMT_3bus/`](EMT_3bus/) 미니 구현과의 대응 관계를 마지막 섹션에서 비교.

---

## 1. 큰 그림 — 3단계 워크플로

```
[Step 0]  main_step0_powerflow.py    PSS/E의 NR PF 풀이 → pfd_*.json 저장
              │
              ▼
[Step 1]  main_step1_simulation.py   ← 핵심 (이 문서의 중심)
              │  ① initialize_emt()           : 위상자 PF → bumpless 초기화
              │  ② 시간 영역 EMT 루프         : ts=50µs, Tlen=10s 등
              │  ③ dump_res()                 : snapshot.pkl + result.pkl 저장
              ▼
[Step 2]  main_step2_saveresults.py  .pkl → CSV로 익스포트 (열별 라벨링)
```

Step 0의 PF는 **외부 PSS/E**에 위임 (`psspy.fnsl` — Fast Newton-Raphson Solution). 시간 영역 EMT는 PSS/E의 결과만 받아 별도로 풀이.

**핵심 입력 파일**:
- `cases/pfd_*.json` — PSS/E PF 결과 (Step 0 산출물)
- `models/<system>/*.xls(x)` — 동특성 데이터 (GENROU/SEXS/TGOV1/IEEEST/REGCA/…)
- snapshot `sim_snp_S*_*u_1pt.pkl` — 정상상태 재시작용 (선택)

지원 케이스: 2-gen, 9-bus, 39-bus, **WECC 179-bus**, **WECC 240-bus**, 2-area.

---

## 2. 디렉터리 구조

```
ParaEMT_public-main/
├── main_step0_powerflow.py         PSS/E PF 진입점
├── main_step1_simulation.py        EMT 메인 루프 ⭐
├── main_step2_saveresults.py       CSV 익스포트
│
├── Lib_BW.py                       핵심 라이브러리 ⭐ 3469줄
│   ├── PFData                      파워플로 데이터 컨테이너
│   ├── DyData                      동특성 모델 파라미터 (+ ToEquiCirData)
│   ├── Initialize                  bumpless 초기화 (PF → 시간 영역)
│   ├── EmtSimu                     메인 시뮬레이터 (step 로직 모음)
│   ├── States, States_ibr          pre-allocated 작업 버퍼
│   └── Lib_BW_CreateLargeCases.py  (별도 파일) 대형 케이스 생성
│
├── lib_numba.py                    Numba JIT inner loop 모음 ⭐ 2050줄
│                                   8개 함수 — predict, updateIg/Iibr, BusMea,
│                                              updateX/Xibr, updateIhis
│
├── psutils.py                      initialize_emt, initialize_from_snp 진입 helper
├── preprocessscript.py             JSON → storage 객체 변환
│
└── (BBD 병렬화 — 본 문서에선 13장에서 한 줄 요약)
    ├── partitionutil.py            METIS partitioning → BBD form
    ├── bbd_matrix.py               BBD 행렬 클래스
    └── serial_bbd_matrix.py        Schur complement LU
```

---

## 3. 데이터 모델

### 3.1 `PFData` — 파워플로/토폴로지 데이터 [Lib_BW.py:26-472]

PSS/E의 COM API (`psspy`)에서 데이터를 끌어오거나 JSON에서 deserialize.

**필드 그룹**:
| 그룹 | 어트리뷰트 |
|---|---|
| 시스템 | `basemva`, `ws = 2π·60` |
| 모선 | `bus_num`, `bus_type`, `bus_Vm`, `bus_Va`, `bus_kV`, `bus_basekV`, `bus_name` |
| 부하 | `load_id/_bus/_Z/_I/_P/_MW/_Mvar` (Z/I/P split = ZIP 분해) |
| IBR | `ibr_bus/_id/_MW/_Mvar/_MVA_base` (gen_mod∈{1,3}인 발전기에서 분리) |
| 동기기 | `gen_id/_bus/_S/_mod/_MW/_Mvar/_MVA_base/_status` |
| 라인 | `line_from/_to/_id/_P/_Q/_RX/_chg` |
| 변압기 | `xfmr_from/_to/_id/_P/_Q/_RX/_k` |
| 션트 | `shnt_bus/_id/_gb`, `shnt_sw_bus/_gb` (switched) |

**주요 메서드**:

#### `PFData.getdata(self, psspy)` [102-203]
- **목적**: PSS/E에서 모든 데이터 추출.
- **알고리즘**: `psspy.abusint`, `psspy.amachint`, `psspy.abrnint`, `psspy.atrnint`, `psspy.afxshuntint`, `psspy.aswshint` 호출. 비활성 발전기 제외. **`gen_mod ∈ {1,3}`인 항목을 IBR로 분리** (배열 뒤쪽부터 pop).

#### `PFData.LargeSysGenerator(self, ItfcBus, r, c)` [205-465]
- **목적**: 기본 케이스를 `r×c`번 복제해 대형 벤치마크 생성 (예: 16×Kundur).
- **알고리즘**: 모선/부하/션트/발전기/라인/변압기를 N=r·c번 복제 (bus ID에 `k·N_bus` 오프셋), secondary slack을 PV로 강등, 인접 블록 사이를 interface bus(`upb/rtb/dnb/lfb`)로 연결. 라인 X는 흐름 추정값 `X = Vf·Vt·sin(Δθ)/Pmin`에서 도출.

#### `PFData.load_from_json(storage)` [staticmethod, 28-33]
- **목적**: deserialized storage에서 새 `PFData` 인스턴스 생성.

### 3.2 `DyData` — 동특성 데이터 [Lib_BW.py:474-1177]

GENROU, SEXS, TGOV1/HYGOV/GAST, IEEEST, REGCA/REECB/REPCA, PLL, V-mag 측정 파라미터를 보관. **PSS/E 표준형 → 등가회로(EC) 파라미터 변환**을 담당.

**클래스 상수**:
```python
gov_model_map = {'GAST': 0, 'HYGOV': 1, 'TGOV1': 2}    # Numba 분기용 int 코드
```

**필드 그룹**:
| 모델 | 파라미터 (per device) | order |
|---|---|---|
| GENROU | `Td0p, Td0pp, Tq0p, Tq0pp, H, D, Xd, Xq, Xdp, Xqp, Xdpp, Xl, S10, S12, Ra, X0` | 18 |
| SEXS | `TA, TB, K, TE, Emin, Emax, TA_o_TB` | 2 |
| TGOV1 | `R, T1, T2, T3, Vmax, Vmin, Dt` | 3 |
| HYGOV | `R, r, Tr, Tf, Tg, VELM, GMAX, GMIN, TW, At, Dturb, qNL` | 5 |
| GAST | `R, T1, T2, T3, LdLmt, KT, VMAX, VMIN, Dturb` | 4 |
| IEEEST | `A1–A6, T1–T6, KS, LSMAX, LSMIN, VCU, VCL` | 10 |
| REGCA+REECB+REPCA | (수십 개) | 41 |
| Bus PLL+Vmag | `[ze, de, we, vt, vtm, dvtm]` | 6 |
| Load | `[ZL_mag, ZL_ang, PL, QL]` | 4 |

#### `DyData.getdata(self, file_dydata, pfd, N)` [765-1076]
- 10개 시트 (gen/exc/gov/pss/regca/reecb/repca/pll/vm/mea)의 XLS 워크북 파싱.
- 발전기-거버너 매칭: `pfd.gen_bus`+`pfd.gen_id`로 룩업해 `gov_type[gen_idx]` 셋팅.
- **주의**: `gov_type`은 `len = gov_n`이지만 gen-index로 인덱싱 → 모든 gen에 governor가 있어야 정의됨.

#### `DyData.spreaddyd(self, pfd, dyd, N)` [1079-1106]
- `LargeSysGenerator`로 N배 복제했을 때 dyd도 N배 복제.
- 스칼라는 `×N`, 배열은 N번 concat, `*_bus`는 `i·N_bus/N` 오프셋, `*_idx`는 `i·N_gen/N` 오프셋.

#### `DyData.ToEquiCirData(self, pfd, dyd)` [1109-1177] ⭐
- **목적**: PSS/E 표준형 파라미터 → **Kundur/Park 등가회로(EC)** 변환. EMT 솔버가 실제로 쓰는 형식.
- **알고리즘** (per 발전기):

  **Base 계산**:
  ```
  base_es = V_LL,base · √(2/3) · 1000          [V, peak L-N]
  base_is = MVA_base·1e6 / (base_es·3/2)        [A, peak]
  base_Is = base_is/√2                          [A, RMS]
  base_Zs = base_es/base_is                     [Ω]
  base_Ls = base_Zs·1000 / (2π·60)              [H]
  ```

  **자화/누설 인덕턴스**:
  ```
  Lad = Xd − Xl,    Laq = Xq − Xl,    Ll = Xl
  Ld  = Lad + Ll,   Lq  = Laq + Ll
  ```

  **필드 누설**:  $L_{fd} = \dfrac{(X'_d - X_l)·L_{ad}}{L_{ad} - (X'_d - X_l)}$

  **1q 댐퍼 누설**:  $L_{1q} = \dfrac{(X'_q - X_l)·L_{aq}}{L_{aq} - (X'_q - X_l)}$

  **1d 댐퍼**: $z = X''_d - X_l$, $y = L_{ad}·L_{fd}/(L_{ad}+L_{fd})$, $L_{1d} = yz/(y-z)$, $R_{1d} = (y+L_{1d})/T''_{d0}$

  **2q 댐퍼**: 대칭 → $L_{2q}, R_{2q}$

  **저항**:
  ```
  Rfd = (Lad+Lfd)/Td0p
  R1q = (Laq+L1q)/Tq0p
  ```

  **상호 인덕턴스**:
  ```
  Lf1d  = Lad
  Lffd  = Lad²/(Xd − Xdp)
  L11d  = Lad²/(Xd − Xdpp)
  L11q  = Laq²/(Xq − Xqp)
  L22q  = Laq²/(Xq − Xdpp)        ← 주의: Xqpp 아님
  ```

- **호출 시점**: 초기화 한 번. 이후 모든 EC 파라미터(`ec_*`)는 발전기 모델 코드에서 직접 참조.

---

## 4. 초기화 단계 — `Initialize` 클래스 [Lib_BW.py:2483-3469]

위상자 PF 결과를 시간영역 EMT의 t=0 상태로 변환 (**bumpless start**). 17개 메서드의 호출 순서가 중요:

```
ini = Initialize(pfd, dyd)             # 모든 어트리뷰트 0/빈 배열
ini.InitNet(pfd, ts, loadmodel_option)  # 네트워크 G 빌드 (numba_InitNet 위임)
ini.InitMac(pfd, dyd)                   # GENROU bumpless 시작값
ini.InitExc(pfd, dyd)                   # SEXS exciter 시작값
ini.InitGov(pfd, dyd)                   # TGOV1/HYGOV/GAST 시작값
ini.InitPss(pfd, dyd)                   # IEEEST 시작값 (모두 0)
ini.InitREGCA(pfd, dyd)                 # IBR generic 모델
ini.InitREECB(pfd, dyd)                 # IBR electrical control
ini.InitREPCA(pfd, dyd)                 # IBR plant control
ini.InitPLL(pfd)                        # IBR PLL
ini.InitBusMea(pfd)                     # bus 측정용 PLL+V-mag
ini.InitLoad(pfd)                       # constant-Z 부하 파라미터
ini.CheckMacEq(pfd, dyd)                # 12 algebraic 방정식 잔차 체크
ini.MergeMacG(pfd, dyd, ts, [], mode, nparts)  # 발전기 G_equiv stamping + LU/BBD 분해 ⭐
```

이후 `EmtSimu.preprocess(ini, pfd, dyd)`가 `ini.CombineX(pfd, dyd)`를 호출해 모든 초기값을 단일 state 벡터로 flatten.

### 4.1 `InitNet(self, pfd, ts, loadmodel_option)` [2679-2724]
- **목적**: 라인/변압기/부하/션트의 **사다리꼴 Norton 등가**를 G_0에 stamping. Base 양 + 위상자 시작값 + 초기 history 전류 모두 산출.
- **실제 작업**: `numba_InitNet(...)`에 위임 (자세한 내용은 §11.2).
- **출력**:
  - VbaseA = `√(2/3)·1000·kV_base` (peak L-N V)
  - ZbaseA = `kV²/basemva` [Ω]
  - StA = 발전기 복소 전력 (위상자)
  - Vt = `Vm·exp(j·Va)` (peak 위상자)
  - **N = 3·nbus** (3상 비결합 노드 솔브)
  - `Init_net_coe0`: 각 분기당 9-tuple `[Fidx, Tidx, Req, icf, Gv1, R, L, C, i_phasor]`
  - COO triplet `(rows, cols, data)` for G_0.

### 4.2 `InitMac(self, pfd, dyd)` [2726-2793] ⭐
- **목적**: GENROU의 표준 bumpless 시작 절차 (PSS/E 매뉴얼 5장과 동일).
- **알고리즘** (per 발전기):

  ```
  ① S = √(P² + Q²),   φ = asin(Q/S)
  ② IgA = It·IbaseA / (base_Is/1000)              # 머신 pu 단자전류
  ③ δ_internal = sign(P)·atan( (Lq|I|cosφ − Ra|I|sinφ) / (|V| + Ra|I|cosφ + Lq|I|sinφ) )
  ④ δ_t0 = δ_internal + V_angle                   # 절대 회전자 각
  ⑤ stator dq:    ed = |V|·sin(δ_int),   eq = |V|·cos(δ_int)
                  id = |I|·sin(δ_int+φ),  iq = |I|·cos(δ_int+φ)
  ⑥ 댐퍼 전류 0 (정상상태):  i1d = i1q = i2q = 0
  ⑦ Flux:   ψd = eq + Ra·iq,   ψq = -(ed + Ra·id)
  ⑧ Field:  ifd = (eq + Ld·id + Ra·iq) / Lad
            EFD = ifd·Lad                          # PSS/E EFD base
  ⑨ 로터 flux:  ψfd = Lffd·ifd − Lad·id
                ψ1d = Lad·(ifd − id)
                ψ1q = ψ2q = -Laq·iq
  ⑩ te = ψd·iq − ψq·id (≈ed·id + eq·iq 정상상태)
     qe = ψd·id + ψq·iq
     Init_mac_pref = te                            # governor 레퍼런스
  ```

### 4.3 `InitExc(self, pfd, dyd)` [2795-2804]
- **SEXS**: 정상상태 `v1 = EFD/K`, `vref = v1 + |Vt|`.

### 4.4 `InitGov(self, pfd, dyd)` [2806-2871]
- **TGOV1**: `pm = pref`, `gref = pref·R`, `p1 = p2 = pref`.
- **HYGOV**: `Tm0 = pref`, $q_0 = T_{m0}/A_t + q_{NL}$, $g_0 = q_0$, $n_{ref} = g_0·R$. `xc, xg, xq` 정상상태, `xe = 0`.
- **GAST**: `p1 = p2 = p3 = pref`, `gref = pref`.
- 부작용: gov→gen 매핑 (`tgov1_2gen`, `hygov_2gen`, `gast_2gen`) 채움.

### 4.5 `InitPss(self, pfd, dyd)` [2873-2885]
- **IEEEST**: 모든 상태 0 (washout/lead-lag 정상상태 = 0).

### 4.6 `InitREGCA(self, pfd, dyd)` [2918-2961]
- PF에서: `It = conj(S/Vt)`, `Ip_out = Re(It)·cos(Va) + Im(It)·sin(Va)`, `Iq_out = Im(It)·cos(Va) − Re(It)·sin(Va)`.
- HVPL: `i1 = max(0, (Vm − Volim)·Khv)`.
- LVPL gain: `i2 = 1` if Vm > Lvpnt1, 그 외 ramp.
- 시작값: `s0 = Ip_out/i2`, `s1 = -(Iq_out + i1)`, `s2 = Vm`.

### 4.7 `InitREECB(self, pfd, dyd)` [2963-2996]
- `s0 = Vm, s1 = P, s4 = Q/Vm, s5 = P`, `Pref = Ipcmd·Vm`, `Qext = Q`, `pfaref = atan(Q/P)`.

### 4.8 `InitREPCA(self, pfd, dyd)` [2998-3069]
- 분기 P/Q 측정 (또는 단자 Vt). VCFlag로 line-drop compensation 분기.

### 4.9 `InitPLL(self, pfd)` [3071-3078]
- IBR PLL: `ze=0, de=bus_Va, we=1.0`.

### 4.10 `InitBusMea(self, pfd)` [3080-3087]
- Bus PLL+Vmag: `ze=0, de=bus_Va, we=1, vt=vtm=bus_Vm, dvtm=0`.

### 4.11 `InitLoad(self, pfd)` [3089-3099]
- Constant-Z 부하: `PL, QL = MW, Mvar/basemva`, $|Z_L| = |V|²/|S|·basemva$, $\angle Z_L = atan(Q/P)$.

### 4.12 `CheckMacEq(self, pfd, dyd)` [2888-2916]
- **목적**: bumpless 시작값의 sanity check.
- **12개 algebraic 방정식**:
  ```
  eq[0]: ed + ψq + Ra·id = 0                    # stator d KVL
  eq[1]: eq − ψd + Ra·iq = 0                    # stator q KVL
  eq[2]: EFD·Rfd/Lad − Rfd·ifd = 0              # field 정상상태
  eq[3..5]: 댐퍼 전류 × 저항 = 0 (정상상태)
  eq[6..11]: flux-current Park 인덕턴스 행렬 관계
  ```
- SoS 잔차 > 1e-10이면 경고.

### 4.13 `MergeMacG(self, pfd, dyd, ts, i_gentrip, mode='inv', nparts=4)` [3101-3293] ⭐⭐
- **목적**: **초기화의 정점**. 모든 발전기의 3×3 abc Norton 등가를 G_0에 stamping → 최종 G factorization.
- **알고리즘** (per 발전기):

  **① d/q 축 3×3 인덕턴스 행렬**:
  - `Ld` = stator/field/1d 댐퍼 (3×3)
  - `Lq` = stator/1q/2q 댐퍼 (3×3)

  **② 사다리꼴 companion 저항 행렬** (수치 댐핑 α=99/101≈0.98):
  ```
  Rd1[i,j,k] = R_diagonal + (1+α)/(ts·ws) · Ld[i,j,k]      # stator 행
                          + (1+α)/ts        · Ld[i,j,k]    # field/damper 행
  Rd2[i,j,k] = R·α − (1+α)/(ts·...) · Ld[i,j,k]            # history 계수
  ```
  같은 식으로 `Rq1, Rq2`.

  **③ Schur reduction** — 필드+댐퍼 자유도를 stator로 reduce:
  ```
  Rd1[1:,1:]⁻¹                    = Rd1inv (2×2)
  Rd = Rd1[0,0] − Rd1[0,1:]·Rd1inv·Rd1[1:,0]         # 스칼라 reduced stator d-축 R
  Rd_coe = Schur multiplier row vector (updateIg에서 his_red_d 계산용)
  ```
  대칭으로 `Rq, Rq_coe`.

  **④ Round-rotor 평균**: `Rav = (Rd + Rq)/2`.

  **⑤ abc Norton 저항** (0-sequence 포함):
  ```
  R0 = Ra + (1+α)/(ts·ws)·L0
  Rs = (R0 + 2·Rav)/3              # 자기 저항
  Rm = (R0 − Rav)/3                # 상호 저항
  Requiv = [[Rs,Rm,Rm],
            [Rm,Rs,Rm],
            [Rm,Rm,Rs]]
  Gequiv = inv(Requiv) · ZbaseA / base_Zs      # base 환산
  ```

  **⑥ G_0에 stamping**: `addtoG0(genbus, genbus+N, genbus+2N)` 3×3 블록 (트립된 발전기는 skip).

  **⑦ 최종 G_0 (CSC sparse) 분해**:
  - `mode='inv'`: `la.inv(G0.tocsc())` (소형 시스템)
  - `mode='lu'`: `la.splu(G0.tocsc())` (중형 시스템) ⭐ 디폴트
  - `mode='bbd'`: `form_bbd(self, nparts)` + `schur_bbd_lu` → 병렬 (13장 참조)

- **호출 시점**: 초기화 1회. **gen trip 시 다시 호출** (트립된 발전기 제외하고 재분해).

### 4.14 `addtoG0(self, row, col, addedvalue)` [3295-3306]
- 기존 COO 엔트리에 누적하거나 새 엔트리 append. O(n) 룩업 — 발전기 stamping 한정으로 OK.

### 4.15 `CombineX(self, pfd, dyd)` [3308-3468]
- **목적**: 모든 초기값을 거대한 `Init_x` 벡터로 flatten. JIT 커널이 `_xi_st`(시작 오프셋) + `_odr`(차수)로 슬라이싱.
- **인덱싱 구조** (per 발전기):

| 서브시스템 | 차수 | 변수 |
|---|---|---|
| GENROU | 18 | δ, ω₀, id, iq, ifd, i1d, i1q, i2q, ed, eq, ψd, ψq, ψfd, ψ1q, ψ1d, ψ2q, te, qe |
| SEXS | 2 | v1, EFD |
| TGOV1 | 3 | p1, p2, pm |
| HYGOV | 5 | xe, xc, xg, xq, pm |
| GAST | 4 | p1, p2, p3, pm |
| IEEEST | 10 | y1..y7, x1, x2, vs |
| IBR | 41 | REGCA(8) + REECB(12) + REPCA(21) |
| Bus | 6 | ze, de, we, vt, vtm, dvtm |
| Load | 4 | ZL_mag, ZL_ang, PL, QL |

---

## 5. `EmtSimu` — 메인 시뮬레이션 객체 [Lib_BW.py:1180-2227]

시간 영역 EMT의 driver. Predictor/network solve/corrector/history를 1 스텝씩 진행.

### 5.1 핵심 필드 [`__init__` 1181-1265]

| 카테고리 | 어트리뷰트 |
|---|---|
| 시간 | `ts=50µs`, `Tlen=0.1s`, `Nlen` |
| State 시계열 (dict, k=step) | `t, x, x_ibr, x_load, x_bus, v, i` |
| 이전값 (latest) | `x_pv_1, x_ibr_pv_1, x_load_pv_1, x_bus_pv_1` |
| Predictor 캐시 | `x_pred = {0,1,2}` (3 스텝 과거) |
| 작업 버퍼 | `xp = States(ngen), xp_ibr = States_ibr(nibr)` |
| 전류 주입 | `Igs (3·nbus)`, `Isg (3·ngen)`, `Igi (3·nbus)`, `Il (3·nbus)`, `Iibr (3·nibr)` |
| 분기 history | `brch_Ihis, brch_Ipre, node_Ihis, I_RHS, Vsol, Vsol_1` |
| 변조/각 | `theta, ed_mod, eq_mod` (per gen) |
| 이벤트 | `t_sc, i_gen_sc, flag_exc_gov, dsp, flag_sc` (step change) |
|  | `t_gentrip, i_gentrip, flag_gentrip, flag_reinit` (gen trip) |
|  | `t_release_f` (PLL freq lock 종료 시각) |
|  | `loadmodel_option` (1=const RLC, 2=const-Z 동적) |
| Vref/gref | `vref, vref_1, gref` (per gen) |
| Playback | `data, playback_enable, _t_chn, _sig_chn, _tn` (외부 시계열 재생) |

### 5.2 `preprocess(self, ini, pfd, dyd)` [1269-1309]
- **목적**: `Initialize`의 결과를 `EmtSimu` 버퍼로 펌프.
- `ini.CombineX()`로 flatten된 `Init_x` 등을 `self.x[0], x_pv_1, x_pred[0..2]`에 복사.
- `Init_net_Vt` (위상자 → t=0 인스턴스) → `self.v[0], Vsol, Vsol_1`.

### 5.3 시간 영역 한 스텝 (main_step1_simulation.py loop) ⭐

```
─── while tn·ts < Tlen ─── tn += 1 ────────────────────────────────
│
│  ① emt.StepChange(dyd, ini, tn)           # vref/gref 스텝 변경 이벤트
│  ② emt.GenTrip(pfd, dyd, ini, tn, netMod) # 발전기 트립 이벤트
│
│  ③ emt.predictX(pfd, dyd, ts)             # 회전자 외삽 → pd_*
│
│  ④ emt.Igs.fill(0)
│     emt.updateIg(pfd, dyd, ini)           # 동기기 Norton 주입 → Igs
│
│  ⑤ emt.Igi.fill(0)
│     emt.Iibr.fill(0)
│     emt.updateIibr(pfd, dyd, ini)         # IBR Norton 주입 → Igi, Iibr
│
│  ⑥ if loadmodel_option == 2:
│       emt.Il.fill(0)
│       emt.updateIl(pfd, dyd, tn)          # 동적 부하 전류
│
│  ⑦ emt.solveV(ini)                        # V_sol = G_0⁻¹·(Igs + Igi + node_Ihis [+Il])  ⭐
│
│  ⑧ emt.BusMea(pfd, dyd, tn)               # 모선 PLL + Vt 측정
│
│  ⑨ emt.updateX(pfd, dyd, ini, tn)         # GENROU + SEXS + Gov + PSS 상태 갱신
│
│  ⑩ emt.updateXibr(pfd, dyd, ini, ts)      # REGCA + REECB + REPCA 갱신
│
│  ⑪ if loadmodel_option == 2:
│       emt.updateXl(pfd, dyd, tn)          # 부하 측정 P/Q
│
│  ⑫ x_pred shift: {0:x_pred[1], 1:x_pred[2], 2:x_pv_1}  # 외삽 history 진행
│
│  ⑬ if tn mod DSrate == 0:  save snapshot to dicts
│
│  ⑭ if flag_gentrip==0 and flag_reinit==1:
│        emt.Re_Init(pfd, dyd, ini)         # gen trip 후 warm restart
│     else:
│        emt.updateIhis(ini)                # 사다리꼴 history 갱신 → node_Ihis (다음 스텝용)
│
─── 한 스텝 끝 ─────────────────────────────────────────────────────
```

각 단계는 **Numba JIT 커널 호출 wrapper**일 뿐 (Python overhead 최소화).

### 5.4 메서드별 디테일

#### `predictX(self, pfd, dyd, ts)` [1311-1404]
- `numba_predictX(x_pv_1, x_pv_2, x_pv_3, …)` 호출.
- 출력: `pd_w, pd_id, pd_iq, pd_EFD, pd_u_d, pd_u_q, pd_dt` + history 묶음.
- 외삽 차수: history 깊이에 따라 1점 / 2점 / 3점 hybrid.

#### `updateIg(self, pfd, dyd, ini)` [1407-1455]
- 동기기 Norton: `numba_updateIg`에 위임. 자세한 수식은 §11.4.
- 단자 EMF source `(ed_mod, eq_mod)` → reduced stator `Rav` → 역 Park → abc Norton 전류 → `Igs[3·nbus]`에 누적.
- `flag_gentrip==0`이고 `i==i_gentrip`이면 해당 발전기 skip.

#### `updateIibr(self, pfd, dyd, ini)` [1458-1485]
- IBR Norton: `numba_updateIibr`. PLL 각으로 inverse Park, REGCA의 `(ip, iq)` 출력을 abc로.

#### `solveV(self, ini)` [1486-1503] ⭐
- **네트워크 해**:
  ```
  Vsol_1 = Vsol                                # 직전값 보관
  I_RHS = Igs + Igi + node_Ihis (+ Il if dyn)
  ```
- 분해 모드 분기:
  - `'inv'`: `Vsol = G0_inv @ I_RHS` (밀집)
  - `'lu'`: `Vsol = G0_lu.solve(I_RHS)` (SuperLU 재사용)
  - `'bbd'`: 재정렬 → Schur LU solve → 역치환
- **G_0는 한 번만 빌드, LU는 한 번만 분해, RHS만 매 스텝 갱신** — 빠름.

#### `BusMea(self, pfd, dyd, tn)` [1774-1888]
- `numba_BusMea`에 위임. SRF-PLL + 3-phase magnitude + Tvm 필터링.
- `tn·ts < t_release_f` 동안 `we=1.0` 고정 (초기 transient 억제).

#### `updateX(self, pfd, dyd, ini, tn)` [1506-1659] ⭐
- **모든 동기기 + Exc + Gov + PSS** 상태를 1 스텝 갱신.
- `numba_updateX`에 위임 (§11.7). 직접 들여다보지 않아도 됨 — JIT 위임 wrapper.

#### `updateXibr(self, pfd, dyd, ini, ts)` [1662-1687]
- IBR 동특성 (REGCA + REECB + REPCA). `numba_updateXibr`.

#### `updateIhis(self, ini)` [1691-1700] ⭐
- **사다리꼴 history 갱신 — EMT의 핵심**.
- `numba_updateIhis(brch_Ihis, Vsol, Init_net_coe0, N)` 호출.
- 각 분기 i에 대해:
  - **2단자 R-L**: $I_{his}(t) = \alpha·I_{pre} + G_{eq}·(V_F − V_T)$
  - **션트** (Tidx==-1): $I_{his}(t) = \alpha·I_{pre} + G_{eq}·V_F$
  - 노드 누적: `node_Ihis[F] -= Ihis`, `node_Ihis[T] += Ihis`

#### `updateIl(self, pfd, dyd, tn)` [1703-1731]
- `loadmodel_option==2` (동적 const-Z, PLL freq tracked) 전용.
- 각 부하 모선에서 `Imag = Vmag/|ZL|`, `i_a = -Imag·cos(V_ang_a + ω·Δt − ∠ZL)` (b/c는 ±2π/3).
- `t_release_f` 이전: `ω = ws` (고정). 이후: `we·ws` (PLL 결과).

#### `updateXl(self, pfd, dyd, tn)` [1733-1771]
- `Imag, ZL_ang`은 const. 측정 `P, Q`만 매 스텝 갱신.
- 3-phase instantaneous power:
  ```
  pe = -(va·ia + vb·ib + vc·ic) · 2/3
  qe = -((vb-vc)·ia + (vc-va)·ib + (va-vb)·ic)/√3 · 2/3
  ```

#### `StepChange(self, dyd, ini, tn)` [1890-1912]
- `t_sc`에 vref(exc) 또는 gref(gov) 한 번 스텝 변경.
- `flag_exc_gov`로 분기: 0→ vref, 1→ gref.
- `dyd.gov_type[i]`로 GAST/HYGOV/TGOV1 분기.

#### `GenTrip(self, pfd, dyd, ini, tn, netMod)` [1914-1924]
- `t_gentrip`에 발전기 개방.
- `Igs[3·nbus]`의 해당 모선 a/b/c slot을 0으로.
- 첫 트리거에서 `ini.InitNet()` + `ini.MergeMacG(..., i_gentrip, netMod)` 재호출 → **G_0 재분해** (트립된 머신의 G_equiv 제외).

#### `Re_Init(self, pfd, dyd, ini)` [1926-2164]
- **목적**: gen trip 후 한 스텝짜리 warm restart. 토폴로지 급변 시 ringing 방지.
- 내부에서 `updateIhis-like + predictX-like + updateIg-like + solveV + updateIhis-like` 5단을 inline (JIT 경로 안 거치고 직접 Python으로) 수행.
- `flag_reinit = 0`으로 설정 후 종료.

#### `dump_res(self, ...)` [2166-2227]
- `x, x_ibr, v, x_bus, x_load`를 transpose해 ndarray로 변환.
- `SimMod==0`: 전체 snapshot + 1-point snapshot.
- SuperLU 객체는 pickle 불가 → `Init_net_G0_lu = []`로 비우고 저장.

---

## 6. 작업 버퍼 — `States`, `States_ibr` [Lib_BW.py:2230-2479]

JIT 커널이 per-step allocation 없이 재사용하는 길이 `ngen`/`nibr` 사전 할당 배열 묶음.

### `States(ngen)` — 동기기 작업 버퍼
필드 네이밍 컨벤션:
- `pv_<var>_k` — past value, k 스텝 전 (k=1,2,3)
- `pd_<var>` — predicted (외삽한 다음 스텝 추정치)
- `nx_<var>` — next (방금 계산한 새 값)

| 그룹 | 변수 |
|---|---|
| Rotor | `dt` (δ), `w` (ω), `id, iq` (gen 컨벤션) |
| Damper/Field | `ifd, i1d, i1q, i2q` |
| Internal EMF | `ed, eq` |
| Flux | `psyd, psyq, psyfd, psy1q, psy1d, psy2q` |
| Torque/Terminal | `te`, `i_d_1/i_q_1` (load 컨벤션), `u_d/u_q`, `pd_u_d/pd_u_q` |
| History terms | `his_d_1, his_fd_1, his_1d_1, his_q_1, his_1q_1, his_2q_1, his_red_d_1, his_red_q_1` |
| EXC (SEXS) | `EFD, v1` |
| GOV | `pm, p1, p2` (TGOV1/GAST); `xe, xc, xg, xq` (HYGOV); `p1, p2, p3` (GAST 같은 슬롯 재사용) |
| PSS (IEEEST) | `y1..y7, x1, x2, vs` |

### `States_ibr(nibr)` — IBR 작업 버퍼
- **REGCA**: `s0, s1, s2`, `Vmp, Vap, i1, i2, ip2rr`
- **REECB**: `s0..s5`, `Ipcmd, Iqcmd, Pref, Qext, q2vPI, v2iPI`
- **REPCA**: `s0..s6`, `Vref, Qref, Freq_ref, Plant_pref, LineMW/Mvar/MVA, QVdbout, fdbout, Pref_out, vq2qPI, p2pPI`
- **PLL**: `ze, de, we`
- **freq**: `nx_freq`

---

## 7. 핵심 수치 알고리즘

### 7.1 사다리꼴 Norton 등가 (R-L 분기)

직렬 R-L 분기:  $v(t) = R·i(t) + L·\dfrac{di}{dt}$

사다리꼴 이산화:
$$
v(t) + v(t-\Delta t) = R[i(t)+i(t-\Delta t)] + \dfrac{2L}{\Delta t}[i(t) - i(t-\Delta t)]
$$

→ Norton equivalent:
$$
\boxed{
G_{eq} = \dfrac{1}{R + 2L/\Delta t}, \quad
\alpha = \dfrac{2L/\Delta t - R}{2L/\Delta t + R}
}
$$
$$
i(t) = G_{eq}·v(t) + I_{hist}, \quad
I_{hist} = G_{eq}·v(t-\Delta t) + \alpha·i(t-\Delta t)
$$

ParaEMT가 추가로 적용:
- **수치 댐핑 저항** `Rp = 20/3 · 2L/ts` (병렬, CDA-style)
- 코드 내 `Req`, `icf` (= α 변형), `Gv1` (= G_eq 변형)이 `Init_net_coe0`에 사전 계산.

### 7.2 사다리쥴 Norton (R-C, capacitor)

션트 캐패시터:
```
Rs = 0.15·ts/(2C)         # snubber resistance (실험적 0.15)
Rc = ts/(2C)
Req = Rs + ts/(2C)
icf = -(Rc-Rs)/(Rc+Rs)
Gv1 = -1/(Rc+Rs)
```

### 7.3 GENROU 8-winding state update (corrector)

각 발전기에 대해 `numba_updateX`에서 (자세한 내용은 §11.7):

**직접 Park (V_abc → e_d, e_q)** at $\theta = \delta_{pred} - \pi/2$:
$$
P_k[0,k] = \cos(\theta + k\cdot 2\pi/3), \quad
P_k[1,k] = -\sin(\theta + k\cdot 2\pi/3)
$$

**Stator 전류** (modified EMF source):
$$
i_d^{new} = (e_d^{mod} - e_d^{new}) / R_{av}, \quad
i_q^{new} = (e_q^{mod} - e_q^{new}) / R_{av}
$$

**Field + 댐퍼** (d-축 2×2 시스템):
$$
R_{d1} \begin{bmatrix} i_{fd} \\ i_{1d} \end{bmatrix} = \begin{bmatrix} EFD\cdot R_{fd}/L_{ad} - h_{fd} + R_{d1}[1,0]\cdot i_d \\ -h_{1d} + R_{d1}[2,0]\cdot i_d \end{bmatrix}
$$
$$
i_{fd}^{new} = R_{d1inv}[0,0]\cdot v_1 + R_{d1inv}[0,1]\cdot v_2
$$
대칭으로 q-축에서 $i_{1q}, i_{2q}$.

**Flux** (6 winding 갱신):
$$
\begin{aligned}
\psi_d  &= -(L_{ad}+L_l)\,i_d + L_{ad}\,i_{fd} + L_{ad}\,i_{1d}\\
\psi_q  &= -(L_{aq}+L_l)\,i_q + L_{aq}\,i_{1q} + L_{aq}\,i_{2q}\\
\psi_{fd}  &= -L_{ad}\,i_d + L_{ffd}\,i_{fd} + L_{f1d}\,i_{1d}\\
\psi_{1q}  &= -L_{aq}\,i_q + L_{11q}\,i_{1q} + L_{aq}\,i_{2q}\\
\psi_{1d}  &= -L_{ad}\,i_d + L_{f1d}\,i_{fd} + L_{11d}\,i_{1d}\\
\psi_{2q}  &= -L_{aq}\,i_q + L_{aq}\,i_{1q} + L_{22q}\,i_{2q}
\end{aligned}
$$

**Torque / Q**:
$$
P_e = \psi_d\,i_q - \psi_q\,i_d, \qquad Q_e = \psi_d\,i_d + \psi_q\,i_q
$$

**Swing**:
$$
\dot{\omega} = \dfrac{\omega_0(P_m/(w/\omega_0) - P_e) - D\Delta\omega}{2H},\quad \dot\delta = \omega_0\Delta\omega
$$
적분: 명시적 Euler (swing은 trapezoidal-ish).

### 7.4 Schur reduction — 왜 필요한가

GENROU의 풀 8×8 (또는 6×6 reduced) 시스템을 매 스텝 풀지 않고:

1. 회전자 winding (field, 1d, 1q, 2q)을 Schur complement로 **eliminate**
2. Stator 전류 `i_d, i_q`만 남는 reduced 2×1 시스템으로 축소
3. `Rd_coe, Rq_coe`에 reduction 계수 보관 (한 번 계산, 매 스텝 재사용)

이 덕분에 매 스텝 inverse는 `Rd1inv` (2×2) 하나만, 매우 가벼움.

### 7.5 Predictor — 외삽 방식

`numba_predictX`에서 history 깊이에 따라:
- **1점** (`xlen=1`): `pd = pv_1` (zero-order hold)
- **2점** (`xlen=2`): `pd = 2·pv_1 - pv_2` (linear)
- **3점** (`xlen=3`): `pd = 1.25·pv_1 + 0.5·pv_2 - 0.75·pv_3` (ParaEMT 자체 hybrid filter)

3점 식은 표준 quadratic 외삽이 아닌 **노이즈 감쇠와 외삽 균형용 ParaEMT 특제 가중치**. 일반적인 quadratic은 `pd = 3·pv_1 − 3·pv_2 + pv_3`인데, 이 식은 고주파 게인을 4배 증폭시키므로 ParaEMT는 사용하지 않음.

**δ 외삽은 별도**:  `pd_dt = pv_dt_1 + ts·0.5·(pv_w_1 + pd_w)` (사다리꼴).

---

## 8. SRF-PLL (Bus 측정용)

`numba_BusMea`에서 각 모선마다 SRF (Synchronous Reference Frame) PLL 굴림:

**state**: `[ze (적분), de (각), we (주파수, pu), vt, vtm (필터링된 V), dvtm]`

```
① theta = de_1 + ts·we_1·ws            # 각 진행
② vq = -2/3·(sin(θ)·va + sin(θ-2π/3)·vb + sin(θ+2π/3)·vc)
③ PI:  nx_ze = ze_1 + (Ke/Te)·vq·ts    # 적분
       nx_we = 1 + Ke·vq + ze_1        # PI 출력 (pu, 1.0 base)
④ nx_de = de_1 + we_1·ws·ts            # 각 적분
⑤ Vt = √((va² + vb² + vc²)·2/3)
       nx_dvtm = (Vt - vtm_1)/vm_te    # 1st-order lag dxdt
       nx_vtm  = vtm_1 + nx_dvtm·ts
```

**Lock 후 vq → 0** → PLL이 단자 전압의 a-축에 정렬.

`tn·ts < t_release_f` 동안 `we=1.0` 고정 (초기 transient 억제).

---

## 9. IBR 모델 (REGCA + REECB + REPCA)

WECC 표준 generic 재생에너지 모델. 세 모듈로 분해:

### 9.1 REGCA — Generator/Converter Model

| 상태 | 의미 |
|---|---|
| `s0` | Active current Ip (with LVPL block) |
| `s1` | Filtered Iq |
| `s2` | Vmag filter (LVPL용) |
| `Vmp, Vap` | PLL 통과 V mag/angle |
| `i1` | HV reactive boost: `max(0, (Vm-Volim)·Khv)` |
| `i2` | LV active partial block: `ramp(Vm; Lvpnt0, Lvpnt1) ∈ [0,1]` |
| `ip2rr` | Ip 변화율 추적 (rate limit용) |

**출력**:  `ip = s0·i2`,  `iq = -s1 - i1`

**Iq 갱신**: `s1_new = s1 + (Iqcmd - s1)·ts/Tg`, with rate limits `Iqrmax/Iqrmin`.

**Ip 갱신**: rate limit by `Rrpwr`, then `s0_new = s0 + tempin·ts/Tg`, clip to `lvpl`.

### 9.2 REECB — Electrical Control

| 상태 | 의미 |
|---|---|
| `s0` | Vt 필터 (Trv) |
| `s1` | P 측정 필터 (Tp) |
| `s2` | Q PI integrator (Kqp/Kqi) |
| `s3` | V PI integrator (Kvp/Kvi) |
| `s4` | Open-loop Iq filter (Tiq) |
| `s5` | Pref lag (Tpord) |
| `Ipcmd, Iqcmd` | 출력 명령 |

**PQFLAG**: 0=Q 우선 → `Ipmax=√(Imax²-Iqcmd²)`, 1=P 우선.

**Voltage dip detection**: `Vdip ≤ Vm ≤ Vup`이면 dip 아님. dip 동안 PI 적분 freeze (anti-windup).

**Voltage deadband + Iqv**: `Vref - Vt`가 `dbd1~dbd2` 밖이면 deadband 통과량 × Kqv → Iqv → clip → Iqinj.

**Iqcmd = (QFLAG ? s3 : s4) + Iqinj** → clip.

**Ipcmd = s5/V3** (V3 = max(Vt, 0.01)) → clip `Ipmin/Ipmax`.

### 9.3 REPCA — Plant Control

| 상태 | 의미 |
|---|---|
| `s0` | V 필터 |
| `s1` | Q 필터 |
| `s2` | Q PI integrator (Kp/Ki, freeze if Vm < Vfrz) |
| `s3` | Q lead-lag (Tfv) |
| `s4` | P 필터 (Tp) |
| `s5` | P PI integrator (Kpg/Kig) |
| `s6` | Pref lag (Tg) |

**Frequency droop + deadband**: f_temp = Freq_ref - Vf, deadband (`fdbd1/fdbd2`):
$$
\text{pfdroop} = \min(f·D_{dn}, 0) + \max(f·D_{up}, 0) \quad \text{(asymmetric)}
$$

**Line compensation** (`VCFlag`): V1 = |Vcom - (Rc+jXc)·Ibranch| if 1, else `Vm + LineMvar·Kc`.

---

## 10. 이벤트 처리

| 이벤트 | 트리거 시각 | 메커니즘 |
|---|---|---|
| 스텝 변경 (vref/gref) | `t_sc` | `StepChange` — `vref[i]` 또는 `gref[i]`에 `dsp` 가산, `flag_sc=0` |
| 발전기 트립 | `t_gentrip` | `GenTrip` — `Igs[3·nbus]`의 해당 모선 슬롯 0으로, `InitNet+MergeMacG` 재호출 (트립된 머신 제외), 다음 스텝에 `Re_Init` |

**`Re_Init`** (warm restart): 발전기 트립 직후 한 스텝을 inline Python으로 처리. `updateIhis-like → predictX-like → updateIg-like → solveV → updateIhis-like`. 토폴로지 급변 시 ringing 방지.

---

## 11. `lib_numba.py` — 함수 전수 분석

모든 함수에 동일한 데코레이터: `@numba.jit(nopython=True, nogil=True, boundscheck=False, parallel=False)`. `fastmath`/`cache` 미사용. **`parallel=False`지만 내부에 `numba.prange` 사용** — 추후 parallel 켜질 때 대비.

### 11.1 `numba_set_coo(rows, cols, data, idx, rval, cval, val)` [lines 5-18]
- COO sparse 배열에 한 엔트리 (idx) 쓰기. 헬퍼 (인라이닝 기대).

### 11.2 `numba_InitNet(...)` [20-487] ⭐

초기화 1회. 전체 네트워크 (라인 pi, 변압기, 부하, 션트)의 사다리꼴 companion + 위상자 + history 모두 구축.

**R-L (line·xfmr·inductive load) trapezoidal 환산**:
```
Rp = damptrap · (20/3 · 2L/ts)                              # CDA 수치 댐핑
Req = (1 + R·(ts/2L + 1/Rp)) / (ts/2L + 1/Rp)
icf = (1 - R·(ts/2L - 1/Rp)) / (1 + R·(ts/2L + 1/Rp))       # i_prev 계수 (= α 변형)
Gv1 = (ts/2L - 1/Rp) / (1 + R·(ts/2L + 1/Rp))               # v_prev 계수
```

**R-C (capacitive shunt) trapezoidal**:
```
Rs = 0.15·ts/(2C)               # 0.15 계수 — load의 경우 추가 ×0.001
Rc = ts/(2C)
Req = Rs + ts/(2C),  icf = -(Rc-Rs)/(Rc+Rs),  Gv1 = -1/(Rc+Rs)
```

**Base**: $V_{base} = kV/\sqrt 3$, $Z_{base} = 3V²/MVA$, $I_{base} = MVA/(3V)$.

**3상 위상자 → 인스턴스**: A=원본, B=`V·(-0.5 - j√3/2)`, C=`V·(-0.5 + j√3/2)`.

**G_0 stamping**: 각 R-L branch당 4-block × 3-phase = 12 COO 엔트리. `+1/Req` 대각 2개, `-1/Req` 비대각 2개 × 3 phases.

**Init_net_coe0**: 분기별 9-tuple `[Fidx, Tidx, Req, icf, Gv1, R, L(or X), C, i_branch_phasor]`.

**초기 history**: `brch_Ihis = icf·I_pre + Gv1·V_pre` → 노드로 aggregate.

상수: `damptrap = 1` (line 134 hardcoded).

### 11.3 `numba_predictX(x_pv_1, x_pv_2, x_pv_3, gen_bus, ws, gen_genrou_odr, exc_sexs_xi_st, exc_sexs_odr, ts, xlen)` [489-727]

**u_d, u_q 정의** (속도 효과 EMF):
$$
u_d = -\psi''_q · \omega/\omega_0, \quad u_q = \psi''_d · \omega/\omega_0
$$

**외삽**: history 깊이별 가중치 (§7.5).

**State 인덱싱** (gen i):
- 0: δ (dt), 1: ω, 2: i_d, 3: i_q, 4: i_fd, 5: i_1d, 6: i_1q, 7: i_2q, 8: ed, 9: eq, 10: ψ''_d, 11: ψ''_q

SEXS EFD: `x_pv[i·exc_sexs_odr + exc_sexs_xi_st + 1]`.

출력 10-tuple: `pd_w, pd_id, pd_iq, pd_EFD, pd_u_d, pd_u_q, pd_dt` + 3개 history 묶음.

### 11.4 `numba_updateIg(Igs, Isg, ...)` [729-845] ⭐

발전기 Norton 등가 전류 → abc 좌표 변환 → 노드 RHS 누적.

**History EMF (per winding)** — d축 3개, q축 3개:
```
pv_his_d  = -α·ed  + α·u_d  + Rd2[i,0,:] · [-id, ifd, i1d]     # d-stator
pv_his_fd = -α·EFD·(Rfd/Lad) + Rd2[i,1,:] · [-id, ifd, i1d]    # field
pv_his_1d = Rd2[i,2,:] · [-id, ifd, i1d]                       # d-damper
pv_his_q  = -α·eq + α·u_q + Rq2[i,0,:] · [-iq, i1q, i2q]       # q-stator
pv_his_1q = Rq2[i,1,:] · [-iq, i1q, i2q]                       # q-damper 1
pv_his_2q = Rq2[i,2,:] · [-iq, i1q, i2q]                       # q-damper 2
```

**Schur reduction**:
```
pv_his_red_d = pv_his_d - (Rd_coe[i,0]·(pv_his_fd - EFD·Rfd/Lad) + Rd_coe[i,1]·pv_his_1d)
pv_his_red_q = pv_his_q - (Rq_coe[i,0]·pv_his_1q + Rq_coe[i,1]·pv_his_2q)
```

**단자 modified EMF**:
```
ed_temp = pd_u_d + pv_his_red_d
eq_temp = pd_u_q + pv_his_red_q
ed_mod = ed_temp - (Rd - Rq)/2·pd_id        # saliency 보정 (R_av 사용 cross-term)
eq_mod = eq_temp + (Rd - Rq)/2·pd_iq
id_src = ed_mod / R_av
iq_src = eq_mod / R_av
```

**Inverse Park** (θ = `pd_dt - π/2`):
```
iPk[k, :] = [cos(θ + k·2π/3), -sin(θ + k·2π/3), 1]  for k = 0,1,2
i_abc = iPk[:,0]·id_src + iPk[:,1]·iq_src
```

**base 환산**: `i_phys = res · base_Is / (Ibase·1000)` → `Isg` + `Igs[phaseA/B/C]`.

`EFD2efd = Rfd/Lad` — PSS/E EFD를 field winding pu로 변환.

### 11.5 `numba_updateIibr(Igi, Iibr, ...)` [847-904]

IBR Norton 전류:
- PLL 각: `theta = pll_de_1 + ts·pll_we_1·2π·60` (**60 Hz hardcoded** — 50 Hz 시 부정확)
- REGCA 출력: `ip = regca_s0·regca_i2`, `iq = -regca_s1 - regca_i1`
- 역 Park + base 환산.

### 11.6 `numba_BusMea(...)` [907-974]

§8의 SRF-PLL + V-mag 필터링. `numba.prange(nbus)`.

### 11.7 `numba_updateX(...)` [976-1517] ⭐⭐

**가장 큰 hot-path**. GENROU + PSS (IEEEST) + Exciter (SEXS) + Governor (TGOV1/HYGOV/GAST) 모두 1 스텝 진행.

수식은 §7.3 참조.

#### IEEEST PSS [lines 1232-1342]
입력: `pss_input = (nx_w - ws)/ws` (pu freq deviation).

7개 transfer-function state (y1..y7) + 2개 입력 derivative 추정 (x1, x2):
- y2 = 입력 + A5/A6 미분 보정 (A1·A2 lag if nonzero)
- y3, y4 — 또 다른 lead-lag (A3, A4)
- y5 — T1/T2 lead-lag
- y6 — T3/T4 lead-lag
- y7 — T5/T6 lead-lag (KS gain)

**Limits**: dy7dt → LSMAX/LSMIN clip → vss. VCU/VCL hysteresis → vs (PSS 출력, exciter input에 가산).

#### SEXS Exciter [1344-1376]
```
input = vref - Vt_filtered - v1 + vs
nx_v1 = pv_v1 + ts·((input)/TB + (TA/TB)·(dvref - dvt_filtered))    # lead-lag
EFD   = pv_EFD + (K·pv_v1 - pv_EFD)·ts/TE                            # 1st-order lag
EFD   = clip(EFD, Emin, Emax)
```
`TE=0`이면 단순 gain `K·v1`.

#### Governors [1378-1483]
공통: `gov_input = nx_w/ws - 1` (pu freq deviation).

**TGOV1** (`gov_type==2`):
```
p1_new = p1 + ts/T1·((gref - gov_input)/R - p1)        # 스팀 밸브, droop R
p1_new = clip(p1_new, Vmin, Vmax)
p2_new = p2 + ts/T3·(p1 - p2 + dp1dt·T2)               # 터빈 lead-lag
Pm     = p2 - Dt·gov_input
```

**HYGOV** (`gov_type==1`):
```
xc filter → xg actuator lag (Tg, rate-limit VELM, position-limit GMIN/GMAX)
xq 물 동역학:  dxq/dt = (1 - (xq/xg)²)/TW            # penstock
Pm = (xq - qNL)·(xq/xg)²·At - Dturb·gov_input·xg
```

**GAST** (`gov_type==0`):
```
가스 터빈 3-state lag chain (T1/T2/T3)
Load limit LdLmt로 fuel cap
Pm = p2 - Dturb·gov_input
```

#### 단자 abc P/Q 재계산 [1488-1515]
state 12/13 슬롯의 te/qe는 abc 단자에서 다시 측정해 덮어씀:
```
pe = (va·ia + vb·ib + vc·ic)·2/3·basemva/gen_MVA_base
qe = ((vb-vc)·ia + (vc-va)·ib + (va-vb)·ic)/√3·2/3·basemva/gen_MVA_base
```
`Isg`에서 `Init_mac_Gequiv·V_abc` 빼서 실제 단자 전류 추출 (Norton 등가 → 실제 분기 전류).

### 11.8 `numba_updateXibr(...)` [1519-2027] ⭐⭐

**REGCA + REECB + REPCA + PLL**. §9 참조.

41 states/IBR:
- REGCA 8 + REECB 12 + REPCA 18 + 측정 3 (Vf, pe, qe)

**모든 controller**: explicit Euler + clip + dip freeze (anti-windup).

**Rate limits**: REGCA Iqrmax/min (Iq), Rrpwr (Ip); REECB dPmin/dPmax (Pref).

### 11.9 `numba_updateIhis(brch_Ihis, Vsol, Init_net_coe0, nnodes)` [2030-2050] ⭐

매 스텝 끝 — 다음 스텝 RHS 준비. **EMT의 핵심 inner loop**.

```
for each branch i:
    if Tidx == -1 (shunt):
        i_pre_new = V[F]/Req + brch_Ihis_prev[i]
        i_his_new = icf·i_pre + Gv1·V[F]
        node_Ihis[F] -= i_his_new
    else (2-terminal R-L):
        i_pre_new = (V[F] - V[T])/Req + brch_Ihis_prev[i]
        i_his_new = icf·i_pre + Gv1·(V[F] - V[T])
        node_Ihis[F] -= i_his_new
        node_Ihis[T] += i_his_new

skip if Init_net_coe0[i,2]==0 (비활성 branch)
```

**`parallel=False` 강제**: parallel 켜면 `node_Ihis[F]/[T] +=` race condition (주석에 명시 line 2029).

매 branch O(1), 전체 O(nbranch) per step.

---

## 12. 핵심 컨벤션 / 메모

| 항목 | ParaEMT 선택 |
|---|---|
| Park 변환 | $\theta_{Park} = \delta - \pi/2$ (q-축을 정상상태 단자 V에 정렬) |
| Park scaling | **Amplitude-invariant 2/3** (peak base) |
| Phasor 컨벤션 | **Peak amplitude** (`base_es = V_LL·√(2/3)·1000`, RMS = `base_Is = base_is/√2`) |
| Stator transient | **포함** — `(1+α)/(ts·ws)·L` companion. (cf. 우리 EMT_3bus는 stator algebraic) |
| GENROU form | **Full 8th-order EC** (Lad-base reciprocal pu). `Lf1d` mutual, `Lffd/L11d/L11q/L22q` 모두 포함 |
| Field/Damper 적분 | **Trapezoidal companion + Schur reduce** → stator만 매 스텝 풀이 |
| Saturation S10/S12 | 데이터엔 있지만 **코드엔 미사용** |
| 0-sequence | `L0, R0` 명시 모델링 (Rs/Rm 패턴) |
| 회전자 / Exc / Gov / PSS / IBR | 모두 **explicit Euler** (Swing만 부분적 trapezoidal) |
| Network solve | sparse G·V=I, LU 1회 분해, 3 modes (`inv`/`lu`/`bbd`) |
| BBD 병렬 | `partitionutil.form_bbd` + `serial_bbd_matrix.schur_bbd_lu` |
| EFD base | PSS/E (`EFD = ifd·Lad`) — `EFD2efd = Rfd/Lad` 비율 |
| 시간 스텝 | `ts = 50 µs` (디폴트) |
| 수치 댐핑 α | `99/101 ≈ 0.98` (`MergeMacG`), `damptrap=1` (`numba_InitNet`), `0.15` (capacitor snubber) |
| Numba JIT | 모든 hot-path (`predictX`, `updateIg/Iibr/X/Xibr`, `updateIhis`, `BusMea`, `InitNet`) JIT |
| 60 Hz hardcoded | `numba_updateIibr` line 887 `2π·60` — 50 Hz 시스템에 부정확 |

### 알려진 코드 quirks

1. **`PFData.lists_to_arrays`** (467-472) — `LargeSysGenerator` 내부에 `return pfd` 이후 정의 → unreachable dead code.
2. **`gov_type`** in `DyData` — gov-count 크기 배열을 gen-index로 인덱싱 → 모든 gen에 governor 있을 때만 정상.
3. **`States.__init__`** — GAST의 `pv_p1_*`이 TGOV1의 `pv_p1_*` 슬롯과 같은 이름 (같은 버퍼 재사용).
4. **`Init_mac_te`** — 코드는 `te = ed·id + eq·iq` (P 표현) 저장하지만 주석은 토크 `ψd·iq − ψq·id` 언급 (정상상태에선 동일).
5. **REGCA HVPL 초기화** — `s1 = -Iq_out`만 저장, `+i1` 기여는 매 스텝 첫 호출에서 추가됨 (Tfltr로 빨리 washing out).
6. **`Re_Init`** — `Init_net_coe0`을 `ini.`경유 없이 글로벌처럼 참조 (lines 2141, 2143) — 외부에서 단독 호출 시 NameError 위험.
7. **`MergeMacG`** — `Init_mac_Rq_coe`의 `np.vstack` 첫 iter일 때만 사용 분기 — bug-prone idiom.
8. **FFT-based symmetric-component voltage measurement** — 1778-1869 라인에 완전 구현되어 있지만 **주석 처리** (`numba_BusMea`의 SRF-PLL로 대체). 글로벌 `Ainv`는 현재 미사용.

---

## 13. BBD 병렬화 (요약)

`netMod='bbd'` 옵션 시 활성화. README에 의하면 `nxmetis`가 unmaintained라 **현재 비활성 상태**.

**개요**:
1. `partitionutil.admittance_to_BBD(A, num_parts)` — `nxmetis.partition`으로 G 행렬을 num_parts개 블록 + 1개 interface 블록으로 분할
2. `bbd_matrix.bbd_matrix` — Bordered Block Diagonal 행렬 클래스
3. `serial_bbd_matrix.schur_bbd_lu` — interface 블록의 Schur complement 후 블록별 LU
4. 매 스텝 `schur_solve`로 G·V=I 풀이 (블록 LU는 1회만)

**핵심 아이디어**: G를 block diagonal + bordering으로 재배치하면 diagonal blocks를 병렬로 LU 분해 가능. interface 블록은 Schur complement로 따로 처리.

자세한 내용은 [`ParaEMT_public-main/partitionutil.py`](../ParaEMT_public-main/partitionutil.py), [`bbd_matrix.py`](../ParaEMT_public-main/bbd_matrix.py), [`serial_bbd_matrix.py`](../ParaEMT_public-main/serial_bbd_matrix.py) 참조.

---

## 14. 우리 `EMT_3bus`와의 대응 관계

학습 차원에서 만든 [`EMT_3bus/`](EMT_3bus/) 미니 구현과 ParaEMT의 대응:

| 우리 (`EMT_3bus`) | ParaEMT |
|---|---|
| `BaseComponent` (stamp_G/I, predict, update) | `EmtSimu` 단일 클래스 + `Init_net_coe0` 룩업 테이블 |
| `BusManager` | `Init_net_N`, gen_bus/ibr_bus/bus_num 룩업 |
| `Park` (Kundur 2/3) | `numba_predictX/updateIg/updateX`에 inline (같은 컨벤션) |
| `TransmissionLine` (pi-model, trap) | `numba_InitNet`이 라인/변압기 한꺼번에 처리 |
| `Load` (Constant-Z, R/RL/RC) | `numba_InitNet`에 inline + ZIP 모델 (`loadmodel_option`) |
| `VoltageSource` (Norton with R_int) | **없음** — slack도 PSS/E PF에서 결정 |
| `Generator` (GENROU Simple, 6 ODE) | **GENROU Full 8-winding** (lf1d mutual, 0-seq, saturation 데이터 보유) |
| `Simulator.initialize` (picard) | `Initialize.*` 11개 메서드 + PSS/E NR PF (외부) |
| `Simulator.step` | `EmtSimu.predictX → updateIg → ... → updateIhis` |
| `Simulator.rebuild_G` | `EmtSimu.GenTrip` → `InitNet+MergeMacG` 재호출 |
| set_active(False) 등 토폴로지 | gen trip만 지원 (line trip 미구현 in main loop) |

**가장 큰 차이점**:
1. **NR PF**: ParaEMT는 PSS/E에 위임. 우리는 picard fixed-point (slack-dominated 3-bus라 가능).
2. **Stator transient**: ParaEMT는 trapezoidal companion으로 포함. 우리는 algebraic stator (생략).
3. **Full 8-winding**: ParaEMT는 field+1d+1q+2q 모두 ODE. 우리는 Simple GENROU (4 electrical + 2 mechanical).
4. **IBR 모델**: ParaEMT는 REGCA/REECB/REPCA. 우리는 없음.
5. **JIT**: ParaEMT는 Numba로 ~10× 가속. 우리는 순 numpy.
6. **bumpless 시작**: ParaEMT는 PSS/E 표준 5-step (`InitMac`). 우리는 picard + backsolve.
7. **수치 댐핑**: ParaEMT는 α=99/101 자동 적용. 우리는 옵션 (디폴트 0).

---

## 15. 참고 — 한 줄 요약

> **ParaEMT는 [PSS/E NR PF로 정상상태] → [Lib_BW.py의 Initialize 클래스가 모든 모델을 bumpless 셋업 + 사다리꼴 companion G 행렬 LU 분해] → [EmtSimu가 매 스텝 predictor(외삽) → Norton 전류 stamping → G·V=I 풀이 → corrector(상태 갱신, 모두 explicit Euler/trapezoidal) → history 갱신] 하는 구조. 모든 hot-path는 lib_numba.py에서 JIT 컴파일. 토폴로지는 G 행렬을 매 스텝 재분해하지 않고 LU 재사용으로 매우 빠르고, BBD 옵션으로 multi-core 병렬화 가능 (현재 nxmetis unmaintained로 비활성).**
