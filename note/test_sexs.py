import sys, os
sys.path.insert(0, '.')

try:
    from EMT_3bus import SEXS
    print("SEXS import OK")
except Exception as e:
    print("Import error:", e)
    import traceback; traceback.print_exc()
    sys.exit(1)

SEXS_PARAMS = {'TA_TB': 0.0, 'TB': 10.0, 'K': 100.0, 'TE': 0.1, 'Emin': 0.0, 'Emax': 3.0}
DT = 50e-6
E_fd0 = 2.47
Vt0   = 1.04

exc = SEXS(E_fd0=E_fd0, Vt0=Vt0, dt=DT, params=SEXS_PARAMS)
print("초기화:")
print("  v1   = %.8f  (expected E_fd0/K = %.8f)" % (exc.v1, E_fd0/100))
print("  EFD  = %.8f  (expected %.8f)" % (exc.EFD, E_fd0))
print("  vref = %.8f  (expected %.8f)" % (exc.vref, Vt0 + E_fd0/100))

# 정상상태 드리프트 확인
for _ in range(100):
    exc.step(Vt0)
print("\n100스텝 정상상태 유지 확인:")
print("  v1  = %.8f  (초기: %.8f,  오차: %.2e)" % (exc.v1, E_fd0/100, abs(exc.v1 - E_fd0/100)))
print("  EFD = %.8f  (초기: %.8f,  오차: %.2e)" % (exc.EFD, E_fd0,     abs(exc.EFD - E_fd0)))

# 계단 응답: t=1s에 Vt +0.05
exc2 = SEXS(E_fd0=E_fd0, Vt0=Vt0, dt=DT, params=SEXS_PARAMS)
print("\n계단응답 (Vt +0.05 at t=1s, TB=10s, K=100, TE=0.1s):")
print("  t(ms)    v1          EFD         Vt")
for i in range(200001):
    Vt = Vt0 + 0.05 if i >= 20000 else Vt0
    exc2.step(Vt)
    if i in [0, 19999, 20001, 40000, 100000, 200000]:
        print("  %7.1f  %10.6f  %10.6f  %.4f" % (i*DT*1000, exc2.v1, exc2.EFD, Vt))
