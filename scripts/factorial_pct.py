import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
print("cond      k   n     dyn-conv xchance          dyn-conv MEAN_PERCENTILE")
for name, pref in (("mb1f", "result_mb1f_ksweep_"), ("mb1h", "result_mb1h_stimho_"),
                   ("mb1k", "result_mb1k_volctl_")):
    D = load_subject_results(R, pref, quiet=True)
    for k in (100, 500, 2000):
        dx, dp = [], []
        for s in sorted(D):
            c = M.cell(D[s], "conv", k); y = M.cell(D[s], "dynamic_big", k)
            if c[0] is not None and y[0] is not None:
                dx.append(y[0] - c[0]); dp.append(y[2] - c[2])
        if len(dx) < 3:
            continue
        mx, lx, hx = M.ci(dx); mp, lp, hp = M.ci(dp)
        sig = "DYN" if lp > 0 else ("CONV" if hp < 0 else "ns")
        print(f"{name:6s} {k:5d} {len(dx):3d}  {mx:+7.2f}[{lx:+.2f},{hx:+.2f}]"
              f"   {mp:+.4f}[{lp:+.4f},{hp:+.4f}]  {sig}")
