import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
cells = [("recording/recording", "result_mb1f_ksweep_"),
         ("recording/story",     "result_mb1t_cellC_"),
         ("stimulus/recording",  "result_mb1t_cellD_"),
         ("stimulus/story",      "result_mb1k_volctl_")]
print("cell                    k     n    xchance          mean_percentile        pool")
for lbl, pref in cells:
    try:
        D = load_subject_results(R, pref, quiet=True)
    except Exception as e:
        print(f"{lbl:22s} MISSING ({e})"); continue
    for k in (100, 500):
        dx, dp, pl = [], [], []
        for s in sorted(D):
            c = M.cell(D[s], "conv", k); y = M.cell(D[s], "dynamic_big", k)
            if c[0] is not None and y[0] is not None:
                dx.append(y[0]-c[0]); dp.append(y[2]-c[2])
            for p in D[s].get("results",{}).get("conv",{}).get("curve",[]):
                if p["k"]==k and "fm" in p and p["fm"].get("pool"): pl.append(p["fm"]["pool"])
        if len(dx) < 3: continue
        mx,lx,hx = M.ci(dx); mp,lp,hp = M.ci(dp)
        star = "*" if (lp>0 or hp<0) else " "
        print(f"{lbl:22s} {k:5d} {len(dx):3d}  {mx:+6.2f}  {mp:+.4f}[{lp:+.4f},{hp:+.4f}]{star}  {np.mean(pl):6.0f}")
