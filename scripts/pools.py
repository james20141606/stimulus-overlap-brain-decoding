import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
for name, pref in (("mb1f","result_mb1f_ksweep_"), ("mb1k","result_mb1k_volctl_")):
    D = load_subject_results(R, pref, quiet=True)
    for k in (100,500,2000):
        pools=[]
        for s in sorted(D):
            for p in D[s].get("results",{}).get("conv",{}).get("curve",[]):
                if p["k"]==k and "fm" in p and p["fm"].get("pool"): pools.append(p["fm"]["pool"])
        if pools: print(f"{name} k={k:5d} mean pool {np.mean(pools):7.0f}")
