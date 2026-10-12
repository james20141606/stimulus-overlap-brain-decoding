import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
rows = (("SEEN mb1f", "result_mb1f_ksweep_"),
        ("story 0", "result_mb1j_story0_"), ("story 1", "result_mb1s_story1_unseen_"),
        ("story 2", "result_mb1s_story2_unseen_"), ("story 3", "result_mb1h_stimho_"))
for lbl, pref in rows:
    D = load_subject_results(R, pref, quiet=True)
    out = [lbl.ljust(10)]
    for k in (100, 500, 2000):
        d = []
        for s in sorted(D):
            c = M.cell(D[s], "conv", k)[2]; y = M.cell(D[s], "dynamic_big", k)[2]
            if c is not None and y is not None: d.append(y - c)
        if len(d) < 3: out.append(f"k{k}: n<3".ljust(30)); continue
        m, lo, hi = M.ci(d)
        star = "*" if (lo > 0 or hi < 0) else " "
        out.append(f"k{k}: {m:+.4f}[{lo:+.4f},{hi:+.4f}]{star} n={len(d)}")
    print("  ".join(out))
