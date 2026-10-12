import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
rows = (("story 0", "result_mb1j_story0_"), ("story 1", "result_mb1s_story1_unseen_"),
        ("story 2", "result_mb1s_story2_unseen_"), ("story 3", "result_mb1h_stimho_"))
for lbl, pref in rows:
    D = load_subject_results(R, pref, quiet=True)
    d = []
    for s in sorted(D):
        c = M.cell(D[s], "conv", 2000)[2]; y = M.cell(D[s], "dynamic_big", 2000)[2]
        if c is not None and y is not None: d.append(y - c)
    if len(d) < 3: print(f"{lbl}: n={len(d)} too few"); continue
    m, lo, hi = M.ci(d)
    sig = "REVERSED" if hi < 0 else ("dyn" if lo > 0 else "ns")
    print(f"{lbl}  n={len(d):2d}  {m:+.4f}  [{lo:+.4f},{hi:+.4f}]  {sig}")
