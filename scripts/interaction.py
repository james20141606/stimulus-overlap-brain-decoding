import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
cells = {"rr": "result_mb1f_ksweep_",   # recording guard / recording split
         "rs": "result_mb1t_cellC_",    # recording guard / story split
         "sr": "result_mb1t_cellD_",    # stimulus guard / recording split
         "ss": "result_mb1k_volctl_"}   # stimulus guard / story split
D = {c: load_subject_results(R, p, quiet=True) for c, p in cells.items()}
for k in (100, 500):
    subs = sorted(set.intersection(*[set(D[c]) for c in cells]))
    d = {}
    ok_subs = []
    for s in subs:
        vals = {}
        good = True
        for c in cells:
            cv = M.cell(D[c][s], "conv", k)[2]; dy = M.cell(D[c][s], "dynamic_big", k)[2]
            if cv is None or dy is None: good = False; break
            vals[c] = dy - cv
        if good: d[s] = vals; ok_subs.append(s)
    n = len(ok_subs)
    # per-subject contrasts (mean percentile units)
    split_main = np.array([ (d[s]["rs"]+d[s]["ss"])/2 - (d[s]["rr"]+d[s]["sr"])/2 for s in ok_subs])
    guard_main = np.array([ (d[s]["sr"]+d[s]["ss"])/2 - (d[s]["rr"]+d[s]["rs"])/2 for s in ok_subs])
    interact   = np.array([ (d[s]["ss"]-d[s]["sr"]) - (d[s]["rs"]-d[s]["rr"])      for s in ok_subs])
    split_at_recguard = np.array([ d[s]["rs"]-d[s]["rr"] for s in ok_subs])
    guard_at_recsplit = np.array([ d[s]["sr"]-d[s]["rr"] for s in ok_subs])
    print(f"== k={k}  n={n} subjects common to all four cells ==")
    for name, arr in (("split main effect", split_main), ("guard main effect", guard_main),
                      ("guard x split interaction", interact),
                      ("split simple effect @rec guard", split_at_recguard),
                      ("guard simple effect @rec split", guard_at_recsplit)):
        m, lo, hi = M.ci(list(arr))
        import math as _math
        if _math.isnan(lo) or _math.isnan(hi):
            # ci() returns (mean, nan, nan) below 3 subjects; nan>0 is False, so the old code
            # printed a confident-looking "covers 0" for a 2-subject cell (audit 2026-08-19).
            print(f"  {name:32s} {m:+.4f}  [--,--]  insufficient n for an interval")
            continue
        verdict = "excludes 0" if (lo > 0 or hi < 0) else "covers 0"
        print(f"  {name:32s} {m:+.4f}  [{lo:+.4f},{hi:+.4f}]  {verdict}")
