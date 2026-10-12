import sys, io, contextlib
sys.path.insert(0, "scripts")
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from scipy.stats import spearmanr
from result_io import load_subject_results
R = "$SCRATCH/project/results/mb"
doses = [0.00, 0.25, 0.50, 0.75, 1.00]
for tag in ("mb1x", "mb1w"):
    print(f"== {tag} ({'PINNED volume' if tag=='mb1x' else 'confounded'}) ==")
    for k in (500, 2000):
        xs, ys, means = [], [], []
        for d in doses:
            D = load_subject_results(R, f"result_{tag}_dose{d:.2f}_", quiet=True)
            v = [M.cell(D[s], "conv", k)[2] for s in sorted(D)]
            v = [x for x in v if x is not None]
            means.append(np.mean(v)); xs += [d]*len(v); ys += v
        r, _ = spearmanr(xs, ys)
        mono = all(b >= a for a, b in zip(means, means[1:]))
        print(f"  k={k:5d}  " + " ".join(f"{m:.3f}" for m in means)
              + f"   delta {means[-1]-means[0]:+.3f}  rho {r:+.3f}  monotone={mono}  n={len(ys)//5}")
