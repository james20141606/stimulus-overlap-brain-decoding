#!/usr/bin/env python
"""Recompute ONLY the cross-story clock arm, to attach a bootstrap CI to it.

The first version of train_fmri_xstory.py saved the clock as a bare point estimate (timing_x) and
discarded the `top10_ci95` that guarded_retrieval_metrics already computes. The cross-story clock is
the load-bearing number -- it is what shows the timing features generalise to an unseen narrative --
so it cannot stay a number without an interval in a paper whose whole subject is unsupported numbers.

The clock uses no BOLD, so this reruns in seconds on CPU rather than needing the GPU job again.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "$SCRATCH/project/src")
from train_fmri_f1 import build_segments, train_eval

META = "$SCRATCH/project/fmri/narratives_meta"
meta = pd.read_csv(f"{META}/trial_metadata.tsv", sep="\t")
ctx = np.load(f"{META}/narr_contextual.npy", mmap_mode="r")

for A, B in [("piemanpni", "bronx"), ("bronx", "piemanpni")]:
    sa = build_segments(meta, ctx, 1, A, 3.0, 3)
    sb = build_segments(meta, ctx, 1, B, 3.0, 3)
    Ya = np.stack([s[1] for s in sa]).astype(np.float32)
    Yb = np.stack([s[1] for s in sb]).astype(np.float32)
    Ta = np.stack([s[2] for s in sa]).astype(np.float32)
    Tb = np.stack([s[2] for s in sb]).astype(np.float32)
    mu, sd = Ta.mean(0), Ta.std(0) + 1e-6
    Ta, Tb = (Ta - mu) / sd, (Tb - mu) / sd
    on_b = np.array([s[0] for s in sb], dtype=np.float64)
    reps = 19
    m = train_eval(np.tile(Ta, (reps, 1)), np.tile(Ya, (reps, 1)), Tb, Yb, on_b, 40, 3e-4, "cpu")
    # Key names verified by RUNNING against the f2/f3 result JSONs (2026-08-19): these files DO
    # carry mean_pool_size / top10 / top10_ci95 / chance_top10. An earlier comment here claimed the
    # metrics dict used pool_m instead -- that described a different producer and was itself the
    # wrong half; the failed job (15066417) predated the current result schema.
    acc = m["top10"]; ci = m["top10_ci95"]; pool = m["mean_pool_size"]; chance = m["chance_top10"]
    print(f"{A:10s} -> {B:10s}  n_test={len(sb):4d} pool={pool:6.1f} chance={chance:.3f}")
    print(f"   clock top-10 acc {acc:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]  "
          f"=> x chance {acc/chance:.2f}  [{ci[0]/chance:.2f},{ci[1]/chance:.2f}]  "
          f"{'ABOVE chance' if ci[0] > chance else 'NOT above chance'}")
