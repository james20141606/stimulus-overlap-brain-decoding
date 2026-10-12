#!/usr/bin/env python
"""Analyse F2: does the timing baseline still beat the brain arm once BOTH are held out?

Reads result_f2_<task>_blk<N>.json written by train_fmri_f1.py.

The three questions, in the order they have to be answered:

1. DOES THE BRAIN ARM CLEAR CHANCE AT ALL under the segment holdout? Its blk6 peak is ~1.25x. If that
   is not significantly above 1.0 across the 5 held-out folds, then there is no working decoder and
   "a working decoder loses to a clock" is unavailable regardless of what the clock does.
2. DOES THE CLOCK STILL BEAT THE BRAIN? blk6 clock ~1.99 vs brain peak ~1.25 -- still ahead, but 1.6x
   rather than 6x. Report both arms as accuracy too, since x chance depends on pool and the pool
   changes between blk0 (127) and blk6 (53).
3. IS THE COLLAPSE GENERALISATION OR DATA VOLUME? blk6 trains on 56 segments where blk0 trains on 129,
   so the two differ in TWO ways. The --eval_on_train control (tag suffix _ctl) trains on the same 56
   and evaluates on those same 56; comparing it with blk6 isolates the holdout.

Only the brain arm has per-fold values (5 LOSO folds), so significance is reported for the brain arm
and for brain-vs-chance. The clock is a single fit, so we report its point value and do not invent an
interval for it.

Usage: python f2_analyze.py [results_dir]
"""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "$SCRATCH/project/results/fmri"


def boot_ci(v, n_boot=10000, seed=0):
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# f3 (cross-story) results share the schema and previously had no reader (audit 2026-08-19).
files = sorted(glob.glob(f"{R}/result_f2_*.json")) + sorted(glob.glob(f"{R}/result_f3_*.json"))
if not files:
    sys.exit(f"no F2 results in {R}")

for f in files:
    d = json.load(open(f))
    tag = d.get("tag", os.path.basename(f))
    lags = d.get("lags", {})
    if not lags:
        print(f"\n===== {tag}: no lag results =====")
        continue
    any_lag = next(iter(lags.values()))
    pool = any_lag.get("pool")
    ntr, nte = any_lag.get("n_seg_train"), any_lag.get("n_seg_test")
    print(f"\n===== {tag} =====")
    print(f"  pool={pool}  train_seg={ntr}  test_seg={nte}  "
          f"(top-10 chance = {10 / pool:.3f})" if pool else "")

    print(f"  {'lag':>5} {'brain':>7} {'acc':>7} {'95% CI (folds)':>22} {'clock':>7} {'acc':>7}  "
          f"{'clock/brain':>11}")
    best = (None, -np.inf)
    for lg in sorted(lags, key=float):
        r = lags[lg]
        b, t = r["brain_x_mean"], r["timing_x"]
        per = r.get("brain_x_per_subject") or []
        if len(per) >= 3:
            _, lo, hi = boot_ci(per)
            ci = f"[{lo:.2f},{hi:.2f}]" + ("*" if lo > 1.0 else " ")
        else:
            ci = "(n<3)"
        ba = b * 10 / pool if pool else float("nan")
        ta = t * 10 / pool if pool else float("nan")
        print(f"  {float(lg):5.1f} {b:7.2f} {ba:7.3f} {ci:>22} {t:7.2f} {ta:7.3f}  {t / b:11.2f}")
        if b > best[1]:
            best = (lg, b, per, t)

    lg, b, per, t = best
    print(f"\n  BEST LAG {float(lg):.0f}s: brain {b:.2f}x, clock {t:.2f}x, clock/brain {t / b:.2f}x")
    if len(per) >= 3:
        m, lo, hi = boot_ci(per)
        verdict = ("brain CLEARS chance" if lo > 1.0 else "brain does NOT clear chance")
        print(f"  brain per-fold {['%.2f' % v for v in per]}")
        print(f"  brain mean {m:.2f}, 95% CI [{lo:.2f},{hi:.2f}] vs chance 1.0  ->  {verdict}")
        print(f"  => {'a working decoder that still loses to the clock' if lo > 1.0 else 'NO working decoder; the loses-to-a-clock framing is unavailable'}")
