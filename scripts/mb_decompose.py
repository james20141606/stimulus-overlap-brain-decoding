#!/usr/bin/env python
"""Close paper 3's three stated gaps against the MB-1f k-sweep result JSONs.

1. BOOTSTRAP THE DECOMPOSITION. The paper reports the architecture term (noflow-conv) and the flow
   term (dynamic-noflow) as point estimates with an explicit "we have not bootstrapped these" caveat,
   because only dynamic-conv was pre-registered. Compute paired CIs for all three contrasts so the
   caveat can be replaced by numbers (the pre-registration status is unchanged either way -- these
   remain secondary, and we say so).
2. VERIFY POOL CONSTANCY ACROSS k. The paper's headline "the advantage decays with k" compares
   x-chance across budgets, which is only legitimate at fixed pool (see an earlier analysis's
   pool-invariance section). train_mb.py splits eval off before the sweep, so pool SHOULD be constant
   across k. Assert it rather than trust it.
3. FM vs SCRATCH. Each cell also ran a from-scratch encoder on the same k. If scratch matches FM then
   the pretraining -- the whole point of a "transfer" claim -- is doing nothing, and the paper's
   framing would be wrong. Never analysed until now.

Usage: python mb_decompose.py [results_dir]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_io import load_subject_results

R = sys.argv[1] if len(sys.argv) > 1 else "$SCRATCH/project/results/mb"
ARMS = ["conv", "dynamic_big", "dynamic_big_noflow"]
NICE = {"conv": "conv", "dynamic_big": "dynamic", "dynamic_big_noflow": "noflow"}
BOOT, SEED = 10000, 0


def paired_ci(d, n_boot=BOOT, seed=SEED):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


fm, sc, pool = {}, {}, {}
# load via result_io: excludes smoke artefacts and keys on the test_subject FIELD. The previous
# filename-parsing version would have let result_mb1f_ksweep_smoke_s01.json overwrite subject 01.
for s, d in sorted(load_subject_results(R, "result_mb1f_ksweep_", expect=10).items()):
    for arm in ARMS:
        r = d.get("results", {}).get(arm)
        if not r:
            continue
        for p in r.get("curve", []):
            if "fm" in p:
                fm[(s, arm, p["k"])] = p["fm"]["x"]
                pool[(s, arm, p["k"])] = p["fm"].get("pool")
            if "scratch" in p:
                sc[(s, arm, p["k"])] = p["scratch"]["x"]

subs = sorted({k[0] for k in fm})
ks = sorted({k[2] for k in fm})
print(f"subjects={len(subs)}  k={ks}\n")

# ---- 2. pool constancy across k (do this FIRST; it licenses everything else) ----
print("== pool constancy across k (per subject, dynamic arm) ==")
bad = 0
for s in subs:
    vals = [pool.get((s, "dynamic_big", k)) for k in ks]
    vals = [v for v in vals if v is not None]
    uniq = sorted(set(vals))
    flag = "" if len(uniq) == 1 else "  <-- VARIES"
    if len(uniq) != 1:
        bad += 1
    print(f"  s{s}: {uniq}{flag}")
print(f"  => pool varies across k for {bad}/{len(subs)} subjects. "
      f"{'Cross-k comparison is LEGITIMATE.' if bad == 0 else 'Cross-k claim NEEDS REVISION.'}\n")

# ---- 1. decomposition with CIs ----
CONTRASTS = [("dynamic_big", "conv", "dynamic - conv   (PRE-REGISTERED)"),
             ("dynamic_big_noflow", "conv", "noflow  - conv   (architecture term)"),
             ("dynamic_big", "dynamic_big_noflow", "dynamic - noflow (flow term)")]
print("== paired contrasts, 95% CI by subject bootstrap ==")
for a, b, label in CONTRASTS:
    print(f"\n  {label}")
    for k in ks:
        pairs = [(fm.get((s, a, k)), fm.get((s, b, k))) for s in subs]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            print(f"    k={k:>5}: n={len(pairs)} too few"); continue
        d = [x - y for x, y in pairs]
        m, lo, hi = paired_ci(d)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns "
        print(f"    k={k:>5}: n={len(d):2d}  {m:+7.2f}  [{lo:+7.2f},{hi:+7.2f}]  {sig}")

# ---- 3. FM vs SCRATCH: is the pretraining doing anything? ----
print("\n== FM (pretrained) - SCRATCH (same k, no pretraining), dynamic arm ==")
for k in ks:
    pairs = [(fm.get((s, "dynamic_big", k)), sc.get((s, "dynamic_big", k))) for s in subs]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        print(f"  k={k:>5}: n={len(pairs)} too few"); continue
    d = [x - y for x, y in pairs]
    m, lo, hi = paired_ci(d)
    fmm = np.mean([x for x, _ in pairs]); scm = np.mean([y for _, y in pairs])
    sig = "SIG" if lo > 0 else ("scratch ahead" if hi < 0 else "ns")
    print(f"  k={k:>5}: FM {fmm:6.2f}  scratch {scm:6.2f}  diff {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}] {sig}")
