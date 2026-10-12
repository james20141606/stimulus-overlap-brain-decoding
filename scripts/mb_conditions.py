#!/usr/bin/env python
"""Read every MB protocol variant and produce the tables the paper turns on.

One script instead of ad-hoc SSH one-liners, because the same three numbers keep getting recomputed by
hand and one of those hand-rolled queries already silently ate a real subject (a smoke-test file
overwrote subject 01 -- see scripts/result_io.py). Everything here loads through result_io, which
excludes smoke artefacts, keys on the `test_subject` FIELD, and raises on duplicate subjects.

THE CONDITIONS, and what each one isolates:

  mb1f  recording-scoped guard, calib/eval split by recording, ALL stories in pretraining
        -> the original, confounded protocol. This is what the reported numbers came from.
  mb1g  stimulus-scoped guard, calib/eval split by STORY, all stories still in pretraining
        -> isolates the guard + split change alone.
  mb1h  stimulus guard, story 3 removed from PRETRAINING and calibration, evaluated on story 3
        -> no brain in the training set ever heard the evaluation stimulus.
  mb1k  identical pretraining set to mb1h (story 3 excluded), but evaluated on story 0, which
        pretraining DID see
        -> holds pretraining VOLUME fixed and flips only the stimulus-overlap bit. Without this,
           "eval stimulus unseen" and "25% less pretraining data" are confounded.
  mb1j  story 0 withheld instead of story 3 (same subjects)  -> is any reversal story-specific?
        plus a story-3 extension to the remaining subjects -> raises n.
  mb1i  tuned from-scratch baseline (lr x wd grid, selected on VALIDATION) in the clean protocol
        -> four 2025-26 audits find tuned supervised baselines matching brain FMs; an untuned
           from-scratch arm is a strawman and the likeliest rejection reason.

Usage: python mb_conditions.py [results_dir]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_io import load_subject_results

R = sys.argv[1] if len(sys.argv) > 1 else "$SCRATCH/project/results/mb"
CONDS = [("mb1f", "result_mb1f_ksweep_"), ("mb1g", "result_mb1g_clean_"),
         ("mb1h", "result_mb1h_stimho_"), ("mb1k", "result_mb1k_volctl_"),
         ("mb1j0", "result_mb1j_story0_"), ("mb1j3", "result_mb1j_story3ext_"),
         ("mb1i", "result_mb1i_tuned_")]
KS = [100, 200, 500, 1000, 2000]
BOOT = 20000


def ci(d, seed=0):
    d = np.asarray(d, float)
    if len(d) < 3:
        return d.mean(), np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
    return d.mean(), *np.percentile(m, [2.5, 97.5])


def cell(j, arm, k, field="fm"):
    """(x_chance, accuracy_pct, mean_percentile) for one arm/budget, or (None, None, None).

    mean_percentile is the POOL-INVARIANT one and is the only metric that may be compared ACROSS
    conditions here: the conditions have different candidate pools (mb1f ~3626, mb1g ~2406, mb1h
    ~3002), and an earlier analysis is precisely the argument that x chance is not
    pool-invariant. Comparing accuracy across conditions is worse still -- it rises when the pool
    shrinks even at fixed skill, which is exactly what mb1f -> mb1g does.
    """
    for p in j.get("results", {}).get(arm, {}).get("curve", []):
        if p["k"] == k and field in p and p[field].get("pool"):
            return p[field]["x"], p[field]["x"] * 10.0 / p[field]["pool"] * 100.0, p[field].get("meanpct")
    return None, None, None


D = {}
for name, pref in CONDS:
    try:
        D[name] = load_subject_results(R, pref, quiet=True)
    except RuntimeError as e:
        print(f"[{name}] DUPLICATE SUBJECT -- {e}")
        D[name] = {}
print("\ncondition : n subjects")
for name, _ in CONDS:
    print(f"  {name:6s}: {len(D.get(name, {})):2d}  {sorted(D.get(name, {}))}")

# ---- 1. attribution: which change causes the collapse? (conv arm, paired on common subjects) ----
print("\n== 1. ATTRIBUTION (conv arm, accuracy %, paired on subjects common to all listed) ==")
for chain in [("mb1f", "mb1g", "mb1h"), ("mb1h", "mb1k")]:
    common = sorted(set.intersection(*[set(D.get(c, {})) for c in chain])) if all(D.get(c) for c in chain) else []
    if len(common) < 2:
        print(f"  {' -> '.join(chain)}: n={len(common)}, waiting"); continue
    print(f"  {' -> '.join(chain)}   n={len(common)}")
    for label, idx in (("mean_percentile (POOL-INVARIANT, the valid one)", 2),
                       ("x chance   (pool-normalised but not pool-invariant)", 0),
                       ("accuracy % (NOT comparable: rises as pool shrinks)", 1)):
        print(f"    -- {label}")
        for k in [100, 500, 2000]:
            vals = []
            for c in chain:
                a = [cell(D[c][s], "conv", k)[idx] for s in common]
                a = [v for v in a if v is not None]
                vals.append(f"{np.mean(a):7.3f}" if a else "   --  ")
            print(f"      k={k:>5}: " + "  ->  ".join(vals))

# ---- 2. H1 re-test per condition ----
print("\n== 2. H1 (dynamic_big - conv, MEAN PERCENTILE -- the only metric admissible across")
print("      protocols; pools differ up to 7x between conditions. 2026-08-10: this section")
print("      previously printed x chance and the paper shipped it; see the ELN of that date.) ==")
for name, _ in CONDS:
    subs = sorted(D.get(name, {}))
    if len(subs) < 3:
        continue
    out = []
    for k in [100, 500, 2000]:
        d = []
        for s in subs:
            c = cell(D[name][s], "conv", k)[2]      # [2] = mean_percentile. Touching [0]
            y = cell(D[name][s], "dynamic_big", k)[2]  # (x chance) across conditions needs a reason.
            if c is not None and y is not None:
                d.append(y - c)
        if len(d) < 3:
            out.append(f"k{k}: n<3"); continue
        m, lo, hi = ci(d)
        tag = "DYN" if lo > 0 else ("CONV" if hi < 0 else "ns")
        out.append(f"k{k}: {m:+5.2f}[{lo:+.2f},{hi:+.2f}]{tag}")
    print(f"  {name:6s} n={len(subs):2d}  " + "  ".join(out))

# ---- 3. tuned vs untuned from-scratch ----
print("\n== 3. FROM-SCRATCH BASELINE: untuned (mb1h) vs TUNED grid (mb1i), and vs pretrained ==")
if D.get("mb1i"):
    subs = sorted(set(D["mb1i"]) & set(D.get("mb1h", {})))
    print(f"  subjects with both: {subs}")
    for k in [100, 500, 2000]:
        rows = []
        for s in subs:
            fm = cell(D["mb1i"][s], "conv", k)[0]
            sc = cell(D["mb1i"][s], "conv", k, field="scratch")[0]
            su = cell(D["mb1h"][s], "conv", k, field="scratch")[0]
            if None not in (fm, sc, su):
                rows.append((fm, sc, su))
        if len(rows) < 2:
            print(f"  k={k:>5}: n={len(rows)}, waiting"); continue
        fm = np.mean([r[0] for r in rows]); tu = np.mean([r[1] for r in rows])
        un = np.mean([r[2] for r in rows])
        verdict = "PRETRAIN WINS" if fm > tu * 1.2 else "TUNED SCRATCH CLOSES THE GAP"
        print(f"  k={k:>5}: pretrained {fm:6.2f}x | tuned scratch {tu:6.2f}x | untuned scratch {un:5.2f}x"
              f"   -> {verdict}   n={len(rows)}")
    # which hyperparameters won, for the paper
    seen = {}
    for s in subs:
        for p in D["mb1i"][s].get("results", {}).get("conv", {}).get("curve", []):
            sc = p.get("scratch", {})
            if "tuned_lr" in sc:
                seen.setdefault(p["k"], []).append((sc["tuned_lr"], sc["tuned_wd"]))
    for k in sorted(seen):
        print(f"    k={k:>5} winning (lr,wd): {seen[k]}")
else:
    print("  mb1i not present yet")
