#!/usr/bin/env python3
"""Verdict for the head-refit fusion (R10's "re-fit the head jointly" experiment).

Two arms, both starting exactly at the fitted timing solution, both with the target head unfrozen in
phase 2:
  +brain+head   : the brain residual is trained too.
  nullbrain+head: the brain is excluded from the optimizer (its zero-init weights never move, so
                  brain(x)==0), so this arm measures ONLY what the extra head-training epochs buy.

Each fold logs `guarded` (phase-2 end) and `guarded_timing` (phase-1 end = frozen timing). The head
epochs help both arms, so the brain's actual contribution is the DIFFERENCE OF DIFFERENCES:

    brain_effect = (UH.guarded - UH.guarded_timing) - (NULL.guarded - NULL.guarded_timing)

paired per (task, seed), then reported run-level and story-clustered (the conservative unit used
everywhere else in the paper). Run on the cluster (has scipy + the result JSONs).
"""
import json, glob, collections
import numpy as np
from scipy import stats

ROOT = "$SCRATCH/project/results/m1"


def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/result_{tag}_task*_seed*.json")):
        d = json.load(open(f))
        # per-fold within-run mean of (phase2 - phase1) over subjects
        deltas = [v["guarded"]["top10_over_chance"] - v["guarded_timing"]["top10_over_chance"]
                  for v in d["per_subject"].values()]
        p2 = np.mean([v["guarded"]["top10_over_chance"] for v in d["per_subject"].values()])
        p1 = np.mean([v["guarded_timing"]["top10_over_chance"] for v in d["per_subject"].values()])
        out[(str(d["holdout_task"]), d["seed"])] = {"delta": float(np.mean(deltas)),
                                                    "p2": float(p2), "p1": float(p1)}
    return out


def report(name, arr, base):
    a = np.array(arr)
    if len(a) < 2:
        print(f"  {name}: n={len(a)} (need >=2)"); return
    t = stats.ttest_1samp(a, 0)
    ci = stats.t.interval(0.95, len(a) - 1, loc=a.mean(), scale=stats.sem(a))
    print(f"  {name}: n={len(a)}  mean={a.mean():+.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"p={t.pvalue:.3f}  = [{100*ci[0]/base:+.2f}%,{100*ci[1]/base:+.2f}%] of timing")


def main():
    UH = load("addUH")
    NU = load("addNULL")
    keys = sorted(set(UH) & set(NU))
    print(f"paired (task,seed) cells: {len(keys)}  [UH={len(UH)} NULL={len(NU)}]")
    if not keys:
        print("no complete pairs yet"); return
    base = np.mean([UH[k]["p1"] for k in keys])   # timing baseline (phase-1), same in both arms
    print(f"timing baseline (phase-1 mean) = {base:.2f}x\n")

    # raw phase-2 gains per arm (each conflates brain with the extra head epochs)
    print("Phase-2 gain over frozen timing, per arm (NOT the verdict -- head epochs help both):")
    report("+brain+head   ", [UH[k]["delta"] for k in keys], base)
    report("nullbrain+head", [NU[k]["delta"] for k in keys], base)
    print()

    # the verdict: difference of differences, run-level and story-clustered
    dod = {k: UH[k]["delta"] - NU[k]["delta"] for k in keys}
    print("BRAIN EFFECT = (+brain+head gain) - (nullbrain+head gain):")
    report("run-level     ", list(dod.values()), base)
    by_story = collections.defaultdict(list)
    for (t, s), v in dod.items():
        by_story[t].append(v)
    report("story-clustered", [np.mean(v) for v in by_story.values()], base)
    print("  per-story:", {t: round(float(np.mean(v)), 3) for t, v in sorted(by_story.items())})


if __name__ == "__main__":
    main()
