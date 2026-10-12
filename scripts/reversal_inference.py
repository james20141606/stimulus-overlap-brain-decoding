#!/usr/bin/env python3
"""Crossed subject x withheld-story inference for the architecture-reversal claim.

WHY THIS EXISTS

The paper reports the dynamic-minus-conv difference pooled over "14 subject-conditions spanning two
withheld stories and ten subjects" and quotes a bootstrap CI excluding zero. An adversarial review
made the obvious objection: those 14 cells are not 14 independent units. They reuse ten subjects and
two withheld stories, so a bootstrap over cells prices neither dependency axis and the interval is too
narrow. The same objection was raised against this project's other paper and answered there with
`crossed_inference.py`; this is that machinery applied to the reversal.

It computes, on the same per-cell differences:
  1. the naive cell-level bootstrap        -- what the paper currently reports
  2. a subject-clustered bootstrap         -- resample the ten subjects
  3. a story-clustered bootstrap           -- resample the withheld stories (n=2, so this is weak by
                                              construction and is reported to show that it is)
  4. a TWO-WAY crossed cluster bootstrap   -- resample subjects AND stories jointly

Reading rule fixed in advance: if the two-way interval covers zero, the paper claims a reversal in the
POINT ESTIMATE and states that significance is not established at this number of withheld stories. It
does not claim a significant inversion. With only two withheld stories the story axis cannot support
much, and saying so is cheaper than being told.

Usage (on the cluster): python3 scripts/reversal_inference.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_io import load_subject_results

R = "$SCRATCH/project/results/mb"
# clean conditions: the evaluation story was withheld from pretraining
CONDS = [("story3", "result_mb1h_stimho_"),
         ("story0", "result_mb1j_story0_"),
         ("story3", "result_mb1j_story3ext_")]
BOOT = 20000


def x_of(j, arm, k):
    for p in j.get("results", {}).get(arm, {}).get("curve", []):
        if p["k"] == k and "fm" in p:
            # mean percentile, not x chance: the clean conditions this script pools have different
            # candidate pools, and only a rank statistic is admissible across them (2026-08-10).
            return p["fm"].get("meanpct")
    return None


def ci(vals, lo=2.5, hi=97.5):
    return np.percentile(vals, [lo, hi])


def main():
    cells = []          # (subject, story, k, dynamic - conv)
    for story, prefix in CONDS:
        try:
            D = load_subject_results(R, prefix, quiet=True)
        except RuntimeError as e:
            print(f"[{prefix}] duplicate subject -- {e}"); continue
        for s, j in D.items():
            for k in (100, 500, 2000):
                c, y = x_of(j, "conv", k), x_of(j, "dynamic_big", k)
                if c is not None and y is not None:
                    cells.append((s, story, k, y - c))
    if not cells:
        sys.exit("no clean-condition results found")

    subs = sorted({c[0] for c in cells})
    stories = sorted({c[1] for c in cells})
    print(f"cells {len(cells)} | subjects {len(subs)} | withheld stories {stories}\n")

    rng = np.random.default_rng(0)
    for k in (100, 500, 2000):
        rows = [(s, st, d) for s, st, kk, d in cells if kk == k]
        if len(rows) < 3:
            print(f"k={k}: only {len(rows)} cells, skipped"); continue
        d = np.array([r[2] for r in rows])
        S = sorted({r[0] for r in rows}); T = sorted({r[1] for r in rows})
        by_s = {u: np.array([r[2] for r in rows if r[0] == u]) for u in S}
        by_t = {v: np.array([r[2] for r in rows if r[1] == v]) for v in T}

        # 1. naive: resample cells
        naive = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
        # 2. subject-clustered
        subj = np.array([np.concatenate([by_s[S[i]] for i in rng.integers(0, len(S), len(S))]).mean()
                         for _ in range(BOOT // 4)])
        # 3. story-clustered
        stry = np.array([np.concatenate([by_t[T[i]] for i in rng.integers(0, len(T), len(T))]).mean()
                         for _ in range(BOOT // 4)])
        # 4. two-way crossed: resample both axes, keep the intersection
        two = []
        cell = {(r[0], r[1]): [] for r in rows}
        for r in rows:
            cell[(r[0], r[1])].append(r[2])
        for _ in range(BOOT // 4):
            si = [S[i] for i in rng.integers(0, len(S), len(S))]
            ti = [T[i] for i in rng.integers(0, len(T), len(T))]
            vals = [v for u in si for w in ti for v in cell.get((u, w), [])]
            if vals:
                two.append(np.mean(vals))
        two = np.array(two)

        print(f"k={k}   mean {d.mean():+.2f}   n_cells={len(d)}  n_subj={len(S)}  n_story={len(T)}")
        for name, b in (("naive (cells)", naive), ("subject-clustered", subj),
                        ("story-clustered", stry), ("TWO-WAY crossed", two)):
            lo, hi = ci(b)
            verdict = "excludes 0" if (lo > 0 or hi < 0) else "covers 0"
            print(f"    {name:20s} [{lo:+.2f}, {hi:+.2f}]  {verdict}")
        print()

    print("Reading rule: the TWO-WAY interval is the honest one. With only two withheld stories the")
    print("story axis is weak by construction; if it covers zero, report a reversal in the point")
    print("estimate and say significance is not established at this number of stories.")


if __name__ == "__main__":
    main()
