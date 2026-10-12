#!/usr/bin/env python3
"""Crossed subject x story inference for the 40 crossed folds.

Why: the paper reports a paired t(39) over 40 subject x story cells. Two AC reviews objected that
those 40 cells are not 40 independent units -- they reuse 10 subjects and 4 stories, with overlapping
training sets -- so t(39) pseudoreplicates. Collapsing to 4 stories fixes one dependency axis but
leaves n=4, where the distribution-free floor is p=0.125.

This computes, on the same per-fold differences:
  1. the naive paired t(39)                     (what the paper currently reports)
  2. a subject-clustered bootstrap  (resample the 10 subjects)
  3. a story-clustered bootstrap    (resample the 4 stories)
  4. a TWO-WAY crossed cluster bootstrap (resample subjects AND stories jointly)
The two-way interval is the honest headline: it prices both dependency axes at once.

Usage (on the cluster): python3 scripts/crossed_inference.py
"""
import glob
import os
import json
import re

import numpy as np

R = "$SCRATCH/project/results/m1"
KEY = "mean_meanpct"      # pool-normalized percentile: comparable, pool-invariant


def load(pattern):
    # Hardened 2026-08-19 (audit): the previous loader had the exact shape that corrupted a mean on
    # 2026-07-31 -- unsorted glob, silent last-wins on duplicate (task,subj), no smoke filter, and a
    # bare except that swallowed unreadable files. Every failure below is now loud.
    _skip = ("smoke", "debug", "tmp")
    out = {}
    for f in sorted(glob.glob(f"{R}/{pattern}")):
        b = os.path.basename(f)
        if any(k in b.lower() for k in _skip):
            print(f"[load] skip {b}")
            continue
        m = re.search(r"task(\d+)_subj(\d+)\.json$", f)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key in out:
            raise RuntimeError(f"duplicate cell {key}: {b} collides with an earlier file")
        try:
            out[key] = json.load(open(f))["summary"][KEY]
        except (KeyError, ValueError, OSError) as e:
            raise RuntimeError(f"unreadable result {b}: {e}") from e
    return out


timing, brain = load("result_nuis_crossed_task*_subj*.json"), load("result_story_crossed_task*_subj*.json")
cells = sorted(set(timing) & set(brain))
d = np.array([timing[c] - brain[c] for c in cells])
story = np.array([c[0] for c in cells])
subj = np.array([c[1] for c in cells])
print(f"paired cells: {len(cells)}  stories: {sorted(set(story))}  subjects: {len(set(subj))}")
print(f"timing - brain ({KEY}): mean {d.mean():+.4f}, wins {int((d>0).sum())}/{len(d)}\n")

# 1. naive paired t
from math import sqrt
se = d.std(ddof=1) / sqrt(len(d))
t = d.mean() / se
print(f"1. naive paired t({len(d)-1}) = {t:.2f}   (treats all {len(d)} cells as independent)")

rng = np.random.default_rng(0)
B = 20000


def boot(kind):
    out = np.empty(B)
    us, ut = np.unique(subj), np.unique(story)
    for b in range(B):
        if kind == "subject":
            keep = np.isin(subj, rng.choice(us, us.size, replace=True))
            sel = np.concatenate([np.flatnonzero(subj == s)
                                  for s in rng.choice(us, us.size, replace=True)])
        elif kind == "story":
            sel = np.concatenate([np.flatnonzero(story == s)
                                  for s in rng.choice(ut, ut.size, replace=True)])
        else:  # two-way crossed: resample both margins
            ss = rng.choice(us, us.size, replace=True)
            tt = rng.choice(ut, ut.size, replace=True)
            sel = np.concatenate([np.flatnonzero((subj == a) & (story == b_))
                                  for a in ss for b_ in tt])
        out[b] = d[sel].mean() if sel.size else np.nan
    return np.nanpercentile(out, [2.5, 97.5]), np.nanmean(out < 0)


for kind, label in (("subject", "2. subject-clustered"), ("story", "3. story-clustered"),
                    ("crossed", "4. TWO-WAY crossed (honest headline)")):
    ci, p_neg = boot(kind)
    print(f"{label:38} 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"frac(bootstrap mean <= 0) = {p_neg:.4f}")

print(f"\nsign test over the {len(set(story))} story means: "
      f"{int((np.array([d[story==s].mean() for s in sorted(set(story))])>0).sum())}"
      f"/{len(set(story))}; distribution-free floor p=0.125")
