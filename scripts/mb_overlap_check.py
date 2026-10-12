#!/usr/bin/env python
"""Two verifications paper 3 and paper 1 currently assert from indirect evidence.

A. REALISED calibration/evaluation story overlap for MB. paper_manifold.tex states a STRUCTURAL
   estimate -- for the canonical four-stories-in-two-sessions layout a random half-split leaves a
   story on both sides with probability ~0.914. That is a probability, not a measurement. The split
   is deterministic (`split_by_recording`, seed=0, and mb1f never passed --seed), so the realised
   overlap can simply be recomputed and the estimate replaced by a count.

B. MB-1e zero-calibration means, which paper 1 (main.tex, sec:survive) now quotes as
   conv 0.894 / noflow 0.897 / flow 0.909 to argue the flow feature carries ~4/5 of the +0.015.
   Those came from an ELN entry; check them against the result JSONs.

Usage: python mb_overlap_check.py
"""
import sys
import glob
import json
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_io import load_subject_results

META = "$SCRATCH/project/meg_papers/data/meg_masc_full/trial_metadata.tsv"
R = "$SCRATCH/project/results/mb"
SUBS = ["01", "02", "03", "05", "07", "08", "09", "10", "11", "13"]
REC_KEYS = ["subject", "session", "task", "run"]


def split_by_recording(recs, frac_eval=0.5, seed=0):
    """Byte-for-byte the logic in src/train_mb.py:87-95, on the unique recording ids."""
    recs = sorted(set(recs))
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    n_eval = max(1, int(round(len(recs) * frac_eval)))
    return set(recs[n_eval:]), set(recs[:n_eval])       # calib, eval


print("== A. realised calib/eval STORY overlap per held-out subject (seed=0) ==")
meta = pd.read_csv(META, sep="\t", dtype={k: str for k in REC_KEYS})
n_over, rows = 0, []
for s in SUBS:
    sub = meta[meta["subject"] == s]
    if not len(sub):
        print(f"  s{s}: no metadata rows"); continue
    recs = ["|".join(r) for r in sub[REC_KEYS].drop_duplicates().itertuples(index=False)]
    calib, ev = split_by_recording(recs)
    story = lambda rs: {r.split("|")[2] for r in rs}     # task field = story
    shared = story(calib) & story(ev)
    n_over += bool(shared)
    rows.append((s, len(recs), len(calib), len(ev), sorted(shared)))
    print(f"  s{s}: {len(recs)} recordings -> calib {len(calib)} / eval {len(ev)} | "
          f"stories on BOTH sides: {sorted(shared) if shared else 'none'}")
print(f"  => {n_over}/{len(rows)} subjects have at least one story in both calibration and evaluation.")
print(f"     (paper currently states a structural estimate of ~0.914; realised = {n_over}/{len(rows)})\n")

print("== B. MB-1e zero-calibration means (paper 1 sec:survive quotes 0.894 / 0.897 / 0.909) ==")
ARMS = {"conv": "conv", "dynamic_big_noflow": "noflow", "dynamic_big": "flow"}
vals = {v: [] for v in ARMS.values()}
for _s, d in sorted(load_subject_results(R, "result_mb1e_fullres_", expect=10).items()):
    for arm, nice in ARMS.items():
        r = d.get("results", {}).get(arm, {})
        z = r.get("pretrained_zerocalib")
        if isinstance(z, dict) and "meanpct" in z:
            vals[nice].append(z["meanpct"])
for nice in ["conv", "noflow", "flow"]:
    v = vals[nice]
    if v:
        print(f"  {nice:7s} n={len(v):2d}  mean meanpct = {np.mean(v):.4f}")
if all(vals[k] for k in vals):
    c, nf, fl = (np.array(vals[k]) for k in ["conv", "noflow", "flow"])
    print(f"\n  architecture (noflow-conv) = {np.mean(nf - c):+.4f}")
    print(f"  flow         (flow-noflow) = {np.mean(fl - nf):+.4f}")
    print(f"  total        (flow-conv)   = {np.mean(fl - c):+.4f}   flow wins {int((fl > c).sum())}/{len(c)}")
