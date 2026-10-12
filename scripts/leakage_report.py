#!/usr/bin/env python3
"""Turn a dataset's own metadata into a leakage audit, without needing anyone's model or results.

`stimulus_leakage_check.py` audits a split you already have. This audits a DATASET: given how a corpus
is organised, it reports what a conventional cross-subject protocol would leak, and what the corrected
protocol costs. That is the question a practitioner actually faces before writing any code -- "if I use
the standard recipe on this corpus, what am I measuring?" -- and it can be answered from the metadata
alone.

The output is deliberately blunt: a table of what each protocol shares between fitting and evaluation,
and the data cost of fixing it.

Usage:
  python leakage_report.py --manifest manifest.csv
  python leakage_report.py --demo meg-masc          # the corpus analysed in the paper

manifest.csv needs subject,stimulus and optionally session:
  subject,stimulus,session
  sub-01,lw1,0
  sub-01,lw1,1
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict

DEMOS = {
    # subject count, stimuli, presentations each -- MEG-MASC as used in the paper
    "meg-masc": dict(n_subjects=27, stimuli=["lw1", "cable_spool_fort", "easy_money",
                                             "The_Black_Willow"], presentations=2),
}


def rows_from_demo(name):
    d = DEMOS[name]
    return [(f"sub-{s:02d}", st, p)
            for s in range(1, d["n_subjects"] + 1)
            for st in d["stimuli"] for p in range(d["presentations"])]


def read_manifest(path):
    out = []
    try:
        fh = open(path, newline="")
    except OSError as e:
        raise SystemExit(f"cannot read manifest {path!r}: {e}")
    with fh:
        r = csv.DictReader(fh)
        missing = {"subject", "stimulus"} - set(r.fieldnames or [])
        if missing:
            sys.exit(f"manifest missing column(s): {', '.join(sorted(missing))}")
        for row in r:
            out.append((row["subject"].strip(), row["stimulus"].strip(),
                        (row.get("session") or "0").strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest")
    g.add_argument("--demo", choices=sorted(DEMOS))
    a = ap.parse_args()

    rows = rows_from_demo(a.demo) if a.demo else read_manifest(a.manifest)
    subs = sorted({r[0] for r in rows})
    stims = sorted({r[1] for r in rows})
    by_sub = defaultdict(set)
    for s, st, _ in rows:
        by_sub[s].add(st)
    reps = len(rows) / max(1, len(subs) * len(stims))

    print(f"corpus: {len(subs)} subjects x {len(stims)} stimuli"
          f"{f' x {reps:.0f} presentations' if reps > 1.05 else ''}")
    print(f"stimuli: {', '.join(stims[:6])}{' ...' if len(stims) > 6 else ''}\n")

    held = subs[-1]
    others = set(subs[:-1])
    seen_by_others = {st for s in others for st in by_sub[s]}
    shared = by_sub[held] & seen_by_others

    print("PROTOCOL AUDIT -- hold out one subject, as is conventional")
    print(f"  {'protocol':<44}{'eval stimuli seen in fitting':>30}")
    print(f"  {'-'*44}{'-'*30}")
    print(f"  {'hold out subject only':<44}"
          f"{f'{len(shared)}/{len(by_sub[held])}  ({len(shared)/max(1,len(by_sub[held])):.0%})':>30}")
    print(f"  {'+ hold out the evaluation stimulus':<44}{'0/1  (0%)':>30}")

    if reps > 1.05:
        print(f"\n  Each stimulus is presented {reps:.0f}x per subject, so within the held-out subject")
        print(f"  a recording-level calibration/evaluation split also shares the stimulus with itself.")

    n_pre = len(stims) - 1
    print("\nCOST OF THE FIX")
    print(f"  pretraining stimuli: {len(stims)} -> {n_pre}"
          f"   ({(1 - n_pre/len(stims)):.0%} less stimulus diversity)")
    if n_pre == 0 and len(subs) > 1:
        # The regime worth naming explicitly rather than leaving the reader to infer from "100% less".
        # With one stimulus there is no stimulus-disjoint cross-subject protocol AT ALL: withholding
        # the evaluation stimulus leaves nothing to pretrain or train on. No split fixes this, so the
        # honest move is to stop looking for one and scope the claim instead.
        print("  => NOT FIXABLE BY ANY SPLIT. Every subject heard the same single stimulus, so")
        print("     withholding it from fitting leaves nothing to fit on. A cross-subject transfer")
        print("     number computed here cannot be separated from shared exposure by any protocol.")
        print("     Such corpora can support within-subject claims; report those instead.")
        print("  usable evaluation stimuli: 0 -- there is no clean cross-subject measurement here.")
    else:
        if len(stims) <= 4:
            print(f"  with only {len(stims)} stimuli, withholding one is a large fraction of the "
                  f"corpus;")
            print(f"  report the volume-matched and composition-matched variants and bracket the "
                  f"estimate.")
        print(f"  usable evaluation stimuli: {len(stims)} (one clean measurement each)")

    print("\nWHAT TO REPORT")
    print("  1. the score under the conventional protocol, and under stimulus holdout, side by side")
    print("  2. a stimulus-derived baseline with no brain data, in the same protocol")
    print("  3. a rank statistic, since the candidate pool changes between protocols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
