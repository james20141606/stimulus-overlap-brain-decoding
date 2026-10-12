#!/usr/bin/env python3
"""stimulus_leakage_check — does your cross-subject split let the model see the evaluation stimulus?

WHAT THIS IS FOR

In naturalistic neuroimaging every participant is given the same stimuli, because cross-subject
alignment requires it. So a model pretrained (or trained) on other subjects has usually already been
fitted to the very material it is evaluated on, through those subjects' recordings. Holding out the
SUBJECT does not hold out the STIMULUS.

That overlap is worth a large fraction of reported cross-subject performance. On MEG-MASC, holding the
evaluation narrative fixed and varying only whether it appeared in pretraining removes roughly half of
the above-chance signal; on Narratives fMRI an analogous held-out-material manipulation leaves about a quarter. It can also
change conclusions: in the accompanying paper, a pre-specified architecture comparison had no
detectable advantage once the overlap was removed.

Unlike benchmark contamination in language models, this is not an accident you can clean up. Remove the
shared stimulus and you remove the premise cross-subject alignment rests on. The point of this script
is therefore NOT to tell you to eliminate the overlap. It is to tell you how much you have, so you can
report performance with and without it.

USAGE

  python stimulus_leakage_check.py --manifest manifest.csv
  python stimulus_leakage_check.py --manifest manifest.csv --propose-split --out clean_split.json

  manifest.csv needs three columns; extra columns are ignored:
      subject,stimulus,split
      sub-01,pieman,train
      sub-02,pieman,train
      sub-27,pieman,test          <- held-out SUBJECT, but the stimulus is in train: leakage

  `split` values are matched case-insensitively. Anything in {train, pretrain, pretraining, fit,
  calib, calibration} counts as model-fitting; anything in {test, eval, evaluation, val, valid,
  validation, holdout} counts as evaluation.

WHAT IT REPORTS

  * per evaluation split: the fraction of evaluation ROWS whose stimulus also appears in a fitting
    split, and the fraction of distinct evaluation STIMULI that do
  * whether subjects are disjoint, so you can distinguish "no leakage" from "no subject holdout"
  * with --propose-split: a stimulus-disjoint reassignment (stimuli only; it does NOT manage
    subject disjointness -- combine with a subject holdout, as the emitted rule says)

EXIT STATUS
  0  no evaluation stimulus appears in any fitting split
  1  leakage found (or the manifest could not be read)

The exit status makes this usable in CI. Note that status 1 is not a verdict that your result is
wrong -- for many designs the overlap is unavoidable. It is a prompt to report both numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict

FIT = {"train", "pretrain", "pretraining", "fit", "calib", "calibration"}
EVAL = {"test", "eval", "evaluation", "val", "valid", "validation", "holdout"}


def read_manifest(path):
    """Rows of (subject, stimulus, split). Requires the three named columns; tolerates others."""
    rows = []
    try:
        fh = open(path, newline="")
    except OSError as e:
        raise SystemExit(f"cannot read manifest {path!r}: {e}")
    with fh:
        rdr = csv.DictReader(fh)
        missing = {"subject", "stimulus", "split"} - set(rdr.fieldnames or [])
        if missing:
            raise SystemExit(f"manifest is missing column(s): {', '.join(sorted(missing))}\n"
                             f"found: {', '.join(rdr.fieldnames or ['<none>'])}")
        for r in rdr:
            s = (r["split"] or "").strip().lower()
            if not s:
                continue
            rows.append((str(r["subject"]).strip(), str(r["stimulus"]).strip(), s))
    if not rows:
        raise SystemExit("manifest contained no usable rows")
    return rows


def analyse(rows):
    fit_stim, fit_subj = set(), set()
    by_eval = defaultdict(list)
    unknown = set()
    for subj, stim, split in rows:
        if split in FIT:
            fit_stim.add(stim); fit_subj.add(subj)
        elif split in EVAL:
            by_eval[split].append((subj, stim))
        else:
            unknown.add(split)
    return fit_stim, fit_subj, by_eval, unknown


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with columns subject,stimulus,split")
    ap.add_argument("--propose-split", action="store_true",
                    help="emit a stimulus-disjoint reassignment")
    ap.add_argument("--frac-eval", type=float, default=0.25,
                    help="fraction of stimuli to reserve for evaluation in the proposal")
    ap.add_argument("--out", default="", help="write the proposal here as JSON")
    a = ap.parse_args()

    rows = read_manifest(a.manifest)
    fit_stim, fit_subj, by_eval, unknown = analyse(rows)

    print(f"rows {len(rows)} | fitting stimuli {len(fit_stim)} | fitting subjects {len(fit_subj)}")
    if unknown:
        print(f"  note: ignored unrecognised split label(s): {', '.join(sorted(unknown))}")
    if not by_eval:
        raise SystemExit("no evaluation rows found -- expected one of: " + ", ".join(sorted(EVAL)))

    leaked_any = False
    for split in sorted(by_eval):
        pairs = by_eval[split]
        stims = {s for _, s in pairs}
        subjs = {u for u, _ in pairs}
        shared = stims & fit_stim
        n_rows_leaked = sum(1 for _, s in pairs if s in fit_stim)
        row_frac = n_rows_leaked / len(pairs)
        stim_frac = len(shared) / len(stims) if stims else 0.0
        subj_overlap = subjs & fit_subj
        leaked_any |= bool(shared)

        print(f"\n[{split}] {len(pairs)} rows, {len(stims)} distinct stimuli, {len(subjs)} subjects")
        print(f"  stimulus leakage : {len(shared)}/{len(stims)} stimuli "
              f"({stim_frac:6.1%}) also in a fitting split")
        print(f"                     {n_rows_leaked}/{len(pairs)} rows ({row_frac:6.1%})")
        print(f"  subject holdout  : {'YES' if not subj_overlap else 'NO'}"
              f"{'' if not subj_overlap else f' -- {len(subj_overlap)} subject(s) also in fitting'}")
        if shared:
            show = sorted(shared)[:6]
            print(f"  leaked stimuli   : {', '.join(show)}"
                  f"{' ...' if len(shared) > len(show) else ''}")
            if not subj_overlap:
                print("  => held-out SUBJECTS but shared STIMULI. Cross-subject scores here include")
                print("     the model's prior exposure to this material through other participants.")
                print("     Report the stimulus-held-out number alongside this one.")
        else:
            print("  => no evaluation stimulus appears in any fitting split.")

    if a.propose_split:
        all_stim = sorted({s for _, s, _ in rows})
        n_eval = max(1, round(len(all_stim) * a.frac_eval))
        # deterministic: reserve the lexicographically last stimuli, so the proposal is reproducible
        # without a seed and reviewers can regenerate it exactly.
        ev = all_stim[-n_eval:]
        proposal = {"evaluation_stimuli": ev,
                    "fitting_stimuli": [s for s in all_stim if s not in set(ev)],
                    "rule": "no stimulus appears in both partitions; combine with a subject holdout "
                            "for a design that is disjoint in both subject and stimulus"}
        print(f"\nproposed stimulus-disjoint split: {len(proposal['fitting_stimuli'])} fitting / "
              f"{len(ev)} evaluation stimuli")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(proposal, fh, indent=2)
            print(f"written to {a.out}")

    return 1 if leaked_any else 0


if __name__ == "__main__":
    sys.exit(main())
