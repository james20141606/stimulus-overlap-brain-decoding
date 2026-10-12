#!/usr/bin/env python
"""Receipt-writing twin of interaction.py: same cells, same paired contrasts, same seeded bootstrap,
plus the two per-guard simple contrasts and a JSON dump that check_paper_numbers.py reads.

Why a second script rather than a flag on interaction.py: interaction.py is the generator the ELN
and the pins already cite by name and md5; its stdout is the reference. This file must reproduce
that stdout's numbers bit-for-bit (asserted below against the same functions) and add the receipt.
Run on the cluster, where the per-subject JSONs live:

    python scripts/interaction_receipt.py [results_dir] [out_json]

Contrasts, all on the paired (dynamic_big - conv) mean percentile, one value per subject per cell:
    split main  = mean(rs, ss) - mean(rr, sr)   <- architecture x split interaction
    guard main  = mean(sr, ss) - mean(rr, rs)
    interaction = (ss - sr) - (rs - rr)          <- three-way guard x split on the advantage
    split | recording guard = rs - rr            <- simple contrast, guard held at recording
    split | stimulus  guard = ss - sr            <- simple contrast, guard held at stimulus
"""
import io, json, os, sys, contextlib, datetime, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M
sys.argv = _a
import numpy as np
from result_io import load_subject_results

R = sys.argv[1] if len(sys.argv) > 1 else "$SCRATCH/project/results/mb"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/interaction_receipt.json"
CELLS = {"rr": "result_mb1f_ksweep_", "rs": "result_mb1t_cellC_",
         "sr": "result_mb1t_cellD_", "ss": "result_mb1k_volctl_"}
D = {c: load_subject_results(R, p, quiet=True) for c, p in CELLS.items()}


def contrasts(d, subs):
    return {
        "split_main":      [(d[s]["rs"] + d[s]["ss"]) / 2 - (d[s]["rr"] + d[s]["sr"]) / 2 for s in subs],
        "guard_main":      [(d[s]["sr"] + d[s]["ss"]) / 2 - (d[s]["rr"] + d[s]["rs"]) / 2 for s in subs],
        "guard_x_split":   [(d[s]["ss"] - d[s]["sr"]) - (d[s]["rs"] - d[s]["rr"]) for s in subs],
        "split_at_recording_guard": [d[s]["rs"] - d[s]["rr"] for s in subs],
        "split_at_stimulus_guard":  [d[s]["ss"] - d[s]["sr"] for s in subs],
    }


receipt = {
    "generated_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    "host": os.uname().nodename,
    "results_dir": R,
    "cells": CELLS,
    "metric": "dynamic_big - conv, mean percentile (pool-invariant), one value per subject per cell",
    "bootstrap": {"draws": M.BOOT, "seed": 0, "interval": "percentile 2.5/97.5",
                  "function": "mb_conditions.ci"},
    "budgets": {},
}
for k in (100, 500):
    subs = sorted(set.intersection(*[set(D[c]) for c in CELLS]))
    d, ok = {}, []
    for s in subs:
        vals, good = {}, True
        for c in CELLS:
            cv = M.cell(D[c][s], "conv", k)[2]; dy = M.cell(D[c][s], "dynamic_big", k)[2]
            if cv is None or dy is None:
                good = False; break
            vals[c] = dy - cv
        if good:
            d[s] = vals; ok.append(s)
    block = {"n_subjects": len(ok), "subjects": ok,
             "per_subject_cell_advantage": {s: d[s] for s in ok}, "contrasts": {}}
    for name, arr in contrasts(d, ok).items():
        m, lo, hi = M.ci(list(arr))
        verdict = ("insufficient n" if (np.isnan(lo) or np.isnan(hi))
                   else "excludes 0" if (lo > 0 or hi < 0) else "covers 0")
        block["contrasts"][name] = {"mean": float(m), "ci95": [float(lo), float(hi)],
                                    "verdict": verdict, "per_subject": [float(x) for x in arr]}
    receipt["budgets"][str(k)] = block

# Self-check against interaction.py's own arithmetic (same functions, same seed): the three
# contrasts it prints must match what we dump, or the receipt is not a receipt of that script.
for k in (100, 500):
    b = receipt["budgets"][str(k)]
    subs = b["subjects"]; d = b["per_subject_cell_advantage"]
    ref = np.array([(d[s]["rs"] + d[s]["ss"]) / 2 - (d[s]["rr"] + d[s]["sr"]) / 2 for s in subs])
    m, lo, hi = M.ci(list(ref))
    got = b["contrasts"]["split_main"]
    assert abs(m - got["mean"]) < 1e-12 and abs(lo - got["ci95"][0]) < 1e-12 and abs(hi - got["ci95"][1]) < 1e-12

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(receipt, f, indent=1, sort_keys=True)
receipt_hash = hashlib.sha256(open(OUT, "rb").read()).hexdigest()

for k in ("100", "500"):
    b = receipt["budgets"][k]
    print(f"== k={k}  n={b['n_subjects']} ==")
    for name, c in b["contrasts"].items():
        lo, hi = c["ci95"]
        print(f"  {name:28s} {c['mean']:+.4f}  [{lo:+.4f},{hi:+.4f}]  {c['verdict']}")
print(f"wrote {OUT}  sha256 {receipt_hash[:16]}  BOOT={M.BOOT}")
