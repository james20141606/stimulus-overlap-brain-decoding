#!/usr/bin/env python3
"""Assemble the primary MATCHED input set of the calibration-diversity control (PREREG Deviation 2).

Written 2026-09-25 before any MATCHED result from subjects 07-13 existed and before any clusterB job
was submitted. The selection is VALUE-BLIND: it looks only at which result/receipt files exist and at
the receipts' `written` timestamps; it never opens a result JSON's numbers.

Rule (PREREG Deviation 2, fixed before the runs):
  A  clusterA has all ten primary subjects (result + receipt, receipt written <= cut-off)
       -> primary = clusterA x10; clusterB results are descriptive only.
  B  otherwise, evaluated at the selection time --at (07:30 if clusterB has all six, else 09:30)
       -> subjects 01 02 03 05 from clusterA; subjects 07 08 09 10 11 13 from clusterB, falling back
          to clusterA for a subject only if clusterB has no complete result for it by the cut-off;
          any subject still missing -> INCOMPLETE (no paper change).
Outputs <out>/inputs (reused SEEN/UNSEEN + chosen MATCHED), <out>/receipts (chosen receipts and
preflights), <out>/descriptive_bp or descriptive_clusterA (the non-chosen MATCHED files), and
<out>/selection.json (rule, per-subject site, file SHA-256s, script SHA).
Usage: calmatch_assemble_20260925.py --reuse results/calmatch_20260924/inputs \
          --clusterA results/calmatch_20260924/clusterA --bp results/calmatch_20260924/clusterB \
          --at 2026-09-25T09:30:00 --out results/calmatch_20260924/primary
  (--clusterA/--bp each hold result_mb1z_calmatch_story0_s*.json plus receipt_/preflight_ files)
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import sys

PRIMARY = ["01", "02", "03", "05", "07", "08", "09", "10", "11", "13"]
CLUSTERA_FIRST = ["01", "02", "03", "05"]          # complete on clusterA before any clusterB job existed
BP_SIX = ["07", "08", "09", "10", "11", "13"]
CUTOFF = dt.datetime.fromisoformat("2026-09-25T09:30:00")
EARLY_B = dt.datetime.fromisoformat("2026-09-25T07:30:00")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def complete(d, s, at):
    """(result, receipt, preflight) paths if subject s is complete in dir d by time `at`, else None.
    Completion = result + receipt exist, receipt's result_sha256 equals the result file's SHA, and the
    receipt was written no later than min(at, CUTOFF). Only the receipt is parsed."""
    tag = f"mb1z_calmatch_story0_s{s}"
    res, rc, pf = (os.path.join(d, f"{p}_{tag}.json") for p in ("result", "receipt", "preflight"))
    if not (os.path.exists(res) and os.path.exists(rc)):
        return None
    r = json.load(open(rc))
    if r.get("result_sha256") != sha(res):
        return None
    w = dt.datetime.strptime(r["written"], "%Y-%m-%dT%H:%M:%S%z")
    if w > min(at, CUTOFF):
        return None
    return res, rc, (pf if os.path.exists(pf) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", required=True)
    ap.add_argument("--clusterA", required=True)
    ap.add_argument("--bp", default="")
    ap.add_argument("--at", required=True, help="selection time, ISO with offset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    at = dt.datetime.fromisoformat(a.at)

    T = {s: complete(a.clusterA, s, at) for s in PRIMARY}
    B = {s: (complete(a.bp, s, at) if a.bp else None) for s in PRIMARY}
    sel = {"script_sha256": sha(__file__), "selection_time": a.at, "cutoff": CUTOFF.isoformat(),
           "clusterA_complete": [s for s in PRIMARY if T[s]], "bp_complete": [s for s in PRIMARY if B[s]]}
    if all(T[s] for s in PRIMARY):
        rule, site = "A", {s: "clusterA" for s in PRIMARY}
    elif at < EARLY_B:
        sys.exit("REFUSING: clusterA incomplete and selection time is before 07:30; rule B may not be "
                 "evaluated yet (PREREG Deviation 2)")
    elif at < CUTOFF and not all(B[s] for s in BP_SIX):
        sys.exit("REFUSING: before 09:30 rule B needs all six clusterB subjects (PREREG Deviation 2)")
    else:
        rule, site = "B", {}
        for s in CLUSTERA_FIRST:
            site[s] = "clusterA" if T[s] else None
        for s in BP_SIX:
            site[s] = "clusterB" if B[s] else ("clusterA" if T[s] else None)
    missing = [s for s in PRIMARY if site.get(s) is None]
    sel.update({"rule": rule, "site": site, "missing": missing,
                "status": "INCOMPLETE" if missing else "COMPLETE", "files": {}})

    inp, rec = os.path.join(a.out, "inputs"), os.path.join(a.out, "receipts")
    for d in (inp, rec):
        os.makedirs(d, exist_ok=True)
        if os.listdir(d):
            sys.exit(f"REFUSING: {d} not empty")
    for f in sorted(glob.glob(os.path.join(a.reuse, "result_*.json"))):
        if "mb1z_calmatch" in os.path.basename(f):
            continue
        shutil.copy2(f, inp)
        sel["files"][os.path.basename(f)] = {"site": "reused", "sha256": sha(f)}
    for s in PRIMARY:
        if site.get(s) is None:
            continue
        res, rc, pf = (T if site[s] == "clusterA" else B)[s]
        shutil.copy2(res, inp)
        for p in (rc, pf):
            if p:
                shutil.copy2(p, rec)
        sel["files"][os.path.basename(res)] = {"site": site[s], "sha256": sha(res),
                                               "receipt_sha256": sha(rc)}
    # story 1 (secondary, descriptive) is clusterA-only (Deviation 2 (a)): copy every complete clusterA story-1 run
    for res in sorted(glob.glob(os.path.join(a.clusterA, "result_mb1z_calmatch_story1_s*.json"))):
        rc = res.replace("result_", "receipt_"); pf = res.replace("result_", "preflight_")
        if os.path.exists(rc) and json.load(open(rc)).get("result_sha256") == sha(res):
            shutil.copy2(res, inp)
            for p in (rc, pf):
                if os.path.exists(p):
                    shutil.copy2(p, rec)
            sel["files"][os.path.basename(res)] = {"site": "clusterA", "sha256": sha(res), "receipt_sha256": sha(rc)}
    # the MATCHED files that were NOT chosen stay descriptive, in their own directory
    for name, M in (("clusterA", T), ("bp", B)):
        other = [s for s in PRIMARY if M[s] and site.get(s) != ("clusterA" if name == "clusterA" else "clusterB")]
        if other:
            dd = os.path.join(a.out, f"descriptive_{name}")
            os.makedirs(dd, exist_ok=True)
            for s in other:
                for p in M[s]:
                    if p:
                        shutil.copy2(p, dd)
            sel[f"descriptive_{name}"] = other
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=2)
    print(json.dumps({k: sel[k] for k in ("rule", "site", "missing", "status")}, indent=1))
    return 0 if not missing else 6


if __name__ == "__main__":
    sys.exit(main())
