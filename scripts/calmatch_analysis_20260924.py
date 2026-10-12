#!/usr/bin/env python3
"""Pre-registered analysis of the calibration-diversity control (notes/PREREG_calibmatch_20260924.md).

Reads per-subject result JSONs (never writes them) and emits one analysis receipt. Nothing here is
tuned on the new arm: the estimator, bootstrap, margin and verdict rule are the ones fixed in the
pre-registration, and Gate 0 refuses to report anything unless the reused files reproduce the
published Table 1 story-0 rows first.

  rho_pub_s   = (r_U123_s - 1/2) / (r_S_s - 1/2)       published retained fraction (mean of ratios)
  rho_match_s = (r_U12_s  - 1/2) / (r_S_s - 1/2)       calibration-diversity-matched
  D_s         = rho_match_s - rho_pub_s = (r_U12_s - r_U123_s) / (r_S_s - 1/2)
Mean over subjects; 95% paired subject bootstrap, 20,000 draws, default_rng(0), percentile -- the
retained_fraction.py convention, so the published CIs are reproduced exactly.

Verdict (primary cells: static conv, story 0, k=500 and k=2000; margin 10 percentage points):
  NO_MATERIAL_CHANGE  both CIs of mean D inside (-10, +10)
  MATERIAL_CHANGE     at either k, CI of mean D excludes 0 and |mean D| >= 10
  INCONCLUSIVE        otherwise
  INCOMPLETE          fewer than the pre-registered subjects in any primary cell (no verdict)
  GATE_FAIL           reproduction or design gate failed (no verdict)

Usage:
  python scripts/calmatch_analysis_20260924.py --results <dir with result_*.json> \
      --receipts <dir with preflight_/receipt_ json> --out results/calmatch_20260924/analysis.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

CHANCE = 0.5
BOOT = 20000
MARGIN = 10.0
BUDGETS = (100, 500, 2000)
ARMS = (("static conv", "conv"), ("dynamic", "dynamic_big"))
SKIP = ("smoke", "test", "tmp", "debug", "scratch")

# Primary: all ten Table-1 subjects. One subject has no k=2000 SEEN curve, so n=9 there.
PRIMARY_N = {100: 10, 500: 10, 2000: 9}

ROWS = {
    "story0": {"seen": "result_mb1k_volctl_", "u123": "result_mb1j_story0_",
               "u12": "result_mb1z_calmatch_story0_", "calib": "1,2", "eval": "0"},
    "story1": {"seen": "result_mb1s_story1_seen_", "u123": "result_mb1s_story1_unseen_",
               "u12": "result_mb1z_calmatch_story1_", "calib": "0,2", "eval": "1"},
}
# Reused-only secondary rows from MB-1n (JID 15159191, 2026-08-02). Their values have never been
# read in any session log; they are included if present and are descriptive.
MB1N = {
    "xstory_calmatched": ("result_mb1k_volctl_", "result_mb1n_unseen_"),   # vs 'cross-story' row
    "seen_replay":       ("result_mb1k_volctl_", "result_mb1n_seen_"),     # rerun of SEEN config
}

# Gate 0: Table 1 story-0 rows as printed (paper_iclr.tex, tab:headline) at HEAD 152a0de.
PUBLISHED = {
    ("conv", 100): (0.779, 0.689, 71.7, 60.2, 83.6),
    ("conv", 500): (0.861, 0.695, 54.1, 51.4, 56.6),
    ("conv", 2000): (0.884, 0.707, 53.9, 52.2, 55.8),
    ("dynamic_big", 100): (0.799, 0.677, 60.7, 53.8, 67.9),
    ("dynamic_big", 500): (0.855, 0.689, 53.5, 50.3, 56.6),
    ("dynamic_big", 2000): (0.878, 0.690, 50.3, 48.5, 52.4),
}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(results, prefix):
    """{subject: json}. Subject from the test_subject FIELD (result_io.py convention)."""
    kept, files = {}, {}
    for f in sorted(glob.glob(os.path.join(results, f"{prefix}*.json"))):
        b = os.path.basename(f)
        if any(k in b.lower() for k in SKIP):
            continue
        d = json.load(open(f))
        s = d.get("test_subject")
        if s is None:
            continue
        s = str(int(s)) if str(s).strip().isdigit() else str(s)
        if s in kept:
            raise RuntimeError(f"duplicate subject {s} for prefix {prefix}: {b}")
        kept[s] = d
        files[s] = f
    return kept, files


def subj_key(x):
    """Numeric subject order (01, 02, ..., 13) -- the order in which the published Table 1 intervals
    were bootstrapped. Lexicographic order of the int-normalised keys ('1','10','11','13','2',...)
    pairs the same bootstrap index draws with different subjects and moves four published CI bounds
    by 0.1 (Deviation 1 in the PREREG, 2026-09-24, found at Gate 0 before any new-arm run)."""
    return (0, int(x)) if str(x).isdigit() else (1, str(x))


def cell(j, arm, k, field="fm"):
    for p in j.get("results", {}).get(arm, {}).get("curve", []):
        if p["k"] == k and field in p and p[field].get("pool"):
            return p[field]
    return None


def boot_ci(fn, n, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(BOOT, n))
    bs = np.array([fn(i) for i in idx])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def triple(S, U, M, arm, k):
    subs, seen, u123, u12, eval_checks = [], [], [], [], []
    for s in sorted(set(S) & set(U) & set(M), key=subj_key):
        a, b, c = cell(S[s], arm, k), cell(U[s], arm, k), cell(M[s], arm, k)
        if a is None or b is None or c is None:
            continue
        subs.append(s)
        seen.append(a["meanpct"]); u123.append(b["meanpct"]); u12.append(c["meanpct"])
        eval_checks.append({"subject": s, "n": [a.get("n"), b.get("n"), c.get("n")],
                            "pool": [a.get("pool"), b.get("pool"), c.get("pool")]})
    return subs, np.array(seen, float), np.array(u123, float), np.array(u12, float), eval_checks


def summarise(seen, u123, u12):
    n = len(seen)
    den = seen - CHANCE
    # Ratios in [0,1] units, x100 AFTER the mean: the exact operation order of retained_fraction.py.
    rp = (u123 - CHANCE) / den
    rm = (u12 - CHANCE) / den
    D = (u12 - u123) / den
    out = {
        "n": n, "seen_mean": float(seen.mean()),
        "u123_mean": float(u123.mean()), "u12_mean": float(u12.mean()),
        "min_seen_denominator": round(float(den.min()), 4),
        # 2026-09-25 (PREREG Deviation 3): full precision; rounding here and again to 0.1 in the
        # renderer double-rounded two printed bounds. The renderer is the only place that rounds.
        "rho_pub": float(rp.mean() * 100.0), "rho_match": float(rm.mean() * 100.0),
        "D_pp": float(D.mean() * 100.0),
        "delta_r_u12_minus_u123": round(float((u12 - u123).mean()), 4),
    }
    out["rho_pub_ci"] = list(boot_ci(lambda i: rp[i].mean() * 100.0, n))
    out["rho_match_ci"] = list(boot_ci(lambda i: rm[i].mean() * 100.0, n))
    out["D_ci"] = list(boot_ci(lambda i: D[i].mean() * 100.0, n))
    out["delta_r_ci"] = [round(v, 4) for v in boot_ci(lambda i: (u12[i] - u123[i]).mean(), n)]
    return out


def verdict(primary):
    """primary: {k: summary} for static conv story 0 at k in (500, 2000)."""
    for k in (500, 2000):
        if k not in primary or primary[k]["n"] < PRIMARY_N[k]:
            return "INCOMPLETE"
    inside = all(-MARGIN < primary[k]["D_ci"][0] and primary[k]["D_ci"][1] < MARGIN
                 for k in (500, 2000))
    if inside:
        return "NO_MATERIAL_CHANGE"
    material = any((primary[k]["D_ci"][0] > 0 or primary[k]["D_ci"][1] < 0)
                   and abs(primary[k]["D_pp"]) >= MARGIN for k in (500, 2000))
    return "MATERIAL_CHANGE" if material else "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--receipts", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_gate0", action="store_true", help="synthetic tests only")
    ap.add_argument("--bp_replication", action="store_true",
                    help="clusterB route BP-B only (requires the PREREG amendment): story-0 SEEN and "
                         "UNSEEN are the clusterB reruns (result_bp_*), and Gate 0 becomes a "
                         "tolerance check against Table 1 (means +/-0.02, retained fraction +/-5 pp)")
    a = ap.parse_args()
    if a.bp_replication:
        ROWS["story0"]["seen"] = "result_bp_mb1k_volctl_"
        ROWS["story0"]["u123"] = "result_bp_mb1j_story0_"

    report = {"prereg": "notes/PREREG_calibmatch_20260924.md", "script_sha256": sha(__file__),
              "mode": "bp_replication" if a.bp_replication else "reuse_published",
              "bootstrap": {"draws": BOOT, "rng": "numpy default_rng(0)", "ci": "percentile 2.5/97.5",
                            "resample": "subjects, (SEEN,U123,U12) triples kept together"},
              "margin_pp": MARGIN, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "inputs": {}, "gates": {}, "rows": {}, "problems": []}
    P = report["problems"]

    loaded = {}
    for row, spec in ROWS.items():
        for role in ("seen", "u123", "u12"):
            d, files = load(a.results, spec[role])
            loaded[(row, role)] = d
            for s, f in files.items():
                report["inputs"][os.path.basename(f)] = sha(f)

    # ---- Gate 1: design fields of the new arm and identical evaluation sets
    for row, spec in ROWS.items():
        for s, d in loaded[(row, "u12")].items():
            want = {"holdout_story": spec["eval"], "calib_stories": spec["calib"],
                    "guard_scope": "stimulus"}
            for key, v in want.items():
                if str(d.get(key)) != v:
                    P.append(f"{row} s{s}: {key}={d.get(key)!r}, expected {v!r}")
            if str(d.get("eval_story") or "") not in ("", spec["eval"]):
                P.append(f"{row} s{s}: eval_story={d.get('eval_story')!r}")
    if a.receipts:
        for f in sorted(glob.glob(os.path.join(a.receipts, "preflight_mb1z_calmatch_*.json"))):
            pf = json.load(open(f))
            report["inputs"][os.path.basename(f)] = sha(f)
            if pf.get("status") != "PASS":
                P.append(f"{os.path.basename(f)}: preflight {pf.get('status')} {pf.get('problems')}")
        for f in sorted(glob.glob(os.path.join(a.receipts, "receipt_mb1z_calmatch_*.json"))):
            rc = json.load(open(f))
            report["inputs"][os.path.basename(f)] = sha(f)
            rf = os.path.join(a.results, rc.get("result_file", ""))
            if os.path.exists(rf) and sha(rf) != rc.get("result_sha256"):
                P.append(f"{os.path.basename(f)}: result SHA differs from receipt")

    # ---- per-row summaries
    for row in ROWS:
        S, U, M = (loaded[(row, r)] for r in ("seen", "u123", "u12"))
        report["rows"][row] = {}
        for label, arm in ARMS:
            for k in BUDGETS:
                subs, seen, u123, u12, ev = triple(S, U, M, arm, k)
                for e in ev:
                    if len(set(e["n"])) != 1 or len(set(e["pool"])) != 1:
                        P.append(f"{row} {arm} k={k} s{e['subject']}: eval n/pool differ {e}")
                if len(subs) < 3:
                    continue
                if np.min(seen - CHANCE) <= 0:
                    P.append(f"{row} {arm} k={k}: a SEEN subject at/below chance")
                    continue
                summ = summarise(seen, u123, u12)
                summ["subjects"] = subs
                report["rows"][row][f"{arm}|{k}"] = summ

    # ---- Gate 0: reproduce the published story-0 rows from the reused files (all 10 subjects,
    # independent of whether the new arm exists yet -- pairs SEEN with U123 only).
    g0 = {}
    S, U = loaded[("story0", "seen")], loaded[("story0", "u123")]
    for (arm, k), (ps, pu, pr, plo, phi) in PUBLISHED.items():
        seen, u123 = [], []
        for s in sorted(set(S) & set(U), key=subj_key):
            x, y = cell(S[s], arm, k), cell(U[s], arm, k)
            if x is not None and y is not None:
                seen.append(x["meanpct"]); u123.append(y["meanpct"])
        seen, u123 = np.array(seen), np.array(u123)
        if len(seen) < 3:
            g0[f"{arm}|{k}"] = {"ok": False, "why": "no data"}
            continue
        rp = (u123 - CHANCE) / (seen - CHANCE)
        lo, hi = boot_ci(lambda i: rp[i].mean() * 100.0, len(seen))
        got = (round(seen.mean(), 3), round(u123.mean(), 3), round(rp.mean() * 100.0, 1),
               round(lo, 1), round(hi, 1))
        if a.bp_replication:
            ok = (abs(got[0] - ps) <= 0.02 and abs(got[1] - pu) <= 0.02 and abs(got[2] - pr) <= 5.0)
        else:
            ok = got == (ps, pu, pr, plo, phi)
        g0[f"{arm}|{k}"] = {"ok": bool(ok), "n": int(len(seen)), "got": got,
                            "published": (ps, pu, pr, plo, phi)}
    report["gates"]["gate0_reproduction"] = g0
    gate0_ok = all(v.get("ok") for v in g0.values())
    if not gate0_ok and not a.skip_gate0:
        P.append("Gate 0: reused files do not reproduce the published Table 1 story-0 rows")

    # ---- MB-1n reuse (descriptive)
    report["mb1n"] = {}
    for name, (ps, pu) in MB1N.items():
        Sx, _ = load(a.results, ps)
        Ux, fx = load(a.results, pu)
        for s, f in fx.items():
            report["inputs"][os.path.basename(f)] = sha(f)
        for label, arm in ARMS:
            for k in BUDGETS:
                seen, un = [], []
                for s in sorted(set(Sx) & set(Ux), key=subj_key):
                    x, y = cell(Sx[s], arm, k), cell(Ux[s], arm, k)
                    if x is not None and y is not None:
                        seen.append(x["meanpct"]); un.append(y["meanpct"])
                if len(seen) >= 3:
                    seen, un = np.array(seen), np.array(un)
                    r = (un - CHANCE) / (seen - CHANCE)
                    report["mb1n"][f"{name}|{arm}|{k}"] = {
                        "n": len(seen), "rho": float(r.mean() * 100.0),
                        "ci": list(boot_ci(lambda i: r[i].mean() * 100.0, len(seen)))}

    primary = {k: report["rows"]["story0"][f"conv|{k}"]
               for k in (500, 2000) if f"conv|{k}" in report["rows"]["story0"]}
    if P:
        report["verdict"] = "GATE_FAIL"
    else:
        report["verdict"] = verdict(primary)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(f"verdict: {report['verdict']}")
    for p in P:
        print(f"  problem: {p}")
    for row, cells in report["rows"].items():
        for key, v in cells.items():
            print(f"  {row:7s} {key:18s} n={v['n']:2d} S={v['seen_mean']:.3f} U123={v['u123_mean']:.3f} "
                  f"U12={v['u12_mean']:.3f} rho_pub={v['rho_pub']:.1f} rho_match={v['rho_match']:.1f} "
                  f"{v['rho_match_ci']} D={v['D_pp']:+.1f} {v['D_ci']}")
    print(f"[out] {a.out}  sha256={sha(a.out)}")
    sys.exit(0 if report["verdict"] not in ("GATE_FAIL",) else 5)


if __name__ == "__main__":
    main()
