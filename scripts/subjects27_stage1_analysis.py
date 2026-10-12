#!/usr/bin/env python
"""Stage 1 bridge analysis (PREREG notes/PREREG_subjects27_rebuttal_20261004.md §3 Q3, §4 rules). Written and
committed before any Stage 1 result exists. Reads result JSONs only; runs once.

Estimand: Delta_s = fm.meanpct(dynamic_big) - fm.meanpct(conv) (3-dp values in the result JSON), RR cell, ten-roster,
rebuilt archive x G. Seed-averaged per subject over seeds 0-2, then paired subject bootstrap, 20,000 draws,
default_rng(0), percentile (draws of mb_conditions.ci; asserted equal). Resolved = interval excludes 0.
Q3 primary (k=100): PASS = interval > 0 and all three seed means > 0; REVERSED = interval < 0; FAIL_SIGN = mean <= 0
and not REVERSED; FAIL_UNRESOLVED = otherwise.  GO to Stage 2 iff PASS (no-go rule, not waivable).
Bridge (label only): b_s = seed-avg new - seed-avg old (old = published mb1f seed 0 + st26_rr seeds 1, 2), CONSISTENT
(interval covers 0) / DRIFT (resolved); per seed for seeds 1-2 (realisation-matched). k=500 labelled; k=2000 descriptive.
Gates: G0 old three-seed k=100 reproduces +0.048 [+0.024, +0.071]; G1 all 30 units present, args echo, input_gain == G,
seg_cache set, per-k n and pool equal to the same subject's st26_rr_seed1 run. E2E: unit (01, seed 0) vs Stage 0b unit B
JSON identical except tag/timestamp/seg_cache (label; a mismatch holds the report until explained).
"""
import argparse, io, contextlib, json, sys
from pathlib import Path
import numpy as np

TEN = ["01", "02", "03", "05", "07", "08", "09", "10", "11", "13"]
BOOT = 20000
KS = (100, 500, 2000)


def ci(d, seed=0, q=(2.5, 97.5)):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(m, list(q))
    return float(d.mean()), float(lo), float(hi)


def lab(m, lo, hi):
    return "resolved_pos" if lo > 0 else "resolved_neg" if hi < 0 else "not_resolved"


def cell(j, arm, k):
    for p in j["results"][arm]["curve"]:
        if p["k"] == k:
            return p["fm"]
    return None


def delta(j, k):
    a, b = cell(j, "dynamic_big", k), cell(j, "conv", k)
    return None if a is None or b is None else a["meanpct"] - b["meanpct"]


def q3_label(per_seed_means, m, lo, hi):
    if lo > 0 and all(x > 0 for x in per_seed_means):
        return "PASS"
    if hi < 0:
        return "REVERSED"
    if m <= 0:
        return "FAIL_SIGN"
    return "FAIL_UNRESOLVED"


def analyse(new, old, k):
    """new/old: {seed: {subject: Delta}}. Returns the estimand block for budget k."""
    subs = sorted(set.intersection(*[set(new[s]) for s in new]))
    per_seed = {s: ci([new[s][x] for x in subs]) for s in sorted(new)}
    avg = np.array([np.mean([new[s][x] for s in new]) for x in subs])
    m, lo, hi = ci(avg)
    out = {"n": len(subs), "subjects": subs, "seed_means": {str(s): round(v[0], 4) for s, v in per_seed.items()},
           "seed_ci": {str(s): [round(v[1], 4), round(v[2], 4)] for s, v in per_seed.items()},
           "seed_avg": [m, lo, hi], "label": lab(m, lo, hi)}
    if old:
        osubs = [x for x in subs if all(x in old[s] for s in old)]
        oavg = np.array([np.mean([old[s][x] for s in old]) for x in osubs])
        navg = np.array([np.mean([new[s][x] for s in new]) for x in osubs])
        out["old_seed_avg"] = list(ci(oavg))
        b = ci(navg - oavg)
        out["bridge"] = {"n": len(osubs), "diff": list(b), "label": "DRIFT" if (b[1] > 0 or b[2] < 0) else "CONSISTENT"}
        out["bridge_per_seed"] = {}
        for s in sorted(set(new) & set(old)):
            if s == 0:
                continue                                  # seed 0 is not realisation-matched (mb1f RNG stream)
            d = ci([new[s][x] - old[s][x] for x in osubs])
            out["bridge_per_seed"][str(s)] = {"diff": list(d), "label": lab(*d)}
    return out


def load_new(runs, G, ref_rr1):
    new, probs, files = {0: {}, 1: {}, 2: {}}, [], {}
    for seed in (0, 1, 2):
        for s in TEN:
            f = Path(runs) / f"result_s27_st1_rr_seed{seed}_s{s}.json"
            if not f.exists():
                probs.append(f"missing {f.name}"); continue
            j = json.loads(f.read_text()); files[(seed, s)] = j
            if not (j.get("test_subject") == s and j.get("seed") == seed and j.get("split_seed") == 0
                    and j.get("guard_scope") == "recording" and j.get("split_by") == "recording"
                    and j.get("input_gain") == G and j.get("seg_cache")):
                probs.append(f"args echo {f.name}")
            r = ref_rr1.get(s)
            for k in KS:
                for arm in ("conv", "dynamic_big"):
                    a = cell(j, arm, k); b = cell(r, arm, k) if r else None
                    if (a is None) != (b is None) or (a and (a["n"] != b["n"] or a["pool"] != b["pool"])):
                        probs.append(f"n/pool {f.name} {arm} k={k}")
            for k in KS:
                d = delta(j, k)
                if d is not None:
                    new.setdefault(("k", k), {}).setdefault(seed, {})[s] = d
    return new, probs, files


def run(a):
    G = json.loads(Path(a.stage0a).read_text())["gain_G"]
    inp = Path(a.old_inputs); oruns = Path(a.old_runs)
    old = {}
    for k in KS:
        old[k] = {0: {}, 1: {}, 2: {}}
        for s in TEN:
            j0 = json.loads((inp / f"result_mb1f_ksweep_s{s}.json").read_text())
            d = delta(j0, k)
            if d is not None:
                old[k][0][s] = d
            for seed in (1, 2):
                j = json.loads((oruns / f"result_st26_rr_seed{seed}_s{s}.json").read_text())
                d = delta(j, k)
                if d is not None:
                    old[k][seed][s] = d
    # G0
    o100 = analyse(old[100], None, 100)
    g0 = [round(x, 3) for x in o100["seed_avg"]] == [0.048, 0.024, 0.071]
    ref1 = {s: json.loads((oruns / f"result_st26_rr_seed1_s{s}.json").read_text()) for s in TEN}
    newraw, probs, files = load_new(a.runs, G, ref1)
    rep = {"prereg": "notes/PREREG_subjects27_rebuttal_20261004.md", "G": G, "G0_old_three_seed_k100": o100["seed_avg"],
           "G0": "PASS" if g0 else "FAIL", "G1_problems": probs, "G1": "PASS" if not probs else "FAIL"}
    if not g0 or probs:
        rep["verdict"] = "NOT_ANALYSED (gate failure)" if g0 else "NOT_ANALYSED (G0)"
        n_done = sum(len(v) for v in newraw.get(("k", 100), {}).values())
        rep["completed_units"] = f"{n_done}/30"
        return rep
    blocks = {}
    for k in KS:
        nk = newraw[("k", k)]
        blocks[str(k)] = analyse(nk, old[k], k)
    q = blocks["100"]
    seeds = [q["seed_means"][s] for s in ("0", "1", "2")]
    rep["Q3_primary_k100"] = q3_label(seeds, *q["seed_avg"])
    rep["GO_stage2"] = rep["Q3_primary_k100"] == "PASS"
    rep["blocks"] = blocks
    # E2E
    e2e = None
    if a.stage0b_unitB and Path(a.stage0b_unitB).exists():
        b0 = json.loads(Path(a.stage0b_unitB).read_text()); n0 = files.get((0, "01"))
        strip = lambda d: {k: v for k, v in d.items() if k not in ("tag", "timestamp", "seg_cache")}
        e2e = json.dumps(strip(b0), sort_keys=True) == json.dumps(strip(n0), sort_keys=True)
    rep["E2E_unit01_seed0_equals_0bB"] = e2e
    rep["verdict"] = "GO" if rep["GO_stage2"] else "NOGO"
    return rep


def selftest():
    res = []
    old_argv = sys.argv; sys.argv = sys.argv[:1]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with contextlib.redirect_stdout(io.StringIO()):
        import mb_conditions as M
    sys.argv = old_argv
    d = np.random.default_rng(3).normal(0.03, 0.04, 10)
    res.append(("bootstrap == mb_conditions.ci", ci(d) == tuple(float(x) for x in M.ci(d))))
    rng = np.random.default_rng(1)

    def mk(mu, sd=0.01, seeds=(0.0, 0.0, 0.0)):
        return {s: {x: mu + seeds[s] + rng.normal(0, sd) for x in TEN} for s in (0, 1, 2)}
    cases = {"PASS": mk(0.05), "REVERSED": mk(-0.05), "FAIL_SIGN": mk(-0.002, 0.03),
             "FAIL_UNRESOLVED": mk(0.004, 0.04), "PASS_but_seed_neg": mk(0.05, 0.005, (0.0, 0.0, -0.07))}
    want = {"PASS": "PASS", "REVERSED": "REVERSED", "FAIL_SIGN": "FAIL_SIGN", "FAIL_UNRESOLVED": "FAIL_UNRESOLVED",
            "PASS_but_seed_neg": None}
    for nm, c in cases.items():
        r = analyse(c, None, 100); seeds = [r["seed_means"][s] for s in ("0", "1", "2")]
        got = q3_label(seeds, *r["seed_avg"])
        ok = (got == want[nm]) if want[nm] else (got != "PASS")
        res.append((f"branch {nm} -> {got}", ok))
    base = mk(0.05)
    shifted = {s: {x: v + 0.05 for x, v in base[s].items()} for s in base}
    res.append(("bridge CONSISTENT on identical", analyse(base, base, 100)["bridge"]["label"] == "CONSISTENT"))
    res.append(("bridge DRIFT on +0.05 shift", analyse(shifted, base, 100)["bridge"]["label"] == "DRIFT"))
    for nm, ok in res:
        print(("PASS  " if ok else "FAIL  ") + nm)
    ok = all(o for _, o in res); print("SELFTEST ALL PASS" if ok else "SELFTEST FAILURES"); return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs"); ap.add_argument("--old_inputs"); ap.add_argument("--old_runs")
    ap.add_argument("--stage0a"); ap.add_argument("--stage0b_unitB"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    rep = run(a)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "blocks"}, indent=1, default=float))


if __name__ == "__main__":
    main()
