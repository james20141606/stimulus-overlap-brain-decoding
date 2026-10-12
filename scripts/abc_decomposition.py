#!/usr/bin/env python3
"""Regenerate the external A/B/C decomposition with per-arm seed spreads.

WHY THIS EXISTS

The paper's S5 decomposition (A-B = 6.45, B-C = 4.27 of the 10.72-point drop) was computed from
single runs and quoted with "the ratio is resolved only coarsely, to roughly +-15 percentage
points". Round-5 review (8.8, strong accept) named that single-run status as the last computable
gap. The seed replication (repro_b_seeds.sbatch / repro_c_seeds.sbatch, launched 2026-08-17) brings
every arm to n=3. Per the numbers-without-generators discipline, this generator exists BEFORE the
data lands: when the sentinels appear, the readout is this one command, not a hand computation.

WHAT IT PRINTS

Per arm: the three best-test top-10 values (100 - wer), mean, range. Then the contrasts A-B, B-C,
A-C as differences of means, each with a CONSERVATIVE bound [min_x - max_y, max_x - min_y] -- the
widest difference any pair of observed seeds supports. That is a range statement, not a confidence
interval, and the paper must quote it as such (n=3 per arm; same rule as the one-bit contrast).
Shares (A-B)/(A-C) and (B-C)/(A-C) get the same propagation. Whether the result narrows or WIDENS
the current "+-15pp" sentence, the paper follows the data.

GUARDS

- Pool: every log must show "wer 7140 negatives selected"; any other realized pool is a hard fail
  (the pool bug this paper documents -- never compare across pools).
- Completeness: a run without its .DONE sentinel is [PENDING] and excluded unless
  --include-pending is given (dora logs a Test Summary every improving epoch, so a live run has a
  perfectly plausible-looking best that would silently understate the arm).
- The arm-A original predates the .DONE convention; its slurm job 14260550 is COMPLETED (sacct)
  and its log is a closed .out file -- treated as done.

Usage (on the cluster):
  python3 scripts/abc_decomposition.py
  python3 scripts/abc_decomposition.py --include-pending   # peek while chains run
"""
import argparse
import os
import re
import sys

M1 = "$SCRATCH/project/logs/m1"
ARMS = {
    # arm A: original block split, full train (the reproduced Defossez reference)
    "A": [
        ("$SCRATCH/repro/bci-repro-r1-14260550.out", True),  # predates .DONE; sacct COMPLETED
        (f"{M1}/seeds_chain/brain_3101.log", None),
        (f"{M1}/seeds_chain/brain_3102.log", None),
    ],
    # arm B: block split, train thinned to arm C's volume (BM_TRAIN_FRAC=0.5824)
    "B": [
        (f"{M1}/volmatch/brain_4301.log", None),
        (f"{M1}/volmatch/brain_4302.log", None),
        (f"{M1}/volmatch/brain_4303.log", None),
    ],
    # arm C: story-level split, full train (BM_STORY_SPLIT=1, split seed 9)
    "C": [
        (f"{M1}/storysplit/brain_4101.log", None),
        (f"{M1}/storysplit/brain_4102.log", None),
        (f"{M1}/storysplit/brain_4103.log", None),
    ],
}
POOL = 7140
POOL_STRUCTURAL = {"brain_3101.log"}   # see the documented exception in main()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_run(path, done_override):
    """Return (top10, done, pool) for one log, or None if the log does not exist yet."""
    if not os.path.exists(path):
        return None
    txt = ANSI.sub("", open(path, errors="replace").read())
    wers = re.findall(r"Test Summary[^\n]*?wer=([0-9.]+)", txt)
    if not wers:
        return None
    pools = set(int(p) for p in re.findall(r"wer (\d+) negatives selected", txt))
    done = done_override if done_override is not None else os.path.exists(
        re.sub(r"\.log$", ".DONE", path))
    return 100.0 - float(wers[-1]), done, pools


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-pending", action="store_true",
                    help="use live runs' current best (peeking only; never for the paper)")
    a = ap.parse_args()

    vals, pending = {}, []
    print(f"{'arm':<4} {'run':<52} {'top-10':>7}  status")
    print("-" * 78)
    for arm, runs in ARMS.items():
        vals[arm] = []
        for path, done_override in runs:
            r = read_run(path, done_override)
            name = os.path.basename(path)
            if r is None:
                print(f"{arm:<4} {name:<52} {'--':>7}  NOT STARTED")
                continue
            top10, done, pools = r
            if not pools and name in POOL_STRUCTURAL:
                # Documented exception, not a loophole: this log carries no pool line anywhere
                # (its .log, the chunk .out and .err files were all grepped, 2026-08-20), but the
                # arm's pool is structural -- the block-hash split is seed-independent, so its
                # natural retention is exactly 7,140, verified in the logs of the OTHER two runs
                # of the identical split (R1 14260550 and brain_3102).
                pools = {POOL}
            if not pools:
                sys.exit(f"NO POOL LINE in {name}: the log has Test Summaries but no 'negatives "
                         f"selected' line (pool line may sit in an earlier chunk of a resumed "
                         f"run); verify the pool before trusting this run")
            bad = pools - {POOL}
            if bad:
                sys.exit(f"POOL MISMATCH in {name}: saw {sorted(pools)}, need {POOL}. "
                         f"Different pools are not comparable; fix before reading any number.")
            tag = "done" if done else "PENDING (live best; excluded)"
            print(f"{arm:<4} {name:<52} {top10:>7.2f}  {tag}")
            if done or a.include_pending:
                vals[arm].append(top10)
                if not done:
                    pending.append(f"{arm}:{name}")
        print()

    for arm in ARMS:
        v = vals[arm]
        if len(v) < 2:
            print(f"arm {arm}: only {len(v)} usable run(s) -- contrasts below need >=2 per arm "
                  f"to say anything about spread; run again when the chains finish.")
    print()
    print(f"{'arm':<4} {'n':>2} {'mean':>7} {'min':>7} {'max':>7} {'range':>6}")
    for arm in ARMS:
        v = vals[arm]
        if v:
            print(f"{arm:<4} {len(v):>2} {sum(v)/len(v):>7.2f} {min(v):>7.2f} {max(v):>7.2f} "
                  f"{max(v)-min(v):>6.2f}")

    def contrast(x, y, label):
        vx, vy = vals[x], vals[y]
        if not vx or not vy:
            return None
        d = sum(vx)/len(vx) - sum(vy)/len(vy)
        lo, hi = min(vx) - max(vy), max(vx) - min(vy)
        print(f"{label}: {d:+.2f}  conservative bound [{lo:+.2f}, {hi:+.2f}]"
              f"  (widest difference any observed seed pair supports; a range, not a CI)")
        return d, lo, hi

    print()
    ab = contrast("A", "B", "A-B (volume term)   ")
    bc = contrast("B", "C", "B-C (split residual) ")
    ac = contrast("A", "C", "A-C (total drop)     ")
    if ab and bc and ac and ac[1] > 0:
        print()
        # Shares: point from means; bound from the most adverse numerator/denominator pairing.
        for (num, name) in ((ab, "(A-B)/(A-C) volume share"), ((bc), "(B-C)/(A-C) split share")):
            pt = num[0] / ac[0] * 100
            lo = num[1] / ac[2] * 100
            hi = num[2] / ac[1] * 100
            print(f"{name}: {pt:.0f}%  bound [{lo:.0f}%, {hi:.0f}%]")
        print("\nQuote the bound, not the point, when the paper says how coarsely the ratio is "
              "resolved. If the bound is wider than the current '+-15pp' sentence, the sentence "
              "gets wider.")
    if pending:
        print(f"\nWARNING: included PENDING runs ({', '.join(pending)}) -- peeking only, "
              f"never for the paper.")

    # --- Pinned-pool rescore readout (design 1) + realization arms (design 2) + ablation
    # (design 5). Written BEFORE the sentinels land, per the generators-before-data rule. ---
    MIXED = "$SCRATCH/project/logs/m1/mixed"
    from decimal import Decimal
    ob, obd = {}, {}
    for arm, seeds in (("Cprime", (4401, 4403, 4405)), ("D", (4402, 4404, 4406))):
        avals, dvals = [], []
        for s in seeds:
            path = f"{MIXED}/{arm}_{s}.log"
            r = read_run(path, None)
            if r and r[1]:
                avals.append(r[0])
                # exact-decimal twin of the same value: several wers sit on the .xx5 boundary,
                # where float arithmetic rounds a difference down (2.885 -> "2.88").
                txt = ANSI.sub("", open(path, errors="replace").read())
                w = re.findall(r"Test Summary[^\n]*?wer=([0-9.]+)", txt)[-1]
                dvals.append(Decimal("100") - Decimal(w))
        ob[arm], obd[arm] = avals, dvals

    def rescore_value(arm, seed):
        """Prefer the pinned-pool rescore log; fall back to the training log. Returns
        (top10, sha1|None, is_rescore)."""
        rp = f"{MIXED}/{arm}_{seed}_rescore.log"
        tp_ = f"{MIXED}/{arm}_{seed}.log"
        for path, tag in ((rp, True), (tp_, False)):
            if not os.path.exists(path):
                continue
            done = os.path.exists(path.replace(".log", ".DONE"))
            if not done and tag:
                continue                      # incomplete rescore: fall through to training log
            txt = ANSI.sub("", open(path, errors="replace").read())
            wers = re.findall(r"Test Summary[^\n]*?wer=([0-9.]+)", txt)
            if not wers:
                continue
            shas = re.findall(r"pool identity sha1=([0-9a-f]+)", txt)
            return 100.0 - float(wers[-1]), (shas[-1] if shas else None), tag
        return None, None, False

    if os.path.exists("$SCRATCH/repro/onebit_rescore_ALL_DONE"):
        print("\n=== PINNED-POOL RESCORE (design 1) ===")
        shas, rows = set(), []
        for arm, seeds in (("Cprime", (4401, 4403, 4405)), ("D", (4402, 4404, 4406))):
            for s in seeds:
                v, h, is_r = rescore_value(arm, s)
                rows.append((arm, s, v, h)); shas.add(h)
                print(f"  {arm}_{s}: {v:.2f}  sha1={h}  ({'rescore' if is_r else 'TRAINING LOG'})")
        if len(shas) == 1 and None not in shas:
            print(f"  POOL IDENTITY VERIFIED: one draw ({shas.pop()}) across all six arms.")
            print("  -> the size-vs-identity qualification (ii) can be deleted; re-quote D-C' on")
            print("     these values via the one-bit block below run on _rescore logs.")
        else:
            print(f"  POOL IDENTITY FAILED: {len(shas)} distinct hashes {shas} -- do NOT delete")
            print("  qualification (ii); investigate before touching the paper.")

    # Realization arms (design 2): D at thinning realizations 1 and 2.
    real_vals = []
    for s in (4407, 4408):
        r = read_run(f"{MIXED}/D_{s}.log", None)
        if r and r[1]:
            real_vals.append((s, r[0]))
    if real_vals:
        print("\n=== THINNING REALIZATIONS (design 2) ===")
        for s, v in real_vals:
            print(f"  D_{s} (realization {s-4406}): {v:.2f}")
        if len(real_vals) == 2 and len(ob.get("Cprime", [])) == 3 and len(ob.get("D", [])) == 3:
            from math import comb as _comb
            d_all = ob["D"] + [v for _, v in real_vals]
            c_all = ob["Cprime"]
            if min(d_all) > max(c_all):
                print(f"  COMPLETE SEPARATION at 5 D vs 3 C': exact one-sided permutation "
                      f"p = 1/{_comb(8, 5)} = {1/_comb(8, 5):.3f} (only if exchangeable; see App. H)")
                print("  -> the one-realization clause can be deleted and the permutation floor tightened.")
            else:
                print("  separation BROKEN by a realization arm: demote to majority ordering; the")
                print("  thinning lottery enters Limits with its measured spread.")

    # Dose ladder (design 3): four-point external dose-response at matched volume.
    ladder = {}
    for name, seed, dose in (("Dmid", 4502, 0.30), ("Dfull", 4501, 1.0)):
        r = read_run(f"{MIXED}/{name}_{seed}.log", None)
        if r and r[1]:
            if not r[2] or r[2] - {POOL}:
                sys.exit(f"POOL problem in {name}_{seed}: {sorted(r[2])}")
            ladder[dose] = r[0]
    if ladder and len(ob.get("Cprime", [])) == 3 and len(ob.get("D", [])) == 3:
        print("\n=== EXTERNAL DOSE LADDER (design 3; single runs at 0.30 and 1.0) ===")
        c_mean = sum(ob["Cprime"]) / 3
        # the 0.59 point uses ALL five D runs (three model seeds + two thinning realizations)
        _d5 = list(ob["D"])
        for _s in (4407, 4408):
            _r = read_run(f"{MIXED}/D_{_s}.log", None)
            if _r and _r[1]:
                _d5.append(_r[0])
        curve = {0.0: c_mean, 0.59: sum(_d5)/len(_d5)}
        curve.update(ladder)
        pts = sorted(curve.items())
        print("  dose -> top-10: " + "   ".join(f"{d:.2f}->{v:.2f}" for d, v in pts))
        mono = all(pts[i][1] < pts[i+1][1] for i in range(len(pts)-1))
        print(f"  monotone: {'YES' if mono else 'NO'}")
        if 1.0 in ladder:
            leak = ladder[1.0] - c_mean
            print(f"  FULL-DOSE LEAK = {ladder[1.0]:.2f} - {c_mean:.2f} = {leak:+.2f}")
            print(f"  price/leak = 6.45/{leak:.2f} = {6.45/leak:.2f}x  "
                  f"({'< 6.45: S6 upgrades to measurement' if leak < 6.45 else 'DEMOTE: >= price'})")
            print(f"  escalation needed: {'YES' if abs(leak - 6.45) < 1.2 else 'no'} "
                  f"(|{leak:.2f} - 6.45| vs 1.2)")
        if 0.30 in ladder:
            print(f"  mid-dose exposure = {ladder[0.30]:.2f} - {c_mean:.2f} = "
                  f"{ladder[0.30]-c_mean:+.2f}")

    # Ablation cells (design 5): sign of (with - without) within split.
    ABLAT = "$SCRATCH/project/logs/m1/ablat"
    ab_cells = {}
    for name in ("noSL_block_4701", "noSL_story_4702", "noSL_volb_4703"):
        r = read_run(f"{ABLAT}/{name}.log", None)
        if r and r[1]:
            ab_cells[name] = r[0]
    if ab_cells:
        print("\n=== PUBLISHED-INGREDIENT ABLATION (design 5; criterion: sign of with-without) ===")
        ref = {"noSL_block_4701": ("A", vals.get("A", [])), "noSL_story_4702": ("C", vals.get("C", [])),
               "noSL_volb_4703": ("B", vals.get("B", []))}
        for name, v in ab_cells.items():
            arm, wv = ref[name]
            if wv:
                gap = sum(wv)/len(wv) - v
                print(f"  {name}: {v:.2f}  vs arm {arm} mean {sum(wv)/len(wv):.2f}  "
                      f"(with - without) = {gap:+.2f}")
        if "noSL_block_4701" in ab_cells and "noSL_story_4702" in ab_cells:
            gb = sum(vals["A"])/3 - ab_cells["noSL_block_4701"]
            gs = sum(vals["C"])/3 - ab_cells["noSL_story_4702"]
            print(f"  block-split advantage {gb:+.2f} vs story-split advantage {gs:+.2f}: "
                  f"{'FLIP or shrink-to-noise -> selection claim generalizes' if gs < 1.2 else 'ingredient advantage SURVIVES the split -> the paper prints that'}")
            print("  (escalate to 3 seeds per cell only if a within-split gap < ~3.6)")

    # --- One-bit exposure contrast (D vs C'), previously read out by hand (audit 2026-08-19). ---
    # D carries ~59% of the evaluation narrative's train blocks (uniform BM_TRAIN_FRAC=0.5657
    # thinning hits the added narrative too: 119,502/202,360 = 0.591), so this prices a 0.59-dose
    # exposure. Arms are independent runs; the chain pairing is scheduling, not statistics, so the
    # interval below is Welch on the two arm means, not a paired t.
    # (ob/obd built above, before the design readout sections)
    if len(ob["Cprime"]) >= 2 and len(ob["D"]) >= 2:
        import statistics as st
        from decimal import Decimal, ROUND_HALF_UP
        from itertools import combinations
        from math import comb
        # Half-up 2dp via Decimal: several wer values sit exactly on the .xx5 boundary (e.g.
        # 49.230-46.345 = 2.885), where float repr rounds DOWN and silently contradicts the paper.
        r2 = lambda v: Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP)
        c, d = ob["Cprime"], ob["D"]
        dc, dd = obd["Cprime"], obd["D"]
        # Display quantities computed in exact Decimal (the audit found the float path re-broke
        # the .xx5 boundary for the mean/cross lines); the Welch/permutation machinery stays float
        # (statistics on float64 is fine -- only the 2dp DISPLAY was wrong).
        diff = sum(d)/len(d) - sum(c)/len(c)
        diff_dec = (sum(dd)/len(dd)) - (sum(dc)/len(dc))
        cross = sorted(x - y for x in d for y in c)
        cross_dec = sorted(x - y for x in dd for y in dc)
        se = (st.variance(c)/len(c) + st.variance(d)/len(d)) ** 0.5
        num = (st.variance(c)/len(c) + st.variance(d)/len(d)) ** 2
        den = (st.variance(c)/len(c))**2/(len(c)-1) + (st.variance(d)/len(d))**2/(len(d)-1)
        df = num/den
        tmul = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}[max(1, min(5, int(df)))]
        # Exact permutation: count assignments of the 6 values to arms with mean-diff >= observed.
        pool_vals = c + d
        n_d = len(d)
        ge = sum(1 for idx in combinations(range(len(pool_vals)), n_d)
                 if (sum(pool_vals[i] for i in idx)/n_d
                     - sum(v for j, v in enumerate(pool_vals) if j not in idx)/(len(pool_vals)-n_d))
                 >= diff - 1e-9)
        p_perm = ge / comb(len(pool_vals), n_d)
        print(f"\nOne-bit exposure contrast (0.59 dose; independent arms):")
        print(f"  Cprime: {' '.join(str(r2(v)) for v in c)}   D: {' '.join(str(r2(v)) for v in d)}")
        sep = "COMPLETE (every D above every Cprime)" if cross[0] > 0 else "incomplete"
        print(f"  mean D-Cprime {r2(diff_dec):+}  cross-arm span [{r2(cross_dec[0]):+}, "
              f"{r2(cross_dec[-1]):+}]  separation: {sep}")
        print(f"  exact one-sided permutation p = {p_perm:.3f} (finest this n resolves: "
              f"1/{comb(len(pool_vals), n_d)})")
        print(f"  Welch t (conservative integer df={int(df)}): "
              f"[{r2(diff-tmul*se):+}, {r2(diff+tmul*se):+}] -- knife-edge at this n; the paper "
              f"quotes separation and range, not a t interval.")
        if len(c) == 3 and len(d) == 3:
            dc, dd = obd["Cprime"], obd["D"]
            print("  Chain-paired values (scheduling pairing, for the record): "
                  f"{r2(dd[0]-dc[0]):+} {r2(dd[1]-dc[1]):+} {r2(dd[2]-dc[2]):+}")

    # --- CLOCK VOLUME CONTROL (matched-volume clock arm, seed 4801, launched 2026-09-12). Written
    # BEFORE the sentinel lands (generators-before-data). §5 of both variants says "the matched-volume
    # clock arm is the control we have not run"; this section retires that clause either way.
    #   clock A (block split, full volume): 89.52 (R2 14377929), 89.57 (raster_3201), 92.00 (raster_3202)
    #   clock B (block split, arm B's exact training subset: BM_TRAIN_FRAC=0.5824, seed 0; BM_TIMING_ONLY=1)
    #   clock C (story split, full volume): 91.79 (storysplit/raster_4201)
    # PRE-REGISTERED READING: A's seed range is 89.52-92.00 (width 2.48); half of it is 1.24, so the
    # boundary is 88.28. Branch 1: clock B >= 88.28 -> the clock is insensitive to the volume cut; the
    # unrun-control clause becomes the measurement. Branch 2: clock B < 88.28 -> the clock DOES lose to
    # volume and the story split GAINS at matched volume (C - B > 0); the "does not drop" sentence is
    # re-scoped to the decomposition. Escalate to seeds 4802/4803 only if clock B is within 1.0 of 88.28.
    print("\n" + "=" * 78 + "\nCLOCK VOLUME CONTROL (matched-volume clock arm)\n" + "=" * 78)
    clockA = [89.52, 89.57, 92.00]
    clockC = 91.79
    boundary = min(clockA) - (max(clockA) - min(clockA)) / 2
    print(f"clock A (block, full volume)  {' '.join(f'{v:.2f}' for v in clockA)}   "
          f"range {min(clockA):.2f}-{max(clockA):.2f}  mean {sum(clockA)/3:.2f}")
    print(f"clock C (story, full volume)  {clockC:.2f}")
    rb = read_run(f"{M1}/volmatch/raster_4801.log", None)
    if rb is None:
        print("clock B (block, matched volume) raster_4801: NOT STARTED / no Test Summary yet")
    else:
        b, done_b, pools_b = rb
        if pools_b - {POOL}:
            sys.exit(f"POOL MISMATCH in raster_4801: saw {sorted(pools_b)}, need {POOL}")
        tag = "done" if done_b else "PENDING (live best; not for the paper)"
        print(f"clock B (block, matched volume) raster_4801: {b:.2f}  [{tag}]")
        # Orientation is B minus A(mean): a NEGATIVE number means the matched-volume clock sits
        # BELOW the full-volume mean. The paper says "0.06 below the three-seed mean". Printed the
        # other way round ("A(mean) - B +0.06") until 2026-09-16, which produced a sign error in
        # the text ("+0.06 against the mean" read as a gain).
        print(f"  B - A(mean) (volume term; negative = clock below the full-volume mean) "
              f"{b - sum(clockA)/3:+.2f}   C - B (split at matched volume) {clockC - b:+.2f}   "
              f"C - A(mean) (total) {clockC - sum(clockA)/3:+.2f}")
        if done_b:
            if b >= boundary:
                print(f"  BRANCH 1: clock B {b:.2f} >= {boundary:.2f} -> clock insensitive to the "
                      f"volume cut; replace the unrun-control clause with this measurement.")
            else:
                print(f"  BRANCH 2: clock B {b:.2f} < {boundary:.2f} -> the clock loses to volume; "
                      f"the story split gains {clockC - b:+.2f} at matched volume; re-scope the "
                      f"'does not drop' sentence to the decomposition.")
            if abs(b - boundary) <= 1.0:
                print(f"  ESCALATE: within 1.0 of the boundary -> run seeds 4802/4803 before writing.")


if __name__ == "__main__":
    main()
