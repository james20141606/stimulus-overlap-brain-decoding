#!/usr/bin/env python3
"""Collect the segment-window (w) sensitivity sweep into one table.

For each w and each held-out story (task 0..3), read the brain and word-count-alone result JSONs.
Two metrics per arm:
  * mean_percentile = mean(1 - (rank-1)/(pool-1))  -- POOL-NORMALIZED (chance 0.5). This is the
    metric to compare ACROSS w, because sweeping w changes the candidate-pool size and this paper's
    whole thesis is that xchance / raw accuracy are NOT pool-invariant. Comparing xchance across w
    would commit the very error the paper warns against.
  * mean_top10x (xchance = top10*N/10, chance 1.0) -- reported too, but it is pool-DEPENDENT, so it
    is only compared BETWEEN the two arms AT THE SAME w (they share the pool there), never across w.

Per w: average over the 4 stories, and count how many stories word-count beats the brain on the
pool-normalized percentile (a fair same-pool paired comparison at each w). w=3.0 reuses the main-run
filenames. Run on the cluster:
  $SCRATCH/envs/analysis-env/bin/python scripts/collect_wsweep.py
"""
import json
from pathlib import Path

OUT = Path("$SCRATCH/project/results/m1")
WSET = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
TASKS = [0, 1, 2, 3]


def wt(w):  # 1.5 -> "15"; 3.0 -> "" (main run has no suffix)
    return "" if w == 3.0 else "_w" + str(w).replace(".", "")


def load(tag):
    p = OUT / f"result_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    s = d["summary"]
    return s.get("mean_meanpct"), s.get("mean_top10x"), d.get("n_train_seg")


# pct = pool-normalized percentile (chance 0.5, cross-w comparable); x = xchance (pool-dependent)
print("pct = pool-normalized mean percentile (chance 0.5, comparable across w)")
print("x   = xchance = top10*N/10 (chance 1.0, pool-dependent: compare arms only at same w)")
print(f"{'w':>4} {'story':>5} | {'brain_pct':>9} {'wc_pct':>7} | {'brain_x':>8} {'wc_x':>7} | {'wc>brain':>8} {'n_seg':>7}")
print("-" * 80)
rows = []
for w in WSET:
    sfx = wt(w)
    per_w = []
    for k in TASKS:
        b = load(f"story_holdout_task{k}{sfx}")
        n = load(f"nuis_keepR_n_words_task{k}{sfx}")
        bpct = f"{b[0]:.3f}" if b else "   --"
        npct = f"{n[0]:.3f}" if n else "   --"
        bx = f"{b[1]:.2f}" if b else "   --"
        nx = f"{n[1]:.2f}" if n else "   --"
        nseg = (n[2] if n else (b[2] if b else 0)) or 0
        win = (b and n and n[0] > b[0])         # same-pool paired comparison at this w
        winstr = str(win) if (b and n) else " -"
        print(f"{w:>4} {k:>5} | {bpct:>9} {npct:>7} | {bx:>8} {nx:>7} | {winstr:>8} {nseg:>7}")
        if b and n:
            per_w.append((b[0], n[0], b[1], n[1], win))
        elif n:                                  # word-count-only phase: still track its trend
            per_w.append((None, n[0], None, n[1], None))
    if per_w:
        def avg(i):
            v = [r[i] for r in per_w if r[i] is not None]
            return sum(v) / len(v) if v else None
        nn = len(per_w)
        mbp, mnp, mbx, mnx = avg(0), avg(1), avg(2), avg(3)
        paired = [r for r in per_w if r[4] is not None]
        wins = sum(r[4] for r in paired)
        rows.append((w, nn, mbp, mnp, mbx, mnx, wins, len(paired)))
        f3 = lambda x: f"{x:.3f}" if x is not None else "   --"
        f2 = lambda x: f"{x:.2f}" if x is not None else "   --"
        wtxt = f"{wins}/{len(paired)} wc wins (pct)" if paired else "wc-only"
        print(f"{w:>4} {'MEAN':>5} | {f3(mbp):>9} {f3(mnp):>7} | {f2(mbx):>8} {f2(mnx):>7} | {wtxt}")
        print("-" * 80)

print("\n=== SUMMARY (mean over stories present); pool-normalized pct is the cross-w metric ===")
print(f"{'w':>4} {'n':>3} {'brain_pct':>9} {'wc_pct':>7} {'brain_x':>8} {'wc_x':>7} {'wc>brain':>10}")
f3 = lambda x: f"{x:.3f}" if x is not None else "   --"
f2 = lambda x: f"{x:.2f}" if x is not None else "   --"
for w, nn, mbp, mnp, mbx, mnx, wins, npaired in rows:
    wtxt = f"{wins}/{npaired}" if npaired else "  --"
    print(f"{w:>4} {nn:>3} {f3(mbp):>9} {f3(mnp):>7} {f2(mbx):>8} {f2(mnx):>7} {wtxt:>10}")
