#!/usr/bin/env python3
"""Appendix data figures, drawn from receipts (never typed by hand); figstyle design system.

fig_factorial_subjects  (a)(b) the ten per-subject advantages (dynamic - static, rank percentile) in each
    of the four guard x split cells at k=100 and k=500, with the cell mean and its 95% bootstrap interval
    (scripts/mb_conditions.ci, seed 0, 20,000 draws -- the same call that produced the receipt and
    Appendix app:factorial); (c) the per-subject split main effect (= the architecture x split
    interaction) at both budgets with the receipt's interval. Source: results/interaction_receipt.json
    (per_subject_cell_advantage, contrasts.split_main.per_subject / ci95), written on the cluster by
    scripts/interaction_receipt.py.
fig_dose_ladder  the reference decoder's external dose ladder (S5): the eight one-bit runs (three
    unexposed C', five exposed D at dose 0.59) and the two single-run ladder arms (dose 0.30 and 1.0).
    No local machine receipt exists for these -- scripts/abc_decomposition.py reads the logs on the cluster --
    so the values are the log readouts pinned with their chain ids in scripts/check_paper_numbers.py
    (one-bit arm 4401-4408; Dmid 4502; Dfull 4501); every printed quantity (45.86, 48.26, +0.77, +3.00,
    monotone curve) is re-derived and asserted here.

    ~/.venvs/figtools/bin/python scripts/fig_results.py
      -> figures/fig_factorial_subjects.{svg,pdf,values.json}, figures/fig_dose_ladder.{svg,pdf,values.json}
"""
import json, sys, pathlib, io, contextlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "figstyle"))
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions   # noqa: E402
import figstyle as fs      # noqa: E402
import matplotlib; matplotlib.use("Agg")   # noqa: E702
import matplotlib.pyplot as plt            # noqa: E402
fs.use(); plt.rcParams.update({"axes.grid": False, "savefig.bbox": None, "savefig.pad_inches": 0})
FIG = ROOT / "figures"

def hu(x, q="0.001"):
    return Decimal(str(round(float(x), 6))).quantize(Decimal(q), rounding=ROUND_HALF_UP)
def f3(x):
    return f"{hu(x):+}".replace("-", "−")
def f2(x):
    return f"{hu(x, '0.01'):+}".replace("-", "−")

# ============================================================ fig_factorial_subjects
REC = ROOT / "results" / "interaction_receipt.json"
r = json.load(open(REC))
assert r["bootstrap"]["draws"] == mb_conditions.BOOT
SEED = r["bootstrap"]["seed"]
CELLS = [("rr", "rec.\nrec."), ("sr", "stim.\nrec."), ("rs", "rec.\nstory"), ("ss", "stim.\nstory")]
out = {"source": str(REC.relative_to(ROOT)), "generated_utc": r["generated_utc"], "bootstrap": r["bootstrap"], "budgets": {}}

fig = plt.figure(figsize=(fs.TEXTWIDTH_IN, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], left=0.105, right=0.975, bottom=0.26, top=0.85, wspace=0.42)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
YL = {"100": (-0.17, 0.26), "500": (-0.065, 0.135)}
for ax, k, lab in zip(axes[:2], ("100", "500"), ("(a)", "(b)")):
    b = r["budgets"][k]; subj = b["subjects"]
    M = np.array([[b["per_subject_cell_advantage"][s][c] for c, _ in CELLS] for s in subj])   # 10 x 4
    x = np.arange(4)
    ax.axvspan(-0.5, 1.5, color=fs.tint(fs.CORAL, 0.08), zorder=0); ax.axvspan(1.5, 3.5, color=fs.tint(fs.TEAL, 0.08), zorder=0)
    for row in M:
        ax.plot(x, row, "-", color="#C9CDD4", lw=0.6, zorder=1)
        ax.plot(x, row, "o", color=fs.GREY, ms=2.6, markeredgewidth=0, zorder=2)
    stats = []
    for j, (c, _) in enumerate(CELLS):
        m, lo, hi = mb_conditions.ci(M[:, j], seed=SEED); ex = bool(lo > 0 or hi < 0)
        col = fs.CORAL if c in ("rr", "sr") else fs.TEAL
        ax.vlines(j, lo, hi, color=col, lw=1.6, zorder=3)
        ax.plot(j, m, "s", color=col, ms=5.5, markeredgecolor=fs.SURF, markeredgewidth=0.8, zorder=4)
        span = YL[k][1] - YL[k][0]
        ax.text(j if j else -0.42, YL[k][1] - 0.02 * span - (0.10 * span if j % 2 else 0), f3(m) + ("*" if ex else ""),
                ha="center" if j else "left", va="top", fontsize=fs.SIZE_TICK, color=fs.INK)
        stats.append({"cell": c, "mean": float(hu(m, "0.0001")), "ci95": [float(hu(lo, "0.0001")), float(hu(hi, "0.0001"))], "excludes_zero": ex})
    out["budgets"][k] = {"cells": stats, "n": len(subj)}
    fs.zero_line(ax)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in CELLS]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(*YL[k])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("guard / split", labelpad=2)
    ax.set_title(f"k = {k}", loc="left")
    fs.panel_label(ax, lab, x=-0.08, y=1.04)
axes[0].set_ylabel("dynamic − static, rank percentile")
axes[0].text(0.5, YL["100"][0] + 0.008, "story shared", ha="center", va="bottom", fontsize=fs.SIZE_TICK, color=fs.CORAL, style="italic")
axes[0].text(2.5, YL["100"][0] + 0.008, "story withheld", ha="center", va="bottom", fontsize=fs.SIZE_TICK, color=fs.TEAL, style="italic")

ax = axes[2]
rows = {"100": 1.0, "500": 0.0}; XL = (-0.165, 0.135)
for k, y in rows.items():
    c = r["budgets"][k]["contrasts"]["split_main"]; per = np.array(c["per_subject"]); m, (lo, hi) = c["mean"], c["ci95"]
    ex = c["verdict"] == "excludes 0"; col = fs.CORAL if ex else fs.GREY
    ax.plot(per, np.full(len(per), y - 0.22), "o", color=fs.GREY, ms=2.6, markeredgewidth=0, zorder=2)
    ax.hlines(y, lo, hi, color=col, lw=1.6, zorder=3)
    ax.plot(m, y, "o", color=col, ms=5.5, markeredgecolor=fs.SURF, markeredgewidth=0.8, zorder=4)
    ax.text(XL[1], y + 0.30, f3(m) + ("*" if ex else ""), ha="right", va="bottom", fontsize=fs.SIZE_TICK, color=fs.INK)
    out["budgets"][k]["split_main"] = {"mean": float(hu(m, "0.0001")), "ci95": [float(hu(v, "0.0001")) for v in c["ci95"]], "verdict": c["verdict"], "per_subject": [float(hu(v, "0.0001")) for v in per]}
ax.axvline(0, color=fs.INK3, lw=0.8, zorder=1)
ax.set_yticks(list(rows.values())); ax.set_yticklabels([f"k = {k}" for k in rows]); ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, 1.85); ax.set_xlim(*XL); ax.set_xticks([-0.1, 0, 0.1]); ax.set_xticklabels(["−0.1", "0", "+0.1"])
ax.set_xlabel("split main effect, per subject", labelpad=2)
ax.set_title("architecture × split", loc="left")
fs.panel_label(ax, "(c)", x=-0.08, y=1.04)
fig.text(0.975, 0.015, "* 95% bootstrap interval excludes zero (20,000 draws, seed 0); dots: subjects", ha="right", va="bottom", fontsize=fs.SIZE_TICK, color=fs.INK2)
fs.save(fig, FIG / "fig_factorial_subjects.svg"); plt.close(fig)
json.dump(out, open(FIG / "fig_factorial_subjects.values.json", "w"), indent=1)

# ============================================================ fig_dose_ladder
ONEBIT = {"Cprime": {"4401": 46.60, "4403": 46.35, "4405": 44.64},
          "D": {"4402": 48.42, "4404": 49.23, "4406": 48.38, "4407": 47.66, "4408": 47.61}}
LADDER = {"Dmid_4502": (0.30, 46.63), "Dfull_4501": (1.0, 48.86)}
C = [Decimal(str(v)) for v in ONEBIT["Cprime"].values()]; D = [Decimal(str(v)) for v in ONEBIT["D"].values()]
c_mean, d_mean = sum(C) / len(C), sum(D) / len(D)
curve = [(0.0, c_mean), (0.30, Decimal("46.63")), (0.59, d_mean), (1.0, Decimal("48.86"))]
derived = {"C_mean": str(hu(c_mean, "0.01")), "D_mean": str(hu(d_mean, "0.01")),
           "mid_dose_exposure": f2(Decimal("46.63") - c_mean), "full_dose_exposure": f2(Decimal("48.86") - c_mean),
           "monotone": all(curve[i][1] < curve[i + 1][1] for i in range(3))}
PINNED = {"C_mean": "45.86", "D_mean": "48.26", "mid_dose_exposure": "+0.77", "full_dose_exposure": "+3.00", "monotone": True}
assert derived == PINNED, (derived, PINNED)

fig = plt.figure(figsize=(3.4, 2.25))
ax = fig.add_axes([0.175, 0.23, 0.79, 0.72])
ax.plot([x for x, _ in curve], [float(y) for _, y in curve], "-", color=fs.INK2, lw=1.1, zorder=2)
ax.plot([0.0, 0.59], [float(c_mean), float(d_mean)], "s", color=fs.INK, ms=5.5, markeredgecolor=fs.SURF, markeredgewidth=0.8, zorder=4, label="arm mean")
ax.plot([0.0] * len(C), [float(v) for v in C], "o", color=fs.GREY, ms=4.5, markeredgecolor=fs.SURF, markeredgewidth=0.8, zorder=3, label="unexposed runs (n = 3)")
ax.plot([0.59] * len(D), [float(v) for v in D], "o", color=fs.CORAL, ms=4.5, markeredgecolor=fs.SURF, markeredgewidth=0.8, zorder=3, label="exposed runs (n = 5)")
ax.plot([x for x, _ in LADDER.values()], [y for _, y in LADDER.values()], "D", color=fs.CORAL, ms=4.5, markeredgecolor=fs.SURF, markeredgewidth=0.8, zorder=3, label="single runs")
ax.annotate("", (1.0, 48.86), (1.0, float(c_mean)), arrowprops=dict(arrowstyle="<->", color=fs.INK2, lw=0.8, shrinkA=0, shrinkB=0), zorder=1)
ax.text(1.03, (48.86 + float(c_mean)) / 2, f"{derived['full_dose_exposure']}", ha="left", va="center", fontsize=fs.SIZE_TICK, color=fs.INK)
ax.text(0.33, 46.28, f"{derived['mid_dose_exposure']}", ha="left", va="top", fontsize=fs.SIZE_TICK, color=fs.INK)
ax.text(0.62, 47.35, f"{derived['D_mean']}", ha="left", va="top", fontsize=fs.SIZE_TICK, color=fs.INK)
ax.text(0.03, 45.55, f"{derived['C_mean']}", ha="left", va="top", fontsize=fs.SIZE_TICK, color=fs.INK)
ax.set_xlim(-0.06, 1.16); ax.set_ylim(44.2, 49.8)
ax.set_xticks([0, 0.30, 0.59, 1.0]); ax.set_xticklabels(["0", "0.30", "0.59", "1.0"])
ax.set_xlabel("fraction of the evaluation story's training blocks retained", labelpad=2)
ax.set_ylabel("top-10 (%), pool 7,140")
ax.legend(loc="upper left", handletextpad=0.4, labelspacing=0.25)
fs.save(fig, FIG / "fig_dose_ladder.svg"); plt.close(fig)
json.dump({"source": "scripts/check_paper_numbers.py pins (one-bit arm 4401-4408; Dmid 4502; Dfull 4501) <- cluster logs via scripts/abc_decomposition.py",
           "runs": ONEBIT, "ladder": LADDER, "curve": [(x, str(hu(y, '0.01'))) for x, y in curve], "derived": derived},
          open(FIG / "fig_dose_ladder.values.json", "w"), indent=1)
print("factorial k=500:", out["budgets"]["500"]["split_main"]["mean"], out["budgets"]["500"]["split_main"]["ci95"])
print("ladder:", derived)
