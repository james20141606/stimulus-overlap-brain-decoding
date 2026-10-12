#!/usr/bin/env python3
"""Body Figures 2 and 3 of paper_iclr.tex, redesigned 2026-09-24 (OVERHAUL_20260924.md, item 4).

Why: the user called the 09-23 Figure 2 "too messy" -- per-subject grey spaghetti across four
conditions, and a third panel mixing a CI with a scattered dot strip. The redesign draws the SAME
receipt values as paired-difference forest plots: one row per protocol cell or contrast, the
bootstrap mean and 95% interval as the mark, the ten subjects as light dots on the row (no
connecting lines). Figure 3 keeps its data and gains exposure deltas read from the receipt.

Receipts (read only, unchanged):
  results/interaction_receipt.json            per-subject cell advantages and paired contrasts
  figures/fig_factorial_subjects.values.json  cell means + 95% bootstrap intervals (same bootstrap)
  results/onebit_wer_readouts.json            eight one-bit runs (top-10 = 100 - wer, 2 dp)
  figures/fig_dose_ladder.values.json         dose 0.30 / 1.0 single runs, arm-mean curve, deltas
Writes figures/integrated_20260924/fig02_factorial.{svg,pdf,png}, fig03_dose.{svg,pdf,png}
through figstyle.compose.export (layout audit + headless-Chrome PDF + Type-3 check).
Run: ~/.venvs/figtools/bin/python scripts/fig_body_20260924.py
"""
import json, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / "scripts" / "figstyle"))
import figstyle as fs          # noqa: E402
from compose import export     # noqa: E402

OUT = B / "figures" / "integrated_20260924"
OUT.mkdir(parents=True, exist_ok=True)
fs.use()
plt.rcParams.update({"savefig.bbox": None, "axes.grid": False, "axes.edgecolor": fs.INK3})
MINUS = "−"
readj = lambda p: json.loads(Path(p).read_text())


def one_page(pdf):
    """Chrome's print can round a small canvas onto a second, blank page; keep page 1 only."""
    import subprocess
    n = int([l for l in subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout.splitlines()
             if l.startswith("Pages:")][0].split()[1])
    if n > 1:
        tmp = pdf.with_suffix(".p1.pdf")
        subprocess.run(["qpdf", str(pdf), "--pages", str(pdf), "1", "--", str(tmp)], check=True)
        tmp.replace(pdf)
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout
    assert "Type 3" not in out, out
    return pdf


def fmt(x):
    return f"{x:+.3f}".replace("-", MINUS)


def forest_row(ax, y, mean, lo, hi, dots, color, resolved, marker="o"):
    """One forest row: light subject dots on the row, then the interval and the mean on top."""
    jitter = np.linspace(-0.13, 0.13, len(dots))
    ax.scatter(dots, y + jitter, s=7, color=fs.GREY, alpha=0.45, lw=0, zorder=2)
    ax.hlines(y, lo, hi, color=color, lw=1.5, zorder=3)
    ax.plot([mean], [y], marker=marker, ms=5.2, mec=color, mew=1.2,
            mfc=color if resolved else "white", ls="", zorder=4)


def fig02():
    r = readj(B / "results/interaction_receipt.json")
    vals = readj(B / "figures/fig_factorial_subjects.values.json")
    cells = ["rr", "sr", "rs", "ss"]
    cell_lab = {"rr": "recording guard, recording split\n(standard protocol)",
                "sr": "stimulus guard, recording split",
                "rs": "recording guard, story split",
                "ss": "stimulus guard, story split"}
    cell_col = {"rr": fs.CORAL, "sr": fs.INK2, "rs": fs.INK2, "ss": fs.TEAL}

    # 2026-09-25 acceptance fix M9: canvas 2.35 -> H in (page budget); absolute margins kept, so the
    # fixed-size text (titles, tick labels, x label) keeps its inches and only the plot area shrinks.
    H = 1.9
    fy = lambda frac_of_235: frac_of_235 * 2.35 / H                  # old figure fraction -> inches -> new fraction
    y0, top = fy(0.20), 1 - (1 - 0.82) * 2.35 / H
    fig = plt.figure(figsize=(5.5, H))
    # (a)(b): the four protocol cells at each budget; (c): the paired factorial contrasts.
    axa = fig.add_axes([0.285, y0, 0.165, top - y0])
    axb = fig.add_axes([0.475, y0, 0.165, top - y0], sharey=axa)
    axc = fig.add_axes([0.815, y0, 0.17, top - y0])

    for ax, k, xl in ((axa, "100", (-0.13, 0.20)), (axb, "500", (-0.055, 0.10))):
        b = r["budgets"][k]
        st = {c["cell"]: c for c in vals["budgets"][k]["cells"]}
        for i, c in enumerate(cells):
            dots = [b["per_subject_cell_advantage"][s][c] for s in b["subjects"]]
            s = st[c]; lo, hi = s["ci95"]
            forest_row(ax, i, s["mean"], lo, hi, dots, cell_col[c], s["excludes_zero"])
        ax.axvline(0, color=fs.INK3, lw=0.8, zorder=1)
        ax.set_xlim(*xl); ax.set_ylim(3.55, -0.55)
        ax.set_title(f"k = {k}", loc="center", fontsize=fs.SIZE_LABEL, pad=4)
        ax.grid(axis="x", color=fs.GRID, lw=0.5); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axa.set_yticks(range(4)); axa.set_yticklabels([cell_lab[c] for c in cells], fontsize=7.2, color=fs.INK)
    plt.setp(axb.get_yticklabels(), visible=False)
    axa.set_xticks([-0.1, 0, 0.1]); axb.set_xticks([-0.05, 0, 0.05])
    fig.text(0.46, fy(0.045), "dynamic " + MINUS + " static rank percentile", ha="center",
             fontsize=fs.SIZE_TICK + 0.5, color=fs.INK2)
    fig.text(0.01, 1 - 0.06 * 2.35 / H, "(a) Advantage in each protocol cell", fontsize=fs.SIZE_LABEL,
             fontweight="bold", color=fs.INK, ha="left")

    # (c) paired contrasts: split main (= architecture x split interaction), guard main, three-way.
    rows = [("split_main", "500"), ("split_main", "100"), ("guard_main", "500"), ("guard_main", "100"),
            ("guard_x_split", "500"), ("guard_x_split", "100")]
    lab = {"split_main": "split main\n(arch." + "×" + "split)", "guard_main": "guard main",
           "guard_x_split": "three-way I"}   # 2026-09-24 final review: same name as Table 2
    col = {"split_main": fs.INK, "guard_main": fs.INK2, "guard_x_split": fs.INK3}
    ypos = [0, 0.8, 2.0, 2.8, 4.0, 4.8]
    for y, (name, k) in zip(ypos, rows):
        c = r["budgets"][k]["contrasts"][name]; lo, hi = c["ci95"]
        forest_row(axc, y, c["mean"], lo, hi, c["per_subject"], col[name], lo > 0 or hi < 0,
                   marker="s" if k == "500" else "o")
    axc.axvline(0, color=fs.INK3, lw=0.8, zorder=1)
    axc.set_xlim(-0.21, 0.23); axc.set_ylim(5.35, -1.45)   # headroom row for the legend (it sat on the k=500 split mark); axc.set_xticks([-0.2, 0, 0.2])
    axc.set_yticks([0.4, 2.4, 4.4]); axc.set_yticklabels([lab[n] for n in ("split_main", "guard_main", "guard_x_split")],
                                                        fontsize=7.2, color=fs.INK)
    axc.tick_params(axis="y", length=0); axc.spines["left"].set_visible(False)
    axc.grid(axis="x", color=fs.GRID, lw=0.5)
    axc.set_xlabel("paired contrast", fontsize=fs.SIZE_TICK + 0.5, labelpad=2)
    fig.text(0.68, 1 - 0.06 * 2.35 / H, "(b) Paired factorial contrasts", fontsize=fs.SIZE_LABEL, fontweight="bold",
             color=fs.INK, ha="left")
    axc.legend([Line2D([], [], marker="s", ls="", mfc="white", mec=fs.INK2, ms=4.5),
                Line2D([], [], marker="o", ls="", mfc="white", mec=fs.INK2, ms=4.5)],
               ["k = 500", "k = 100"], loc="upper center", ncol=2, fontsize=7, handletextpad=0.2,
               borderaxespad=0.1, labelspacing=0.2, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    svg = OUT / "fig02_factorial.svg"
    fig.savefig(svg); plt.close(fig)
    pdf, png = export(svg, width_in=5.5); one_page(pdf)
    print("wrote", pdf)


def fig03():
    r = readj(B / "results/onebit_wer_readouts.json"); old = readj(B / "figures/fig_dose_ladder.values.json")
    runs = {k: float((Decimal("100") - Decimal(v["wer"])).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))
            for k, v in r["runs"].items()}
    C = [v for k, v in runs.items() if k.startswith("Cprime")]
    D = [v for k, v in runs.items() if k.startswith("D_")]
    curve = [(float(x), float(v)) for x, v in old["curve"]]
    base = curve[0][1]
    # cross-check against the receipt's own derived values before drawing anything
    assert f"{np.mean(C):.2f}" == old["derived"]["C_mean"] and f"{np.mean(D):.2f}" == old["derived"]["D_mean"]
    fig, a = plt.subplots(figsize=(2.34, 1.95))
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.2, top=0.96)
    a.plot([x for x, _ in curve], [y for _, y in curve], color=fs.INK3, lw=1.0, zorder=2)
    for x, v, c in [(0.0, C, fs.TEAL), (0.59, D, fs.CORAL)]:
        a.scatter(x + np.linspace(-0.02, 0.02, len(v)), v, s=16, color=c, lw=0.5, edgecolor="white", zorder=4)
        a.scatter(x, np.mean(v), s=30, marker="s", color=fs.INK, edgecolor="white", lw=0.6, zorder=5)
    for x, y in old["ladder"].values():
        a.scatter(x, y, s=28, marker="D", color=fs.AMBER, edgecolor="white", lw=0.6, zorder=4)
    # exposure deltas against the dose-0 arm mean (receipt: +0.77, D_mean - C_mean, +3.00)
    for x, y in curve[1:]:
        dy = (-11, "top") if x == 0.3 else (5, "bottom")
        a.annotate(f"+{y - base:.2f}", (x, y), xytext=(0, dy[0]), textcoords="offset points",
                   ha="center", va=dy[1], fontsize=7.5, color=fs.INK)
    a.set(xlim=(-0.1, 1.1), ylim=(44.2, 50.0), xticks=[0, 0.3, 0.59, 1], xticklabels=["0", "0.30", "0.59", "1.0"])
    a.set_xlabel("fraction of story blocks in training", fontsize=fs.SIZE_TICK + 0.5, labelpad=2)
    a.set_ylabel("top-10 accuracy (%)", fontsize=fs.SIZE_TICK + 0.5)
    a.grid(axis="y", color=fs.GRID, lw=0.5)
    a.legend([Line2D([], [], marker="o", ls="", mfc=fs.TEAL, mec="white", ms=4.5),
              Line2D([], [], marker="o", ls="", mfc=fs.CORAL, mec="white", ms=4.5),
              Line2D([], [], marker="s", ls="", mfc=fs.INK, mec="white", ms=4.5),
              Line2D([], [], marker="D", ls="", mfc=fs.AMBER, mec="white", ms=4.5)],
             [f"unexposed run (n={len(C)})", f"exposed run (n={len(D)})", "arm mean", "single run"],
             loc="lower right", fontsize=7, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1)
    svg = OUT / "fig03_dose.svg"
    fig.savefig(svg); plt.close(fig)
    pdf, png = export(svg, width_in=2.34); one_page(pdf)
    print("wrote", pdf)


if __name__ == "__main__":
    fig02(); fig03()
