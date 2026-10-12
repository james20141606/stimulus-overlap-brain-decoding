"""Figure 1 v4 (2026-10-10): one audit-result panel, FIG1_GUIDE type 4A.

What it shows. Dynamic - static rank percentile (Delta, Eq. 1) with a paired 95% subject bootstrap,
at the two low budgets k=100 and k=500, one row per measurement, grouped by where the test story
reached the model:
  pretraining + calibration (standard protocol, recording guard / recording split)
      seed 0                      figures/fig_factorial_subjects.values.json rr (= Table 2 row 1, Fig. 2)
      seed 0, 2nd run             figures/fig_factorial_subjects.values.json sr (= Table 2 row 2): the
                                  stimulus-guard/recording-split cell; the guard acts only at evaluation and
                                  changes r by < 1e-4 at fixed model (App. H.2(d)), so this is a second
                                  training run of the standard protocol (added 2026-10-11, R1 pre-review P1-4)
      3 seeds                     analysis.json  results.Q3b[k].cells.rr            (= App. H.2)
      3 seeds, rebuilt archive    stage1_analysis.json  blocks[k].seed_avg          (10-10 bridge)
  pretraining only (story-disjoint calibration, recording guard / story split, all 4 stories pretrained)
      seed 0                      figures/fig_factorial_subjects.values.json rs (= Table 2 row 3, Fig. 2)
      3 seeds                     analysis.json  results.Q3b[k].cells.rs            (= App. H.2)
  neither (story withheld from pretraining and calibration, evaluated on it; seed 0)
      story 0..3                  results/table1_receipts/result_{mb1j_story0, mb1s_story1_unseen,
                                  mb1s_story2_unseen, mb1h_stimho}_s*.json, through
                                  mb_conditions.cell/ci exactly as scripts/grid_pct.py (= Table 3)
Grey vertical line: Delta = 0 (no advantage). Filled marker: interval excludes zero; open: it does not.
Ink = the story was in pretraining; accent = it was not.
The k=500 panel is drawn muted (2026-10-11): S4 says both k=500 stars clear zero by less than plausible
percentile-bootstrap undercoverage at n=10 and that the standard-protocol premise rests on k=100, so the
figure must not let k=500 carry the eye.

The script types no data value. It asserts that the seed-0 values reproduce the strings printed in
Tables 2 and 3 of paper_iclr.tex, and that the rebuilt-archive k=100 row is still unresolved (the caption
says so). Values drawn are written to figures/fig1_v4/fig1_v4.values.json.
Run: ~/.venvs/figtools/bin/python scripts/make_fig1_v4.py
"""
import contextlib
import re
import io
import json
import pathlib
import sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figstyle"))

from fig1_rc import C, apply  # noqa: E402

apply(base_pt=8.0)
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["savefig.bbox"] = None          # canvas = figsize, so \includegraphics prints at nominal size
plt.rcParams["savefig.pad_inches"] = 0.0
plt.rcParams["xtick.labelsize"] = 8.0

import compose  # noqa: E402

REC = ROOT / "results" / "table1_receipts"
_argv, sys.argv = sys.argv, [sys.argv[0], str(REC)]
with contextlib.redirect_stdout(io.StringIO()):
    import mb_conditions as M  # noqa: E402  (module body loads REC; only cell() and ci() are used)
sys.argv = _argv
from result_io import load_subject_results  # noqa: E402

KS = (100, 500)
# Public release: analysis.json ships as analysis_summary.json (run_records with compute-node names
# removed; "results" is unchanged), so read whichever is present.
_A = ROOT / "results/strengthen_20260926/analysis.json"
if not _A.exists():
    _A = ROOT / "results/strengthen_20260926/analysis_summary.json"
A = json.loads(_A.read_text())["results"]
S1 = json.loads((ROOT / "results/subjects27_20261004/stage1/stage1_analysis.json").read_text())["blocks"]
# seed-0 factorial cells exactly as Table 2 / Figure 2 print them (same mb_conditions.ci, subject order of
# interaction_receipt.json; analysis.json's Q1 re-bootstrap differs in the 4th decimal)
F0 = json.loads((ROOT / "figures/fig_factorial_subjects.values.json").read_text())["budgets"]


def cell0(k, c):
    d = next(x for x in F0[str(k)]["cells"] if x["cell"] == c)
    return {"mean": d["mean"], "lo": d["ci95"][0], "hi": d["ci95"][1], "n": F0[str(k)]["n"]}


def summ(d):
    return {"mean": d["mean"], "lo": d["ci95"][0], "hi": d["ci95"][1], "n": d["n"]}


def triple(t, n):
    return {"mean": t[0], "lo": t[1], "hi": t[2], "n": n}


def withheld(prefix):
    D = load_subject_results(str(REC), prefix, quiet=True)
    out = {}
    for k in KS:
        d = [M.cell(D[s], "dynamic_big", k)[2] - M.cell(D[s], "conv", k)[2] for s in sorted(D)
             if M.cell(D[s], "conv", k)[2] is not None and M.cell(D[s], "dynamic_big", k)[2] is not None]
        m, lo, hi = M.ci(d)
        out[k] = {"mean": float(m), "lo": float(lo), "hi": float(hi), "n": len(d)}
    return out


STORY = {0: "result_mb1j_story0_", 1: "result_mb1s_story1_unseen_",
         2: "result_mb1s_story2_unseen_", 3: "result_mb1h_stimho_"}
W_ = {s: withheld(p) for s, p in STORY.items()}

# (group, row label, {k: value}); top to bottom
ROWS = [
    ("std", "seed 0", {k: cell0(k, "rr") for k in KS}),
    ("std", "seed 0, rerun\n(stimulus guard)", {k: cell0(k, "sr") for k in KS}),
    ("std", "3 seeds", {k: summ(A["Q3b"][str(k)]["cells"]["rr"]) for k in KS}),
    ("std", "3 seeds,\nrebuilt archive", {k: triple(S1[str(k)]["seed_avg"], S1[str(k)]["n"]) for k in KS}),
    ("pre", "seed 0", {k: cell0(k, "rs") for k in KS}),
    ("pre", "3 seeds", {k: summ(A["Q3b"][str(k)]["cells"]["rs"]) for k in KS}),
] + [("none", f"story {s}" + (f" (n={W_[s][100]['n']})" if W_[s][100]["n"] != 10 else ""), W_[s])
     for s in (0, 1, 2, 3)]

# ---- cross-checks against the printed tables (seed 0) and the caption's claim
TEX = (ROOT / "paper_iclr.tex").read_text()  # public release: copy the arXiv source paper_iclr.tex to the repo root first


def fmt(v):
    """3 decimals, half-up on the decimal string (as the tables were typeset; check_paper_numbers does the same)"""
    q = Decimal(repr(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{q:+.3f}".replace("-0.000", "+0.000")


def need(s):
    assert s in TEX, f"not printed in paper_iclr.tex: {s}"


NUM = r"([+-]\d\.\d{3})"


def printed(line_key):
    """numbers on the one tex line that contains line_key (mean, lo, hi, ... in reading order)"""
    lines = [ln for ln in TEX.splitlines() if line_key in ln]
    assert len(lines) == 1, f"{line_key!r}: {len(lines)} lines"
    return [float(x) for x in re.findall(NUM, lines[0])]


def close(a, b):
    return abs(a - b) <= 0.00051          # 3-decimal typesetting (half-up; Table 2 prints the RS k=500 lo -0.0035 as -0.004 since 2026-10-10)


for key, row in (("recording, recording (standard)", ROWS[0][2]), ("stimulus, recording ", ROWS[1][2]),
                 ("recording, story ", ROWS[4][2])):
    got = printed(key)                     # k100 mean, lo, hi, k500 mean, lo, hi
    want = [row[k][f] for k in KS for f in ("mean", "lo", "hi")]
    assert len(got) == 6 and all(close(g, w) for g, w in zip(got, want)), (key, got, want)
for s in (0, 1, 2, 3):
    got = printed(f"story ${s}$ & ")       # Table 3: k100, k500, k2000 means
    assert all(close(g, W_[s][k]["mean"]) for g, k in zip(got, KS)), (s, got, W_[s])
    assert str(W_[s][100]["n"]) in re.findall(r"& +(\d+) &", [ln for ln in TEX.splitlines() if f"story ${s}$ & " in ln][0])
rb = ROWS[3][2][100]
assert rb["lo"] < 0 < rb["hi"], "caption says the rebuilt-archive k=100 interval includes zero; receipt disagrees"
r2 = ROWS[1][2][100]
assert r2["lo"] < 0 < r2["hi"], "caption says the second seed-0 standard run does not resolve at k=100; receipt disagrees"
assert all(not (W_[s][k]["lo"] > 0) for s in W_ for k in KS), "a withheld row now resolves positive; caption is wrong"
assert all(ROWS[i][2][100]["lo"] > 0 for i in (0, 2)), "standard rows (seed 0, 3 seeds) at k=100 no longer resolve"

# ---- geometry (inches)
W = 5.5
FS = 8.0
GAP = 0.45                                   # extra row units between groups
ys, y, prev = [], 0.0, None
for g, lbl, _ in ROWS:
    tall = 0.3 if "\n" in lbl else 0.0      # a two-line row label gets a little more room
    if prev is not None and g != prev:
        y += GAP
    y += tall
    ys.append(y)
    y += 1.0 + tall
    prev = g
YMAX = ys[-1]
ROW_IN = 0.152
TOP_PAD, BOT_PAD = 0.62, 0.62                # row units above first / below last row
plot_h = (YMAX + TOP_PAD + BOT_PAD) * ROW_IN
TITLE, BOT = 0.20, 0.36
H = plot_h + TITLE + BOT
GCOL, RCOL, PGAP, RIGHT = 0.74, 0.84, 0.22, 0.04
PW = (W - GCOL - RCOL - PGAP - RIGHT) / 2
fig = plt.figure(figsize=(W, H))
yl = (YMAX + BOT_PAD, -TOP_PAD)

axL = fig.add_axes([0, BOT / H, (GCOL + RCOL) / W, plot_h / H])
axL.set_xlim(0, GCOL + RCOL)
axL.set_ylim(*yl)
axL.axis("off")
axes = []
for i, k in enumerate(KS):
    x0 = GCOL + RCOL + i * (PW + PGAP)
    ax = fig.add_axes([x0 / W, BOT / H, PW / W, plot_h / H])
    ax.set_ylim(*yl)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    axes.append(ax)

COL = {"std": C.INK, "pre": C.INK, "none": C.ACC}
MUTED = {"std": "#9a9a9a", "pre": "#9a9a9a", "none": "#e0a39b"}   # k=500 panel: secondary (S4: may undercover)
GROUP = {"std": "pretraining +\ncalibration\n(standard)", "pre": "pretraining\nonly", "none": "neither\n(withheld;\none seed)"}
SEP = "#d0d0d0"

# group separators: one hairline across label column and both panels, midway between groups
bounds = [(ys[i] + ys[i + 1]) / 2 for i in range(len(ys) - 1) if ROWS[i][0] != ROWS[i + 1][0]]
for b in bounds:
    axL.plot([0.02, GCOL + RCOL], [b, b], color=SEP, lw=0.5, clip_on=False)
    for ax in axes:
        ax.axhline(b, color=SEP, lw=0.5, zorder=0)

# group labels (left column, vertically centred on their rows) and row labels (right-aligned)
for g in ("std", "pre", "none"):
    gy = [yy for (gg, _, _), yy in zip(ROWS, ys) if gg == g]
    axL.text(0.02, (gy[0] + gy[-1]) / 2, GROUP[g], ha="left", va="center", fontsize=FS,
             color=COL[g], linespacing=1.05)
for (g, lbl, _), yy in zip(ROWS, ys):
    axL.text(GCOL + RCOL - 0.06, yy, lbl, ha="right", va="center", fontsize=FS, color=COL[g], linespacing=1.0)
axL.text(0.02, -TOP_PAD, "test story was in", ha="left", va="bottom", fontsize=FS, color=C.INK)

LIM = {100: (-0.05, 0.09), 500: (-0.045, 0.05)}
TICKS = {100: [-0.04, 0.0, 0.04, 0.08], 500: [-0.04, -0.02, 0.0, 0.02, 0.04]}
for ax, k in zip(axes, KS):
    ax.set_xlim(*LIM[k])
    ax.set_xticks(TICKS[k])
    ax.set_xticklabels([f"{t:+.2f}".replace("+0.00", "0").replace("-", "−") for t in TICKS[k]])
    ax.axvline(0.0, color=C.REF, lw=0.8, zorder=1)
    col = COL if k == 100 else MUTED
    for (g, _, v), yy in zip(ROWS, ys):
        d = v[k]
        filled = d["lo"] > 0 or d["hi"] < 0
        ax.plot([d["lo"], d["hi"]], [yy, yy], color=col[g], lw=1.1 if k == 100 else 0.9,
                solid_capstyle="butt", zorder=2)
        ax.plot([d["mean"]], [yy], "o", ms=4.2 if k == 100 else 3.6, mew=1.0, mec=col[g],
                mfc=col[g] if filled else "white", zorder=3)
    ttl = f"k = {k} calibration segments" if k == 100 else f"k = {k} (secondary)"
    ax.text(0.5, 1.0, ttl, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=FS, color=C.INK if k == 100 else "#6f6f6f")
    if k != 100:
        for sp in ("bottom",):
            ax.spines[sp].set_color("#9a9a9a")
        ax.tick_params(axis="x", colors="#6f6f6f")

fig.text((GCOL + RCOL + W - RIGHT) / 2 / W, 0.02 / H,
         "dynamic − static rank percentile (Δ)", ha="center", va="bottom", fontsize=FS, color=C.INK)
# marker key, one line under the label column at the x-label's height (marks must read without the caption)
ky = 0.085 / H
for x, fill, txt in ((0.06, C.INK, "interval excludes 0"), (1.10, "white", "includes 0")):
    fig.lines.append(plt.Line2D([x / W], [ky], marker="o", ms=4.2, mew=1.0, mec=C.INK, mfc=fill,
                                transform=fig.transFigure))
    fig.text((x + 0.07) / W, ky, txt, ha="left", va="center", fontsize=FS, color=C.INK)

OUT = ROOT / "figures" / "fig1_v4"
OUT.mkdir(parents=True, exist_ok=True)
svg = OUT / "fig1_v4.svg"
fig.savefig(svg)
(OUT / "fig1_v4.values.json").write_text(json.dumps({
    "generated_by": "scripts/make_fig1_v4.py",
    "metric": "dynamic_big - conv rank percentile (meanpct), paired 95% subject bootstrap, 20000 draws, default_rng(0)",
    "receipts": ["figures/fig_factorial_subjects.values.json", "results/strengthen_20260926/analysis.json", "results/subjects27_20261004/stage1/stage1_analysis.json",
                 "results/table1_receipts/"],
    "rows": [{"group": g, "label": l, **{f"k{k}": v[k] for k in KS}} for g, l, v in ROWS]}, indent=1))
pdf, png = compose.export(svg, width_in=W, min_pt=7.0)
print("wrote", pdf, png, f"{W:.2f} x {H:.2f} in")
