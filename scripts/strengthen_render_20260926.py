#!/usr/bin/env python3
"""Render results/strengthen_20260926/analysis.json into a paste-ready appendix table (LaTeX) and the
PREREG branch sentences with their numbers filled in. Reads ONLY analysis.json; prints nothing for an
incomplete block except "not completed". Usage:
    python scripts/strengthen_render_20260926.py results/strengthen_20260926/analysis.json OUTDIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def f3(x):
    return f"{x:+.3f}".replace("+", "$+$").replace("-", "$-$") if x == x else "--"


def ci(s, key="ci95"):
    lo, hi = s[key]
    return f"{f3(s['mean'])} {{\\scriptsize[{f3(lo)}, {f3(hi)}]}}"


def star(s):
    return "$^{*}$" if s["status"] in ("resolved_pos", "resolved_neg") else ""


def plain(x):
    return f"{x:+.3f}"


def pci(s):
    return f"{plain(s['mean'])} [{plain(s['ci95'][0])}, {plain(s['ci95'][1])}]"


def main():
    a = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    R = a["results"]; B = a["blocks"]
    rows, sent = [], []
    q1 = R["Q1"]
    if q1.get("verdict_Q1") not in (None, "INCOMPLETE"):
        rows.append(r"\multicolumn{3}{l}{\emph{(a) Clean stimulus/story cell (all four stories in pretraining), seed 0}}\\")
        rows.append(f"clean stimulus/story cell & {ci(q1['100']['cell_ssc_clean'])}{star(q1['100']['cell_ssc_clean'])} & {ci(q1['500']['cell_ssc_clean'])}{star(q1['500']['cell_ssc_clean'])} \\\\")
        rows.append(f"standard cell (Table~\\ref{{tab:factorial}}) & {ci(q1['100']['cell_rr_pub'])}{star(q1['100']['cell_rr_pub'])} & {ci(q1['500']['cell_rr_pub'])}{star(q1['500']['cell_rr_pub'])} \\\\")
        for name, lab in (("split_main", r"split main $M_{\mathrm{split}}$"), ("guard_main", r"guard main $M_{\mathrm{guard}}$"),
                          ("three_way_I", "three-way $I$"), ("split_minus_guard", r"$M_{\mathrm{split}}-M_{\mathrm{guard}}$")):
            c1, c5 = q1["100"]["contrasts_clean"][name], q1["500"]["contrasts_clean"][name]
            dag = lambda c: r"$^{\dagger}$" if c["status_bonf4"] != "not_resolved" else ""
            rows.append(f"\\quad {lab} (clean) & {ci(c1)}{star(c1)}{dag(c1)} & {ci(c5)}{star(c5)}{dag(c5)} \\\\")
        sent.append(f"Q1 {q1['verdict_Q1']}: clean SS k=100 {pci(q1['100']['cell_ssc_clean'])}, k=500 {pci(q1['500']['cell_ssc_clean'])}; "
                    f"clean M_split k=500 {pci(q1['500']['contrasts_clean']['split_main'])} Bonferroni {q1['500']['contrasts_clean']['split_main']['ci_bonf4']} -> {q1['verdict_Q1b_split_main_k500']}")
    else:
        sent.append(f"Q1 not completed: {B['A']['missing'][:3]}... {len(B['A']['missing'])} missing, {len(B['A']['gate1_problems'])} gate problems")
    q2 = R["Q2"]
    if q2.get("verdict_Q2") not in (None, "INCOMPLETE"):
        rows.append(r"\midrule\multicolumn{3}{l}{\emph{(b) Standard cell over three pretraining seeds}}\\")
        for sd in ("0", "1", "2"):
            s1, s5 = q2["100"]["per_seed"][sd], q2["500"]["per_seed"][sd]
            rows.append(f"seed {sd}{' (published)' if sd == '0' else ''} & {ci(s1)}{star(s1)} & {ci(s5)}{star(s5)} \\\\")
        s1, s5 = q2["100"]["seed_mean"], q2["500"]["seed_mean"]
        rows.append(f"seed-averaged & {ci(s1)}{star(s1)} & {ci(s5)}{star(s5)} \\\\")
        rows.append(f"SD within subject across seeds / between subjects & {q2['100']['within_subject_across_seed_sd']:.3f} / {q2['100']['between_subject_sd_of_seed_means']:.3f} & {q2['500']['within_subject_across_seed_sd']:.3f} / {q2['500']['between_subject_sd_of_seed_means']:.3f} \\\\")
        sent.append(f"Q2 {q2['verdict_Q2']}: per-seed means k=100 {[round(x, 3) for x in q2['100']['per_seed_means']]}, seed-avg {pci(q2['100']['seed_mean'])}; k=500 seed-avg {pci(q2['500']['seed_mean'])} ({q2['k500_label']})")
    else:
        sent.append(f"Q2 not completed: {len(B['B']['missing'])} missing, {len(B['B']['gate1_problems'])} gate problems")
    q3 = R["Q3"]
    if q3.get("verdict_Q3") not in (None, "INCOMPLETE"):
        rows.append(r"\midrule\multicolumn{3}{l}{\emph{(c) Clean stimulus/story cell over three seeds}}\\")
        s1, s5 = q3["100"]["seed_mean_ssc"], q3["500"]["seed_mean_ssc"]
        rows.append(f"seed-averaged & {ci(s1)}{star(s1)} & {ci(s5)}{star(s5)} \\\\")
        if "diagonal_seed_mean" in q3["100"]:
            d1, d5 = q3["100"]["diagonal_seed_mean"], q3["500"]["diagonal_seed_mean"]
            rows.append(f"clean $-$ standard, seed-averaged & {ci(d1)}{star(d1)} & {ci(d5)}{star(d5)} \\\\")
        sent.append(f"Q3 {q3['verdict_Q3']}: seed-avg clean SS k=100 {pci(s1)}, k=500 {pci(s5)}; diagonal {q3.get('diagonal_labels')}")
    else:
        sent.append("Q3 not completed")
    q4 = R["Q4"]
    if q4.get("verdict_Q4") not in (None, "INCOMPLETE"):
        rows.append(r"\midrule\multicolumn{3}{l}{\emph{(d) Guard radius, seed-averaged over the new runs ($\pm5$ / $\pm10$ / $\pm20$\,s)}}\\")
        names = {"rr": "standard", "ssc": "clean stimulus/story", "rs": "recording guard, story split", "sr": "stimulus guard, recording split"}
        for cell, v in q4.items():
            if cell in ("verdict_Q4", "sensitive_items"):
                continue
            cells = []
            for k in ("100", "500"):
                e = v[k]
                cells.append(" / ".join(f"{f3(e[f'r{r}']['mean'])}{star(e[f'r{r}'])}" for r in ("5", "10", "20")))
            rows.append(f"{names[cell]} (seeds {','.join(map(str, v['seeds']))}) & {cells[0]} & {cells[1]} \\\\")
        sent.append(f"Q4 {q4['verdict_Q4']}: {q4['sensitive_items']}")
    else:
        sent.append("Q4 not completed")
    q5 = R["Q5"]
    if q5.get("verdict_Q5") not in (None, "INCOMPLETE"):
        rows.append(r"\midrule\multicolumn{3}{l}{\emph{(e) Single-ingredient ablation, seed 0 ($\Delta=T+E$)}}\\")
        for cell, lab in (("rr", "standard"), ("ssc", "clean stimulus/story")):
            for term, tl in (("trunk_T_noflow_minus_conv", "trunk $T$ (residual-free, no endpoint)"),
                             ("endpoint_E_dyn_minus_noflow", "endpoint branch $E$")):
                s1, s5 = q5["100"][cell][term], q5["500"][cell][term]
                rows.append(f"{lab}: {tl} & {ci(s1)}{star(s1)} & {ci(s5)}{star(s5)} \\\\")
        sent.append(f"Q5 {q5['verdict_Q5']}: RR T k=100 {pci(q5['100']['rr']['trunk_T_noflow_minus_conv'])}, E {pci(q5['100']['rr']['endpoint_E_dyn_minus_noflow'])}; "
                    f"SSc T {pci(q5['100']['ssc']['trunk_T_noflow_minus_conv'])}, E {pci(q5['100']['ssc']['endpoint_E_dyn_minus_noflow'])}")
    else:
        sent.append("Q5 not completed")
    tex = "\n".join([r"\begin{tabular}{lcc}", r"\toprule", r" & $k{=}100$ & $k{=}500$ \\", r"\midrule", *rows,
                     r"\bottomrule", r"\end{tabular}"])
    (out / "appendix_table.tex").write_text(tex + "\n")
    (out / "branch_numbers.txt").write_text("\n".join(sent) + "\n")
    print(tex); print(); print("\n".join(sent))


if __name__ == "__main__":
    main()
