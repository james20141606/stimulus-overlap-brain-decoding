#!/usr/bin/env python3
"""Render the pre-registered paper text for the calibration-diversity control from analysis.json.

Mechanical: the sentence templates are PREREG_calibmatch_20260924.md §7, verbatim; numbers come only
from results/calmatch_20260924/analysis.json. Prints (1) the Table 1 caption sentence, (2) the
appendix subsection, (3) checker pins as Python tuples. Writes nothing into the paper itself.
"""
import json
import sys

OLD_CAPTION = ("At equal $k$, UNSEEN calibrates on three stories and SEEN on\n"
               "two; no calibration-diversity-matched control was run for this design.}")


def f1(x):
    return f"{x:.1f}"


def sgn(x):
    return "0.0" if round(x, 1) == 0 else f"{x:+.1f}"


def caption(rep):
    v = rep["verdict"]
    p5, p2 = rep["rows"]["story0"]["conv|500"], rep["rows"]["story0"]["conv|2000"]
    head = ("At equal $k$, UNSEEN calibrates on three stories and SEEN on\n"
            "two; restricting UNSEEN to SEEN's two ")
    if v == "NO_MATERIAL_CHANGE":
        body = (f"changes the static retained fraction by ${sgn(p5['D_pp'])}$ and ${sgn(p2['D_pp'])}$ "
                f"points at $k{{=}}500$ and $2000$ (App.~\\ref{{app:calmatch}}).}}")
    elif v == "MATERIAL_CHANGE":
        d = (p5["D_pp"] + p2["D_pp"]) / 2
        tail = ("part of what UNSEEN retains comes from its more diverse calibration" if d < 0
                else "the published fraction understates what survives")
        body = (f"moves the static retained fraction to ${f1(p5['rho_match'])}\\%$ and "
                f"${f1(p2['rho_match'])}\\%$ at $k{{=}}500$ and $2000$ (App.~\\ref{{app:calmatch}}), so {tail}.}}")
    elif v == "INCONCLUSIVE":
        c5, c2 = p5["rho_match_ci"], p2["rho_match_ci"]
        body = (f"gives ${f1(p5['rho_match'])}\\%$ $[{f1(c5[0])},{f1(c5[1])}]$ and ${f1(p2['rho_match'])}\\%$ "
                f"$[{f1(c2[0])},{f1(c2[1])}]$ at $k{{=}}500$ and $2000$, which does not exclude a $10$-point "
                f"change (App.~\\ref{{app:calmatch}}).}}")
    else:
        return None
    return head + body


def appendix(rep, prereg_commit, analysis_sha):
    L = []
    L.append("\\subsection{Calibration-diversity control}\\label{app:calmatch}")
    L.append("At equal $k$ the UNSEEN arm of Table~\\ref{tab:headline} calibrates on stories $\\{1,2,3\\}$ "
             "and SEEN on $\\{1,2\\}$. A MATCHED arm keeps UNSEEN's pretraining ($\\{1,2,3\\}$) and "
             "calibrates on SEEN's stories only; with the same seed it draws SEEN's validation segments "
             "and first-$k$ calibration subsets, and every subject's evaluation set and pool are identical "
             "across the three arms. Its pretraining is a fresh run (a different $1{,}500$-segment "
             "pretraining-validation split and non-deterministic kernels), a realisation difference of "
             "the size of the seed rows of Table~\\ref{tab:headline}. $D$ is the matched minus the "
             "published retained fraction, per subject, in points; intervals are paired subject "
             "bootstraps ($20{,}000$ draws). The decision rule (a $10$-point margin on $D$ for the static "
             f"encoder at $k{{=}}500$ and $2000$) was fixed before any run; verdict: "
             f"\\textsc{{{rep['verdict'].replace('_', ' ').lower()}}}.")
    L.append("\\begin{center}\\small\\setlength{\\tabcolsep}{4pt}")
    L.append("\\begin{tabular}{lrrrrrrll}")
    L.append("\\toprule")
    L.append("row & $k$ & $n$ & SEEN & UNSEEN & MATCHED & published & matched & $D$ (points) \\\\")
    L.append("\\midrule")
    pins = []
    for row, lab in (("story0", "story $0$"), ("story1", "story $1$")):
        for arm, al in (("conv", "static"), ("dynamic_big", "dynamic")):
            for k in (100, 500, 2000):
                c = rep["rows"].get(row, {}).get(f"{arm}|{k}")
                if not c:
                    continue
                line = (f"{lab}, {al} & {k} & {c['n']} & ${c['seen_mean']:.3f}$ & ${c['u123_mean']:.3f}$ & "
                        f"${c['u12_mean']:.3f}$ & ${f1(c['rho_pub'])}\\%$ & ${f1(c['rho_match'])}\\%$ "
                        f"$[{f1(c['rho_match_ci'][0])},{f1(c['rho_match_ci'][1])}]$ & "
                        f"${sgn(c['D_pp'])}$ $[{sgn(c['D_ci'][0])},{sgn(c['D_ci'][1])}]$ \\\\")
                L.append(line)
                pins.append((f"calmatch {row} {arm} k{k}", line.rstrip(" \\")))
    L.append("\\bottomrule")
    L.append("\\end{tabular}\\end{center}")
    L.append(f"Pre-registration: \\texttt{{notes/PREREG\\_calibmatch\\_20260924.md}} (commit "
             f"\\texttt{{{prereg_commit}}}); receipts: \\texttt{{results/calmatch\\_20260924/analysis.json}} "
             f"(SHA-256 \\texttt{{{analysis_sha[:12]}}}) with per-run preflight and receipt files.")
    return "\n".join(L), pins


def main():
    rep = json.load(open(sys.argv[1]))
    prereg_commit, analysis_sha = sys.argv[2], sys.argv[3]
    cap = caption(rep)
    print("=== verdict:", rep["verdict"])
    print("=== caption (replaces):\n" + OLD_CAPTION + "\n=== with:\n" + str(cap))
    app, pins = appendix(rep, prereg_commit, analysis_sha)
    print("=== appendix:\n" + app)
    print("=== pins:")
    for p in pins:
        print(repr(p))


if __name__ == "__main__":
    main()
