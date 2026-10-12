#!/usr/bin/env python3
"""Re-derive every printed one-bit number from the 3-decimal wer readouts in
results/onebit_wer_readouts.json (copied verbatim from the cluster training logs) and
compare with the pins in scripts/check_paper_numbers.py and the ONEBIT dict in
scripts/fig_results.py. Exact Decimal arithmetic, ROUND_HALF_UP; also reports what the
same quantity gives when the two endpoints are rounded to 2 dp first, because a range
endpoint sits on a .xx5 boundary.  Exit 1 on any mismatch."""
import json, re, sys, pathlib
from decimal import Decimal, ROUND_HALF_UP
HERE = pathlib.Path(__file__).resolve().parent
R = json.load(open(HERE.parent / "results/onebit_wer_readouts.json"))["runs"]
q2 = lambda d: d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
top = {k: Decimal("100") - Decimal(v["wer"]) for k, v in R.items()}          # 3-dp exact
C = {k: v for k, v in top.items() if k.startswith("Cprime")}
D = {k: v for k, v in top.items() if k.startswith("D_")}
fails = []
def check(name, got, want):
    ok = str(got) == str(want); print(f"{'ok ' if ok else 'BAD'} {name:<34} derived {got!s:>8}  printed {want}")
    if not ok: fails.append(name)
# 1) per-run top-10 at 2 dp vs fig_results.py ONEBIT
ob = re.search(r"ONEBIT = (\{.*?\}\})", open(HERE / "fig_results.py").read(), re.S).group(1)
ONEBIT = eval(ob.replace("\n", ""))
for arm, seeds in ONEBIT.items():
    for s, v in seeds.items():
        check(f"top-10 {arm}_{s}", q2(top[f"{arm}_{s}"]), Decimal(str(v)).quantize(Decimal("0.01")))
# 2) chain pairs (3-dp difference, then round) vs "+1.83 / +2.89 / +3.75"
pairs = [("D_4402", "Cprime_4401", "1.83"), ("D_4404", "Cprime_4403", "2.89"), ("D_4406", "Cprime_4405", "3.75")]
for d, c, want in pairs:
    exact = top[d] - top[c]
    check(f"chain {d}-{c} ({exact})", q2(exact), Decimal(want))
# 3) 5v3 separation, span, mean
cross = sorted(dv - cv for dv in D.values() for cv in C.values())
print(f"    all 15 cross-arm differences positive: {all(x > 0 for x in cross)}  (p = 1/C(8,5) = 1/56 only if exchangeable; see App. H)")
check("span max (3-dp)", q2(cross[-1]), Decimal("4.59"))
lo3 = cross[0]; lo2 = q2(min(D.values())) - q2(max(C.values()))
print(f"    span min: 3-dp exact {lo3} -> {q2(lo3)} ; endpoints rounded first {q2(min(D.values()))}-{q2(max(C.values()))} = {lo2}")
check("span min (endpoints at 2 dp, paper convention)", lo2, Decimal("1.01"))
check("mean D - mean C' (5v3)", q2(sum(D.values()) / 5 - sum(C.values()) / 3), Decimal("2.40"))
check("mean chain pair (r0, 3 pairs)", q2(sum(top[d] - top[c] for d, c, _ in pairs) / 3), Decimal("2.82"))
print("\nRESULT:", "ALL MATCH" if not fails else f"{len(fails)} MISMATCH: {fails}")
sys.exit(1 if fails else 0)
