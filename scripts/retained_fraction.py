#!/usr/bin/env python3
"""Regenerate the paper's headline table: how much above-chance signal survives holding the
evaluation story out of PRETRAINING.

WHY THIS EXISTS

Until today nothing in this repo computed these numbers. `mb_conditions.py` prints the SEEN and UNSEEN
means, but the retained fractions in Table~\\ref{tab:headline} -- and the "about 54%" in the abstract,
which is the single most load-bearing number in the paper -- had no generator. They were produced once,
by hand or in a session that no longer exists, and have been copied forward since. That is the same
situation that left a stale bootstrap CI in the abstract (see experiments_detail.md 2026-08-09T09:05),
and it is not a defensible state for the headline of a paper whose thesis is that people report numbers
their evidence does not support.

THE ESTIMATOR IS A REAL CHOICE, NOT A FORMATTING DETAIL

"Fraction of above-chance signal retained" can mean two different things:

  ratio of means   (mean(unseen) - 0.5) / (mean(seen) - 0.5)
  mean of ratios   mean over subjects of (unseen_i - 0.5) / (seen_i - 0.5)

They are not the same statistic. The second is a biased estimate of the population ratio -- Jensen --
and the bias grows as the denominator gets small and noisy. Empirically on this data they agree to
0.1pp at k=500 and k=2000, where SEEN sits near 0.86-0.88 and the denominator is stable, and diverge by
4pp at k=100, where SEEN is 0.779 and per-subject denominators scatter:

    k=100   ratio-of-means 67.7    mean-of-ratios 71.7
    k=500   ratio-of-means 54.0    mean-of-ratios 54.1
    k=2000  ratio-of-means 53.9    mean-of-ratios 53.9

The published table reports 71.7 / 54.1 / 53.9, i.e. **mean of ratios**. This repo's own BANNED list in
check_paper_numbers.py already forbids a hand-derived accuracy on exactly this ground ("Jensen error:
mean of ratios != ratio of means"), so the paper is currently using an estimator it elsewhere rejects.

DEFAULT HERE IS RATIO OF MEANS, and `--estimator mean-of-ratios` reproduces the published values so the
difference can be audited rather than argued about. The CI is a paired subject bootstrap: resample
subjects, recompute the whole statistic, take percentiles -- so the interval inherits whichever
estimator is selected.

Metric is mean_percentile throughout, which is pool-invariant. Accuracy and x-chance are not, and the
conditions being compared have different candidate pools. Above-chance is (percentile - 0.5); dividing
raw percentiles would let the 0.5 floor inflate every ratio toward 1.

Usage (on the cluster):
  $SCRATCH/envs/rs-qwen-env/bin/python scripts/retained_fraction.py
  ... scripts/retained_fraction.py --estimator mean-of-ratios     # reproduce the published table
"""
import argparse
import io
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_io import load_subject_results                                    # noqa: E402

# mb_conditions.py reads `sys.argv[1]` at module level to pick its results directory, so importing it
# with our own flags on the command line makes its `R` become "--estimator". Hide argv for the import.
_argv, sys.argv = sys.argv, sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):        # importing it prints its whole report
    import mb_conditions as M                                                 # noqa: E402
sys.argv = _argv

DEFAULT_RESULTS = os.environ.get(
    "BCI_RESULTS", "$SCRATCH/project/results/mb")

CHANCE = 0.5
BOOT = 20000
BUDGETS = (100, 500, 2000)

# (label, SEEN prefix, UNSEEN prefix). SEEN = the evaluation story WAS in pretraining.
# story 0: mb1k holds story 3 out of pretraining and scores story 0, which therefore remains seen;
# mb1j_story0 holds story 0 itself out. Same pretraining volume either way, which is the point.
ROWS = [
    ("story 0", "result_mb1k_volctl_",       "result_mb1j_story0_"),
    ("story 1", "result_mb1s_story1_seen_",  "result_mb1s_story1_unseen_"),
    ("story 2", "result_mb1s_story2_seen_",  "result_mb1s_story2_unseen_"),
    # The three rows below regenerate numbers that previously had NO reader (audit 2026-08-19):
    # the composition-matched bracket half, the mb1u seed replicates, and the cross-story contrast.
    # Missing prefixes are skipped, so these rows are inert on machines without the result files.
    ("story 0 comp-matched", "result_mb1k_volctl_", "result_mb1r_compmatch_"),
    ("story 0 seed 1", "result_mb1u_seen_seed1_",  "result_mb1u_unseen_seed1_"),
    ("story 0 seed 2", "result_mb1u_seen_seed2_",  "result_mb1u_unseen_seed2_"),
    ("story 3 vs story 0 (cross-story)", "result_mb1k_volctl_", "result_mb1h_stimho_"),
]
ARMS = [("static conv", "conv"), ("dynamics-aware", "dynamic_big")]


def subj_key(x):
    """NUMERIC subject order (1, 2, 3, 5, ..., 13), the order in which the published Table 1 intervals
    were bootstrapped. result_io normalises '01' -> '1' (since 2026-08-19); sorting those strings
    lexicographically ('1','10','11','13','2',...) pairs the fixed default_rng(0) draws with different
    subjects and moved four published CI bounds by 0.1. Fixed 2026-09-24; --check-published guards it."""
    return (0, int(x)) if str(x).isdigit() else (1, str(x))


# Table 1 of the paper as printed (mean-of-ratios). Top block: (SEEN, UNSEEN, retained, lo, hi);
# bottom block: retained point estimate only (the paper prints no interval there), except story 2 at
# k=500, whose interval the caption quotes.
PUBLISHED = {
    ("story 0", "conv", 100): (0.779, 0.689, 71.7, 60.2, 83.6),
    ("story 0", "conv", 500): (0.861, 0.695, 54.1, 51.4, 56.6),
    ("story 0", "conv", 2000): (0.884, 0.707, 53.9, 52.2, 55.8),
    ("story 1", "conv", 100): (0.804, 0.679, 61.7, 52.1, 75.7),
    ("story 1", "conv", 500): (0.852, 0.681, 51.1, 43.4, 56.6),
    ("story 1", "conv", 2000): (0.881, 0.706, 54.0, 52.7, 55.5),
    ("story 0", "dynamic_big", 100): (0.799, 0.677, 60.7, 53.8, 67.9),
    ("story 0", "dynamic_big", 500): (0.855, 0.689, 53.5, 50.3, 56.6),
    ("story 0", "dynamic_big", 2000): (0.878, 0.690, 50.3, 48.5, 52.4),
    ("story 0 comp-matched", "conv", 100): 53.4, ("story 0 comp-matched", "conv", 500): 45.2,
    ("story 0 comp-matched", "conv", 2000): 49.3,
    ("story 3 vs story 0 (cross-story)", "conv", 100): 64.5,
    ("story 3 vs story 0 (cross-story)", "conv", 500): 48.3,
    ("story 3 vs story 0 (cross-story)", "conv", 2000): 48.7,
    ("story 3 vs story 0 (cross-story)", "dynamic_big", 100): 55.5,
    ("story 3 vs story 0 (cross-story)", "dynamic_big", 500): 46.8,
    ("story 3 vs story 0 (cross-story)", "dynamic_big", 2000): 46.6,
    ("story 2", "conv", 100): 66.5, ("story 2", "conv", 500): (None, None, 57.6, 56.2, 59.5),
    ("story 0 seed 1", "conv", 100): 59.4, ("story 0 seed 2", "conv", 100): 58.8,
    ("story 0 seed 1", "conv", 500): 53.0, ("story 0 seed 2", "conv", 500): 53.0,
    ("story 0 seed 1", "conv", 2000): 52.0, ("story 0 seed 2", "conv", 2000): 53.7,
}


def retained(seen, unseen, how):
    """Percent of above-chance signal retained. seen/unseen are paired per-subject percentiles."""
    if how == "ratio-of-means":
        den = seen.mean() - CHANCE
        return np.nan if den <= 0 else (unseen.mean() - CHANCE) / den * 100.0
    per = (unseen - CHANCE) / (seen - CHANCE)
    return per.mean() * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estimator", choices=("ratio-of-means", "mean-of-ratios"),
                    default="ratio-of-means")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--check-published", action="store_true",
                    help="exit nonzero unless every Table 1 value in PUBLISHED is reproduced as printed "
                         "(forces --estimator mean-of-ratios, the paper's estimator)")
    a = ap.parse_args()
    if a.check_published:
        a.estimator = "mean-of-ratios"
    got = {}

    print(f"estimator: {a.estimator}   metric: mean_percentile (pool-invariant)   "
          f"above-chance = pct - {CHANCE}")
    print(f"CI: paired subject bootstrap, {BOOT} draws, seed 0\n")
    print(f"{'row':<26} {'k':>5} {'n':>3} {'SEEN':>7} {'UNSEEN':>8} {'retained':>9}  {'95% CI':>16}")
    print("-" * 82)

    any_row = False
    for arm_label, arm in ARMS:
        for row_label, p_seen, p_unseen in ROWS:
            try:
                S = load_subject_results(a.results, p_seen, quiet=True)
                U = load_subject_results(a.results, p_unseen, quiet=True)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"{row_label + ', ' + arm_label:<26} {'--':>5} skipped: {e}")
                continue
            common = sorted(set(S) & set(U), key=subj_key)
            if len(common) < 3:
                print(f"{row_label + ', ' + arm_label:<26} {'--':>5} only {len(common)} paired "
                      f"subjects, skipped")
                continue
            for k in BUDGETS:
                seen, unseen = [], []
                for s in common:
                    x = M.cell(S[s], arm, k)[2]      # index 2 = mean_percentile
                    y = M.cell(U[s], arm, k)[2]
                    if x is not None and y is not None:
                        seen.append(x); unseen.append(y)
                if len(seen) < 3:
                    continue
                seen, unseen = np.array(seen), np.array(unseen)
                # The mean-of-ratios estimator is heavy-tailed when a per-subject denominator
                # (seen_i - 0.5) approaches zero and undefined at zero. Surface the margin instead of
                # silently averaging through it.
                dmin = float(np.min(seen - CHANCE))
                if dmin <= 0:
                    print(f"{row_label + ', ' + arm_label:<26} {k:>5} SKIPPED: a subject sits at or "
                          f"below chance in SEEN (min denominator {dmin:+.4f}); the ratio is "
                          f"undefined there")
                    continue
                pt = retained(seen, unseen, a.estimator)
                rng = np.random.default_rng(0)
                idx = rng.integers(0, len(seen), size=(BOOT, len(seen)))
                # Resample SUBJECTS, keeping each subject's (seen, unseen) pair together -- they are
                # the same fitted model evaluated two ways, so breaking the pairing would price a
                # between-subject difference that the design removes.
                bs = np.array([retained(seen[i], unseen[i], a.estimator) for i in idx])
                bad = int(np.sum(~np.isfinite(bs)))
                if bad:
                    # nanpercentile would silently condition the interval on non-degeneracy; say so.
                    print(f"    note: {bad}/{BOOT} bootstrap draws degenerate (denominator at/below "
                          f"chance in the resample); interval is conditional on the rest")
                bs = bs[np.isfinite(bs)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                print(f"{row_label + ', ' + arm_label:<26} {k:>5} {len(seen):>3} {seen.mean():>7.3f} "
                      f"{unseen.mean():>8.3f} {pt:>8.1f}%  [{lo:>6.1f},{hi:>6.1f}]")
                got[(row_label, arm, k)] = (round(float(seen.mean()), 3), round(float(unseen.mean()), 3),
                                            round(float(pt), 1), round(float(lo), 1), round(float(hi), 1))
                any_row = True
            print()
    if not any_row:
        sys.exit("FAIL: no rows computed. This machine has no result files at "
                 f"{a.results!r} -- the per-table regeneration runs on the cluster, or point "
                 "BCI_RESULTS at a results directory. A green exit here would be a false pass "
                 "(audit 2026-08-19).")

    print("Removed = 100 - retained. Quote the range over the rows you actually show; the paper's "
          "prose must cover its own table.")
    if a.check_published:
        bad = []
        for key, want in PUBLISHED.items():
            g = got.get(key)
            if g is None:
                bad.append(f"{key}: not computed (missing result files?)")
                continue
            if isinstance(want, tuple):
                pairs = [(w, v) for w, v in zip(want, g) if w is not None]
            else:
                pairs = [(want, g[2])]
            if any(w != v for w, v in pairs):
                bad.append(f"{key}: published {want} vs regenerated {g}")
        print(f"\ncheck-published: {len(PUBLISHED) - len(bad)}/{len(PUBLISHED)} Table 1 entries reproduced")
        for b in bad:
            print("  MISMATCH", b)
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
