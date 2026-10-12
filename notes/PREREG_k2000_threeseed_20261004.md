# PREREG — k=2000 three-seed check of "the static encoder leads at k=2000" (#60432, rebuttal-only), 2026-10-04

Written 2026-10-04 02:55 +08:00 and committed before the analysis script exists. Receipt-only: no GPU, no new run,
inputs are the committed files under `results/strengthen_20260926/{inputs,runs,receipts}` (BCI_foundation 9462551).
Seen beforehand: the submitted Table 3 standard row at k=2000 (seed 0, n=9: −0.007, interval excludes zero); that
the strengthen runs carry k=2000 for nine subjects (s03 has none — too few calibration segments, as in Table 3);
nothing else at k=2000 from those runs (`analysis.json` has no k=2000 entry; I listed k-values, not scores).

**Question.** Does the abstract's "at k=2000 the static encoder leads under the standard protocol" (tex l.41–42;
also §4 l.434–436 and §7 l.556–558, which use it to argue the k=2000 reversals are "not purely an exposure
effect") survive averaging over three pretraining seeds? **Estimand.** Standard cell (recording guard,
recording split, four-story pretraining), k=2000, Δ_s = r(dynamic_big) − r(conv) (rank percentile `fm.meanpct`)
per held-out subject, averaged over seeds 0 (published `mb1f_ksweep`), 1 and 2 (`st26_rr_seed{1,2}`), n=9
subjects; interval = paired subject bootstrap, 20,000 draws, `default_rng(0)`, percentile (the published draws).
**Gates (stop on failure, no estimate):** G0′ the seed-0 per-subject k=2000 Δ from `inputs/` gives the published
−0.007 (rounded to 3 dp) with an interval excluding zero below; G1′ for every seed-1/2 RR file, sha256 equals its
receipt's `result_sha256`, the k=2000 n and pool equal the same subject's `mb1f` conv k=2000 n and pool, and the
radius-10 r is within 0.0005 of the primary r. **Decision rule (primary, one test, no correction):** SUPPORT if
the seed-mean interval lies below zero and all three seed means are < 0; REVERSED if the seed-mean interval lies
above zero; WEAKEN otherwise. **Wording per branch (rebuttal; tex unchanged).** SUPPORT: "Averaged over three
pretraining seeds the static encoder still leads at k=2000 under the standard protocol (Δ̄ [lo, hi], n=9; seed
means a / b / c); the story-withheld rows of Table 3 remain single-seed." WEAKEN: "Averaged over three seeds the
standard-protocol k=2000 lead of the static encoder is not resolved (Δ̄ [lo, hi]; seed means a / b / c), so the
abstract's k=2000 clause and the 'not purely an exposure effect' inference in §4 and §7 hold for seed 0 only."
REVERSED: "Averaged over three seeds the dynamic encoder leads at k=2000 under the standard protocol (Δ̄ [lo,
hi]); the seed-0 reversal does not replicate, and the abstract's k=2000 clause and the 'not purely an exposure
effect' inference are withdrawn." **Out of scope, stated whichever branch fires:** the "with the story withheld"
half of the clause (Table 3 story rows) has no seed replicate with the dynamic arm, so it is not tested here.
**Descriptive only (label, no branch, no claim):** the same three-seed k=2000 Δ̄ for the recording-guard/story-split
cell (seed 0 `mb1t_cellC`, seeds 1–2 `st26_rs`) and the clean stimulus/story cell (`st26_ssc` seeds 0–2); the
SR cell is omitted because its seed-1/2 runs are the same trained models as RR's (guard is evaluation-only).
