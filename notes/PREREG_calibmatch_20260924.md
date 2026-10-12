# PRE-REGISTRATION — calibration-diversity control for Table 1 (bci_split)

Written 2026-09-24, before any job for this control exists and before any MB-1n value has
been read. clusterA is unreachable while this is written (no the university VPN), so no result of any kind can have
been seen. The commit that adds this file is the timestamp. After results exist this file is not
edited; any departure is recorded as a dated **Deviation** entry appended at the end, with its reason.

Brief: (an internal review note), item 3.

## 1. Question

Table 1 (`paper_iclr.tex`, `tab:headline`) compares, on the same evaluation story and pool,

| arm | pretraining (other 26 subjects) | calibration stories (held-out subject) | evaluation | files (reused) |
|---|---|---|---|---|
| SEEN | {0,1,2} | {1,2} | story 0 | `result_mb1k_volctl_s*.json` (n=10) |
| UNSEEN | {1,2,3} | **{1,2,3}** | story 0 | `result_mb1j_story0_s*.json` (n=10) |

At equal k the UNSEEN arm draws its k calibration segments from three stories and SEEN from two. The
caption currently says "no calibration-diversity-matched control was run for this design". **Does
matching calibration diversity change the retained fraction that Table 1 reports?**

The only possible matched pair keeps SEEN as published (story 3 is withheld from its pretraining and
story 0 is evaluated, so {1,2} is all it can calibrate on) and restricts UNSEEN to SEEN's stories:

| new arm | pretraining | calibration | evaluation | tag |
|---|---|---|---|---|
| MATCHED (U12) | {1,2,3} (as UNSEEN) | {1,2} (as SEEN) | story 0 | `mb1z_calmatch_story0_s<S>` |

With `--seed 0`, `train_mb.py` draws the validation/calibration permutation from `default_rng(0)` over
the calibration pool. MATCHED's pool is SEEN's pool, so MATCHED gets **SEEN's exact 300 validation
segments and SEEN's exact first-k calibration subsets**, and UNSEEN's evaluation set and pool. The
per-subject preflight (`scripts/calmatch_preflight_20260924.py`) re-derives and hashes all three and
refuses to train if they differ.

What MATCHED does *not* hold fixed relative to the published UNSEEN run (disclosed, not removable):
(a) the same `rng` then draws the 1,500-segment pretraining-validation split, and its state depends
on the calibration-pool length, so MATCHED's pretraining validation split differs from UNSEEN's;
(b) cuDNN kernels are non-deterministic in the historical recipe (determinism is what caused the 195×
slowdown, §6), so pretraining is not bit-identical anyway. Both are pretraining-realisation noise.
The existing seed rows of Table 1 bound that noise at about 1–2 points of retained fraction at
k ≥ 500 (seeds 1/2: 53.0/53.0 vs 54.1 at k=500, 52.0/53.7 vs 53.9 at k=2000), well inside the 10-point
margin below. Change of GPU node or partition (gpuA_partition vs gpuB_partition; both H200) is in the
same class.

MB-1n (JID 15159191, 2026-08-02) pinned calibration to {1,2} for the *older* cross-story design
(UNSEEN evaluated on story 3). The ELN records it only as "4/10" at 2026-08-02 06:00 and no value
from it has ever been logged. It is not the Table 1 design and is not the primary analysis.

## 2. Estimands

Per subject s, encoder m ∈ {static conv `conv`, dynamic `dynamic_big`}, budget k ∈ {100, 500, 2000},
with r the fine-tuned (`fm`) mean rank percentile (`meanpct`, pool-invariant, chance ½):

- ρ_pub,s   = (r_U123,s − ½) / (r_S,s − ½)   — the published retained fraction
- ρ_match,s = (r_U12,s − ½) / (r_S,s − ½)    — calibration-diversity-matched
- D_s = ρ_match,s − ρ_pub,s = (r_U12,s − r_U123,s) / (r_S,s − ½)   (percentage points)

Point estimates are subject means (mean of ratios — the estimator Table 1 uses). Intervals are 95%
paired subject bootstraps: 20,000 draws, `numpy.random.default_rng(0)`, percentile, resampling
subjects and keeping each subject's (SEEN, UNSEEN, MATCHED) triple together — the
`scripts/retained_fraction.py` convention, so the published CIs are reproduced exactly (Gate 0).
Also reported: mean paired Δr = r_U12 − r_U123 with the same bootstrap.

**Primary cells:** static conv, story 0, k = 500 and k = 2000 (the "k ≥ 500" claim in the abstract and
§4). Everything else (dynamic encoder, k=100, story 1, MB-1n rows) is **descriptive** and cannot
change the verdict.

## 3. Sample, inclusion, exclusion

- Primary: the ten Table 1 subjects 01 02 03 05 07 08 09 10 11 13, seed 0, one run each. n = 10 at
  k = 100/500 and n = 9 at k = 2000 (the subject whose SEEN calibration pool is too small for k=2000;
  MATCHED has the same pool, so the same subject drops). **All primary subjects must complete.** If
  fewer have completed by the cut-off (2026-09-25 09:30), the verdict is INCOMPLETE and the
  paper is not changed — no partial-n result is reported in the paper and no interim look decides
  whether to continue. Completed results are still logged in the ELN.
- Secondary (story 1, descriptive): subjects 01 02 05 07 08 (the Table 1 story-1 subjects);
  SEEN `mb1s_story1_seen` (pretrain {0,1,2}, calib {0,2}), UNSEEN `mb1s_story1_unseen` (pretrain
  {0,2,3}, calib {0,2,3}), MATCHED `mb1z_calmatch_story1` (pretrain {0,2,3}, calib {0,2}). Runs after
  the primary; reported only if all five finish by the cut-off.
- Reuse-only (descriptive): MB-1n files `result_mb1n_unseen_s*` (calibration-matched cross-story row,
  vs `mb1k`) and `result_mb1n_seen_s*` (a replay of the SEEN configuration, i.e. an estimate of rerun
  noise), if present on clusterA, whatever their n.
- Exclusion: none by value. A subject is missing only if its job fails an engineering gate (preflight
  FAIL, non-zero exit, result/receipt SHA mismatch); failures are reported, not replaced.
- Files are loaded by the `test_subject` field; names containing smoke/test/tmp/debug/scratch are
  ignored (`scripts/result_io.py` convention).

## 4. Gates (must pass before any number is read as a result)

- **Gate 0 — reproduction.** From the reused SEEN/UNSEEN files alone, the analysis must reproduce
  the six published story-0 rows of Table 1 exactly as printed (SEEN mean, UNSEEN mean to 3 decimals;
  retained fraction and CI bounds to 0.1): conv 0.779/0.689/71.7 [60.2,83.6], 0.861/0.695/54.1
  [51.4,56.6], 0.884/0.707/53.9 [52.2,55.8]; dynamic 0.799/0.677/60.7 [53.8,67.9], 0.855/0.689/53.5
  [50.3,56.6], 0.878/0.690/50.3 [48.5,52.4]. If it does not, stop: `retained_fraction.py --estimator
  mean-of-ratios` is run on the same files; if that reproduces the paper, the new script is fixed (a
  logged Deviation) and rerun; otherwise nothing enters the paper.
- **Gate 1 — design.** Every MATCHED result has holdout_story = eval story, calib_stories as above,
  guard_scope stimulus; its preflight is PASS; the result SHA matches its receipt; and for every
  subject and k the evaluation `n` and `pool` are identical across SEEN, UNSEEN and MATCHED.
- **Gate 2 — backend.** Preflight PASS includes cudnn.deterministic False, CUBLAS_WORKSPACE_CONFIG
  unset, and a conv training step ≤ 0.2 s at B=256 on the allocated GPU.

## 5. Decision rule (margin 10 percentage points; primary cells only)

- **NO_MATERIAL_CHANGE** — at both k = 500 and k = 2000, the 95% CI of mean D lies strictly inside
  (−10, +10). The calibration-diversity asymmetry does not account for a material part of the
  retained fraction; the limitation sentence is replaced by the measured shift.
- **MATERIAL_CHANGE** — at k = 500 or k = 2000, the CI of mean D excludes 0 **and** |mean D| ≥ 10.
- **INCONCLUSIVE** — anything else (e.g. a resolved shift whose CI crosses ±10, or a wide CI).
- **INCOMPLETE / GATE_FAIL** — no verdict; paper unchanged; ELN only.

Why 10 points: the paper's own bracket for the retained fraction is 45–54% at k ≥ 500 and it says
changing stimulus, narrative or seed moves the fraction "by a few points"; a 10-point shift would
move the estimate outside that bracket, a smaller one would not change any claim.

Reporting rule for the non-primary cells: if the dynamic encoder at k = 500 or 2000 meets the
MATERIAL_CHANGE condition while the primary verdict is not MATERIAL_CHANGE, the main-text sentence
names it (one clause); otherwise non-primary cells appear only in the appendix table.

The rule is implemented in `scripts/calmatch_analysis_20260924.py::verdict` and mutation-tested in
`scripts/test_calmatch_20260924.py` (nine hand-built cases incl. the −10 boundary and n=9/n=8
incompleteness) before any real result exists.

## 6. Compute (fixed before submission)

- Code: pinned bundle `scripts/calmatch_20260924/src_pinned/` — `train_mb.py` SHA `1a4953ec…`
  (identical to the file on clusterA that produced the headline runs, checked 2026-09-22) and
  `eval_metrics.py` SHA `c5b94117…` from commit 4566e61 (2026-07-28), the scorer in place for every
  Table 1 run (mb1k 07-31 … mb1s early Aug). The live `src/eval_metrics.py` (08-18 hardening) uses
  `_tie_hits` before assignment in `guarded_retrieval_metrics` and crashes at the first evaluation
  (reproduced locally on synthetic data by the test script); it is not used.
- Backend: historical — no cudnn determinism, `CUBLAS_WORKSPACE_CONFIG` unset. The 2026-09-22 smoke
  18254449 set both and ran ~195× slower (synthetic profile 18321669: conv step 2.03 s vs 0.0094 s).
- Cost: historical comparable single-H200 subject runs (two 80-epoch pretrains, three budgets,
  fine-tune + scratch) took 826–1192 s; 20 of them took 5 h 14 min (JIDs 15282048, 15444988), i.e.
  ≈ 0.26 GPU-h each. Primary 10 runs ≈ 2.6 GPU-h, story 1 five runs ≈ 1.3 GPU-h; with two subjects
  per GPU the allocation is 8 tasks × ≤ 1.5 h wall = ≤ 12 GPU-h ceiling, ≈ 4 GPU-h expected. Under the
  regressed backend a single subject would cost ≈ 50 GPU-h — the preflight refuses to train then.
- Script: `scripts/slurm/calmatch_20260924.sbatch`, array 0-7%2 (2-GPU per-user cap), tasks 0–4
  primary (two subjects per GPU, second starts 300 s later), 5–7 story 1. Per-subject result JSON is
  the checkpoint; a resubmitted array skips completed subjects. GPU utilisation is sampled every 15 s
  by nvidia-smi and summarised in each receipt.
- Outputs: `results/mb/result_mb1z_calmatch_story{0,1}_s*.json` (train_mb.py's own format),
  `results/calmatch_20260924/{preflight_,receipt_}*.json`, analysis receipt
  `results/calmatch_20260924/analysis.json` (input SHAs, script SHA, seed, verdict).

## 7. Paper text for each outcome (written now; numbers filled from `analysis.json` only)

Main-text edits replace existing text with **net zero added lines** (page gate: body ends on p. 9).
The appendix gains `\subsection{Calibration-diversity control}\label{app:calmatch}` in "Extended
results and caveats": one table (story 0 both encoders × three budgets: n, SEEN, UNSEEN, MATCHED,
ρ_pub, ρ_match [CI], D [CI]; story 1 and MB-1n rows if they exist) plus three sentences: the design
of §1, the realisation-noise caveat of §1, and the paths of this file and of `analysis.json`.
`LEAKAGE.md` (supplement) gets the same one sentence as the caption.

Current Table 1 caption, last sentence (the limitation):
> At equal $k$, UNSEEN calibrates on three stories and SEEN on two; no calibration-diversity-matched
> control was run for this design.

**NO_MATERIAL_CHANGE** — caption:
> At equal $k$, UNSEEN calibrates on three stories and SEEN on two; restricting UNSEEN to SEEN's two
> changes the static retained fraction by $D_{500}$ and $D_{2000}$ points at $k{=}500$ and $2000$
> (App.~\ref{app:calmatch}).

(D printed signed to one decimal, e.g. "$-1.8$ and $+0.6$ points".) Abstract, §4 bracket and
conclusion unchanged.

**MATERIAL_CHANGE** — caption:
> At equal $k$, UNSEEN calibrates on three stories and SEEN on two; restricting UNSEEN to SEEN's two
> moves the static retained fraction to $\rho^{m}_{500}\%$ and $\rho^{m}_{2000}\%$ at $k{=}500$ and
> $2000$ (App.~\ref{app:calmatch}), so [D<0:] part of what UNSEEN retains comes from its more diverse
> calibration [D>0:] the published fraction understates what survives.

§4 "About half the signal is the stimulus", bracket sentence: "the two designs bracket the retained
fraction at roughly $45$--$54\%$" → "…at roughly $L$--$H\%$ ($\rho^{m}$ with calibration diversity
matched)", L/H = min/max of {45, 54, ρ^m_500, ρ^m_2000} rounded to integers; words trimmed in the same
paragraph to keep the line count. Abstract: "leaves about $54\%$ of above-chance signal ($k{\geq}500$;
three narratives, three seeds)" → "… ($k{\geq}500$; three narratives, three seeds; $X\%$ with
calibration diversity matched, one narrative)", X = mean of ρ^m_500 and ρ^m_2000 rounded; then
`check_abstract_supported.py` is rerun and the coordinating session is told to sync FINAL_ABSTRACTS.

**INCONCLUSIVE** — caption:
> At equal $k$, UNSEEN calibrates on three stories and SEEN on two; restricting UNSEEN to SEEN's two
> gives $\rho^{m}_{500}\%$ $[lo,hi]$ and $\rho^{m}_{2000}\%$ $[lo,hi]$ at $k{=}500$ and $2000$, which
> does not exclude a $10$-point change (App.~\ref{app:calmatch}).

Abstract and §4 unchanged.

**INCOMPLETE / GATE_FAIL / not finished by 2026-09-25 09:30** — no paper edit. ELN entry with what
ran, what failed and why. The existing limitation sentence stays.

§3 ("The matched exposure contrast") is not edited in any outcome; the caption carries the result.

## 8. Integration gates (after the verdict, before 2026-09-25 12:00)

pdflatex 0 error / 0 undefined; all `check_*.py` in the repo (FAILURES 0) with new pins for every
new number (caption and appendix table rows) and a ban on the retired "no calibration-diversity-
matched control was run" phrase, mutation-tested; (an internal review note);
body ends on p. 9; pdffonts Type 3 = 0; changed pages viewed at 200 dpi; ELN with real clock time;
explicit-path commits; `ICLR_submit/refresh.sh`; supplement rebuilt with the new scripts and
receipts, identity scan (literal + split/encoded), copied to `ICLR_submit/`. No push, no upload.

## Deviation 1 — 2026-09-24T13:43:03 (Gate 0, before any MATCHED job was submitted)

Gate 0 on the reused files (40 JSONs pulled from clusterA) reproduced all SEEN/UNSEEN means and all six
retained-fraction point estimates exactly, but four printed CI bounds differed by 0.1 (conv k=100
lower 60.1 vs 60.2, conv k=2000 lower 52.1 vs 52.2, dynamic k=100 lower 53.9 vs 53.8, dynamic k=2000
lower 48.4 vs 48.5), in both directions. As §4 prescribes, `retained_fraction.py --estimator
mean-of-ratios` was run on the same files: it gives the same four off-by-0.1 bounds, so it does not
reproduce the paper either. Cause, verified: since the 2026-08-19 `result_io.py` change, subject
keys are int-normalised ('01' -> '1') and sorted lexicographically ('1','10','11','13','2',...), so
the fixed `default_rng(0)` index draws are paired with different subjects than when the published
intervals were computed (zero-padded order '01'...'13'). Recomputing with numeric subject order
reproduces all six published rows exactly (means, points, both CI bounds).

Change: `calmatch_analysis_20260924.py` now orders subjects numerically everywhere (`subj_key`);
Gate 0 then passes 6/6 exactly. Estimator, bootstrap, margin, decision rule and paper text are
unchanged. No MATCHED result existed when this was found or fixed (no job had been submitted).
The published generator's ordering drift is reported to the coordinating session separately; it is not
changed here.

## Deviation 2 — 2026-09-25T02:03:47 (mixed compute site; fixed before any clusterB job exists)

State when written: clusterA job 18404006 has finished MATCHED subjects 01, 02, 03, 05 (tasks 0–1); tasks
2–7 (subjects 07 08 09 10 11 13 and story 1) are PENDING on gpuA_partition. No clusterB job exists. No
MATCHED value has entered any analysis (none has been computed by `calmatch_analysis_20260924.py`; the
analysis waits for the selection below). Author decision (relayed by the coordinating session): run the six
remaining primary subjects on clusterB as insurance, in parallel with clusterA.

(a) **What runs where.** clusterA keeps 18404006 queued unchanged (H200, as planned). clusterB runs
MATCHED story 0 for subjects 07 08 09 10 11 13 (`scripts/slurm/calmatch_20260924_clusterB.sbatch (not shipped)`,
`--array=4-9%4`, gpuC_partition, L40S) with the same pinned bundle (train_mb.py 1a4953ec, eval_metrics.py
c5b94117), the same preflight (afa0f3f0) and the same train_mb.py arguments, on data copied from clusterA
and verified byte-identical by SHA-256 (`~/bci_bpA_stage/manifest.json`). Story 1 (secondary) stays on
clusterA only.

(b) **Selection rule, fixed now; value-blind** (`scripts/calmatch_assemble_20260925.py`, which reads
only file existence and receipts, never result values). A subject is complete on a cluster when its
result and receipt exist, the receipt's `result_sha256` matches the result file, and the receipt was
written no later than the selection time and 2026-09-25 09:30.
- **Rule A:** if clusterA has all ten primary subjects complete, the primary analysis is clusterA ×10 and
  the clusterB results are descriptive only (cross-hardware replication; ELN and analysis receipt,
  not the paper).
- **Rule B:** otherwise, primary = clusterA for 01 02 03 05 + clusterB for 07 08 09 10 11 13; for a
  subject that clusterB has not completed by the cut-off, clusterA is used if clusterA completed it (the
  choice depends only on which file exists). Any clusterA result for a subject taken from clusterB is
  descriptive only. Any of the ten still missing → INCOMPLETE (§3), paper unchanged.
- **Selection time:** the earliest of (i) the moment clusterA has all ten (Rule A); (ii) 07:30, if
  clusterB has all six and clusterA does not have ten (Rule B); (iii) 09:30 (Rule A if clusterA has ten,
  else Rule B). Rule B is never evaluated before 07:30. The rule may not be changed after any of the
  six new results exists; `analysis.json` records `selection.json` (rule, per-subject cluster, SHAs).

(c) **Hardware.** L40S (Ada, sm_89) vs H200 (Hopper, sm_90); clusterB env built to match clusterA's
recorded versions (Python 3.11, torch 2.6.0+cu124, numpy 1.26). TF32 defaults are the same on both
(preflight records `matmul_allow_tf32` and `cudnn_allow_tf32`; a clusterB preflight whose flags,
PyTorch version or numpy major.minor differ from the clusterA preflights is treated as an engineering gate
failure for that subject). Kernel selection and reduction order differ between GPUs, so the
arithmetic is not bit-identical; this is pretraining-realisation noise of the class already in §1
(non-deterministic cuDNN), expected at the 1–2-point scale of the Table 1 seed rows at k ≥ 500. No
correction is applied; the estimator, bootstrap, margin and decision rule (§2, §5) are unchanged.
Under Rule B the appendix gains one sentence naming the subjects per cluster (pre-written in
`scripts/calmatch_apply_20260925.py`); the caption sentences of §7 are unchanged.

(d) **Layout-only integration rules** (added now, not result-dependent): the appendix table is set
`\footnotesize`, 3 pt column separation, scaled to the line width (at `\small` it overflows the
margin); to keep the main text at net zero lines, the Table 1 caption is trimmed by the smallest of
two pre-fixed trims — (1) "under the alternative contrasts" → "under the contrasts"; (2) also drop
", three pretraining stories either way" (restated in §3 of the paper) — that restores the pre-change
per-page line counts of pp. 1–9; if neither does, the edit is not made and this is reported.

## Deviation 3 — 2026-09-25T04:09:19 (code defect found after the verdict; rounding only)

After the primary analysis (analysis.json 97a0e4c7, verdict NO_MATERIAL_CHANGE, commit 5c9889c) an
independent recomputation from the result files found two printed CI bounds 0.1 off:
`summarise()` stored ρ, D and their CIs rounded to 2 decimals and the renderer rounded them again to
0.1. conv k=2000 ρ_match upper bound 54.6531 was stored as 54.65 and printed 54.6 (correct 54.7);
dynamic k=100 D lower bound −6.649 was stored −6.65 and printed −6.7 (correct −6.6). Every other
printed value, both primary D intervals and the verdict are unaffected.

Change: `calmatch_analysis_20260924.py` now stores full-precision floats (means, ρ, D, CIs, MB-1n
rows); the renderer is the only place that rounds. The estimator, bootstrap (same draws), margin and
decision rule are unchanged. The analysis is rerun once on the same selected inputs
(results/calmatch_20260924/primary, Rule A) and replaces analysis.json; the 2-decimal receipt stays
in git history (5c9889c).

Presentation note (no number changes): the MB-1n SEEN replay files are identical to the published
SEEN files (all five subjects, both encoders), so its descriptive row is 100.0% with a zero-width
interval. The appendix sentence says so explicitly, so the row is not read as an error.
