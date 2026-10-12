# Release inventory — what ships with the paper, and what each piece needs

This file names every piece of the public release in one place. `bash verify_release.sh` runs every
check that works off-cluster (see README.md).

## 1. The two audit tools (stdlib-only; run on any machine)

- `scripts/stimulus_leakage_check.py` — CI-style split checker: exits 1 when held-out *subjects*
  still share *stimuli* with training. `--propose-split` emits a stimulus-disjoint split proposal
  (partitions stimuli only; combine with a subject holdout — the emitted `rule` says so).
- `scripts/leakage_report.py` — corpus exposure report; `--demo meg-masc` reproduces the paper's
  release-section numbers (4/4 exposure, 25% fix cost, the three-regime table).

Both are verified to run on a stock Python 3.9 with no third-party imports.

## 2. The corrected MEG-MASC split

- `artifacts/meg_masc_stimulus_disjoint_split.json` — the split itself (self-describing `rule`).
- `artifacts/manifest_*.csv` — subject×stimulus manifests for the corpora in the release table.
  NOTE: these manifests carry no `split` column; to audit a *split* (rather than a corpus), add
  `split` labels for your protocol first — `stimulus_leakage_check.py` aborts loudly otherwise.
- Regeneration is deterministic: `--propose-split --out` on a split-annotated manifest reproduces
  the JSON byte-identically.

## 3. The brainmagick patches (`patches/`)

The patch scripts quote short anchor strings from facebookresearch/brainmagick (Défossez et al., 2023),
which is released under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/); the patches modify
that code and are distributed under the same terms, for non-commercial research use.

Env-gated edits to a `facebookresearch/brainmagick` checkout. **Apply ALL twelve, in this order**
(the order they were applied to the checkout used for every reported arm). Later patches anchor on
text earlier ones injected, and the early story-split iterations are *wrong on their own* (their
headers say so; `bm_story_split5.py` documents the final fix):

```
bm_story_split.py -> bm_story_split2.py -> bm_story_split3.py -> bm_story_split4.py -> bm_story_split5.py
  -> bm_train_subsample.py -> bm_wer_pool_match.py -> bm_mixed_split.py
  -> bm_wer_pool_seed.py -> bm_wer_dataset_order.py -> bm_evalstory_frac.py -> bm_mixed_story_choice.py
```

Target checkout: every patch requires `BM_CHECKOUT=/path/to/brainmagick` and exits with a message
if it is unset. Every patch asserts its anchors and writes a `.orig*` backup; a failed anchor means
either wrong order or an upstream drift -- nothing is patched silently. With every env var unset the
patched pipeline behaves as shipped.

Which arm uses which patch (env vars as set in `scripts/slurm/repro_*.sbatch`):

| arm | patches it relies on | env vars |
|---|---|---|
| A, block split as shipped | `bm_wer_pool_match` | `BM_WER_MAX_SEGMENTS=7140` |
| B, block split thinned to C | `bm_train_subsample`, `bm_wer_pool_match` | `BM_TRAIN_FRAC=0.5824 BM_TRAIN_FRAC_SEED=0` |
| C, story split | `bm_story_split` 1-5, `bm_wer_pool_match` | `BM_STORY_SPLIT=1 BM_STORY_SPLIT_SEED=9` |
| C' / D, one-bit arms | `bm_mixed_split`, `bm_train_subsample`, `bm_wer_pool_match` | `BM_MIXED_SPLIT=Cprime\|D`, `BM_TRAIN_FRAC=0.5657` (D) |
| rescore, D realizations | + `bm_wer_pool_seed`, `bm_wer_dataset_order` | `BM_WER_POOL_SEED=0`, `BM_TRAIN_FRAC_SEED=1\|2` |
| dose ladder | + `bm_evalstory_frac` | `BM_TRAIN_FRAC_EVALSTORY` |
| second narrative | + `bm_mixed_story_choice` | `BM_MIXED_EVAL_STORY=easy_money` |
| word-onset clock | same as the arm it shadows, plus a hand-inserted block in `bm/dataset.py` `__getitem__` (see below) | `BM_TIMING_ONLY=1` |

The clock is the one pipeline edit that was not applied by a patch script: the `BM_TIMING_ONLY`
block was inserted by hand into `bm/dataset.py` `__getitem__`, where `meg_torch` is built. Its exact
text is the `old` string of `scripts/brainmagick_patches/patch_modes.py`, which anchors on it (that
script's extra count/shuffle modes are not used by any arm in the paper).

## 4. The protocol note, the model code and the pre-specification record

- `LEAKAGE.md` -- the protocol note (kept in sync with the paper by a pin-based checker).
- `src/` -- the internal model code: `train_mb.py` (every internal result), `tokenizers.py` (the two
  encoders), `train_infonce.py` / `train_m1a_seg.py` (target head, data), `eval_metrics.py` (guarded
  retrieval), `extract_contextual.py` (GPT-2 layer-6 targets). They need the MEG-MASC data, PyTorch
  and a GPU; the launch configurations are `scripts/slurm/mb1*.sbatch`.
  These are the versions that produced the reported numbers: `train_mb.py` SHA-256 `1a4953ec...`
  and `eval_metrics.py` SHA-256 `c5b94117...` (before identity scrubbing, which touches comments
  only); `train_infonce.py`, `train_m1a_seg.py`, `tokenizers.py` are unchanged since every reported
  run. The later live `eval_metrics.py` had an uninitialised counter, fixed on 2026-09-24; the fix
  adds a degenerate-input guard and leaves every score unchanged.
- Subject count. Our copy of MEG-MASC holds 10 of the corpus's 27 subjects (one recorded in a single
  session), so every pretraining set is the other nine. Two files shipped as written say "26": a
  comment in `src/train_mb.py` ("from the other 26 subjects") and the arm-table header of
  `notes/PREREG_calibmatch_20260924.md` ("other 26 subjects"). Both should read nine; neither enters
  any computation (the shipped per-run receipts print the pretraining and held-out segment counts).
- `prespecification/` -- excerpts, verbatim apart from replacing cluster paths, account and host names
  with placeholders (USER, CLUSTER_ACCOUNT, $SCRATCH): the program
  document's hypothesis and decision gate (committed 2026-07-08), the decision record for the
  stimulus-scoped guard (2026-07-29), and the lab-notebook entries that fix each criterion or
  reading rule before its result (pre-registration of the guard re-run, the criterion applied to the
  deciding sweep, and the launch entries of the external follow-up arms). Pre-registered readings of
  the external arms are also in the headers of their `scripts/slurm/repro_*.sbatch`. The full
  notebook is not shipped.
- `figures/*.values.json` and `scripts/fig_body_20260924.py` -- the values plotted in Figures 2-3 and
  the script that draws them.
- Field names in the per-run result JSONs (`result_*.json`, e.g. under `results/calmatch_20260924/`):
  `x` is top-10 accuracy as a multiple of chance; `ci` is the query-bootstrap 95% interval of top-10
  accuracy **in percent**, not an interval for `x`, so the two are on different scales (e.g. `x`=2.45
  with `ci`=[2.96, 6.67]). `meanpct` is the rank percentile r of the paper (3 decimals; ties count as
  below the target), `pool` the mean candidate-pool size, `top1x` top-1 accuracy as a multiple of chance.
- `results/strengthen_20260926/` (revision, added after submission) -- the pre-registered
  pretraining-seed replicates of the paper's appendix "Pretraining-seed replicates of the factorial":
  `analysis_summary.json` (the analysis receipt with its `run_records` block, which carries cluster host
  names, removed), `appendix_table.tex` (the table as `\input` by the paper), `branch_numbers.txt` (the
  registered branch per question), `k2000_threeseed_20261004.json` (the k=2000 reading) and
  `guard_fixed_model_20261007.json` (recording vs stimulus guard on the same trained model, 232 pairs).
  The per-run JSONs and training logs behind them are not shipped; the summaries recompute nothing
  off-cluster. The two pre-registrations ship as `notes/PREREG_strengthen_20260926.md` (two header
  lines naming internal review files replaced by a marker) and `notes/PREREG_k2000_threeseed_20261004.md`.
  The paper prints SHA-256 `9be5e459...` for the full `analysis.json`; `analysis_summary.json` is that file
  minus `run_records`, so it hashes differently. `scripts/strengthen_render_20260926.py` regenerates
  `appendix_table.tex` and `branch_numbers.txt` byte-identically from `analysis_summary.json`.

## 4b. Added for the public release (R1 text, 2026-10-12)

- `results/subjects27_20261004/stage1/stage1_analysis.json` (SHA-256 `a211d1c5...`, byte-identical to the
  receipt the paper cites) -- the rebuilt-archive bridge: the ten subjects re-epoched from the raw
  MEG-MASC release, three seeds. Pre-registration `notes/PREREG_subjects27_rebuttal_20261004.md`
  (three lines naming internal review files and one internal path replaced by markers; "torch", the
  cluster's name, replaced by "the cluster"). Generator `scripts/subjects27_stage1_analysis.py`; it reads
  the 30 per-run JSONs of the rebuilt archive and the earlier per-run JSONs, which are not shipped.
- `results/table1_receipts/` -- the per-subject result JSONs of every Table 1 arm (also read by
  Figure 1's withheld-story rows). `python3 scripts/retained_fraction.py --check-published --results
  results/table1_receipts` reproduces all 26 published Table 1 entries off-cluster.
- Figure 1: `scripts/make_fig1_v4.py` (with `scripts/fig1_rc.py`, `scripts/figstyle/`) draws it from
  `figures/fig_factorial_subjects.values.json`, `results/strengthen_20260926/analysis_summary.json`,
  `stage1_analysis.json` and `results/table1_receipts/`; `figures/fig1_v4/` holds the drawn figure and
  the plotted values. The script asserts its numbers against the paper source: copy the arXiv source
  `paper_iclr.tex` to the repository root first. It needs matplotlib, svgutils and Google Chrome
  (headless, for SVG->PDF).

## What does NOT regenerate off-cluster

The per-table regeneration scripts (`scripts/retained_fraction.py`, `interaction.py`,
`grid_pct.py`, `dose_pinned.py`, `abc_decomposition.py`, …) read result JSONs and run logs that
live on the training cluster and are too large to ship. Off-cluster they now **fail loudly**
(exit nonzero with a message) instead of printing an empty table. Point them at your own results
directory with `BCI_RESULTS=/path/to/results`.

Which script generates which table (all under `scripts/`):

| paper | generator |
|---|---|
| Table 1 (retained fraction, alternative contrasts) | `retained_fraction.py` (conditions via `mb_conditions.py`); runs off-cluster on `results/table1_receipts/` with `--check-published` |
| Table 2 (guard x split factorial) | `interaction.py`; receipt twin `interaction_receipt.py` -> `results/interaction_receipt.json` |
| Table 3 (four withheld stimuli) | `grid_pct.py`; Bonferroni check `reversal_inference.py` |
| Table 4 (reference pipeline A/B/C, one-bit, dose ladder) | `abc_decomposition.py`; off-cluster check `verify_onebit_readouts.py` |
| Appendix C sensitivity | `interaction_sensitivity.py` (runs off-cluster) |
| Appendix D dose table | `dose_pinned.py` |
| Figures 2-3 | `fig_body_20260924.py` from `results/` and `figures/*.values.json` |
| Appendix H.1 (calibration-diversity control) | `calmatch_analysis_20260924.py` on `results/calmatch_20260924/primary/` (runs off-cluster; pre-registration `notes/PREREG_calibmatch_20260924.md`, see `results/calmatch_20260924/README.md`) |
| Appendix H.2 (pretraining-seed replicates) | `strengthen_analysis_20260926.py` -> `analysis.json` and `k2000_threeseed_20261004.py` (not shipped: they read per-run JSONs that stay on the cluster; their summary receipts ship under `results/strengthen_20260926/`); table by `strengthen_render_20260926.py` (shipped, runs off-cluster on `analysis_summary.json`) |
| Appendix H.2, rebuilt archive | `subjects27_stage1_analysis.py` -> `results/subjects27_20261004/stage1/stage1_analysis.json` (needs per-run JSONs, not shipped) |
| Figure 1 | `make_fig1_v4.py` (see section 4b) |
