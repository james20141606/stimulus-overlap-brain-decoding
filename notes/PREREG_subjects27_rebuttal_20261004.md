# PREREG — 27-subject extension of the factorial's key cells (#60432, rebuttal-only), 2026-10-04

Written 2026-10-04 03:15–03:20 +08:00 and committed before any code for it exists and before any job.
**No code is written, nothing is submitted, 0 GPU.** the cluster is unreachable tonight; nothing in this file
was run. The submitted PDF (sha256 `7167259b08ad…`, source
`BCI_foundation@95f4419`) is frozen until reviews (2026-10-29). Results from this plan go only to the
rebuttal / discussion-period revision, never into the submitted PDF.

Trigger: the author asked on 2026-10-04 「27人可以都用吗」 ("can we use all 27?"). Short answer this file
gives: the data exist (`meg_masc_merged27`) but no run on them exists, the per-run cost is not measured,
and the archive is not a drop-in copy of the one the paper used (§6). This plan adds a measured pilot and
a bridge on the original ten. Each stage ends in a hard stop.

Sources read before writing (§0 lists what was *seen*):
[three internal review files, not released],
`notes/PREREG_strengthen_20260926.md`,
`notes/PREREG_k2000_threeseed_20261004.md`, the submitted tex (§3 Problem Setup, Eq. (1)–(3), Prop. 1,
Tables 2–3, §4), `scripts/strengthen_20260926/src_pinned/{train_mb,train_m1a_seg,train_infonce,tokenizers,eval_metrics}.py`,
`scripts/slurm/{mb1f_ksweep,mb1k_volumectl,mb1q_j0ext,strengthen_20260926}.sbatch`, and the MEG project ELN
(not released) entries 2026-07-28T03:40 (participants.tsv), 07-28 re-epoch validation,
07-28T13:47/13:52, 07-29T03:12/17:34, 07-30T01:27 (merge), 07-30T01:46 (artifact floor), and
`MEG/codes/megmasc_preprocess_all.py`.

---

## 0. What I have already seen (discount accordingly)

- Every published number in Tables 2–3 and the 2026-09-29/10-04 three-seed results on the ten
  (standard cell k=100 seed means +0.043 / +0.055 / +0.047, three-seed mean +0.048 [+0.024,+0.071];
  k=500 +0.018 [+0.005,+0.029]; Table 3 story-0 row −0.012 / −0.006 / −0.016*; the published factorial
  SS cell = SEEN arm +0.020 / −0.006).
- **Levels**, read on 2026-10-04 03:10 to set the competence floor in §2 (from
  `results/strengthen_20260926/inputs/`): published standard cell (`mb1f`) per-subject r at k=500, conv
  min 0.795 / mean 0.861, dynamic_big min 0.803 / mean 0.874; SEEN arm (`mb1k`) at k=500, conv min 0.833,
  dynamic_big min 0.812; at k=100 the lowest per-subject r in either cell is 0.680.
- **Nothing from `meg_masc_merged27` in BCI format has been seen.** No BCI model has ever been trained on it.
  What the MEG project recorded about it is in §6.
- Code facts that shaped the design (read, not run): the input layer of both encoders is
  `nn.BatchNorm1d(208)` at the default `eps=1e-5` (`tokenizers.py:25,79`; `train_infonce.py:57`); the
  loader passes the archive values unscaled (`train_infonce.load_meg_window`); `train_mb.py` reads
  `subject/session/task/run` as strings and selects the held-out subject by string equality with
  `--test_subject` (`train_mb.py:210–221`); `megmasc_preprocess_all.py:111` writes `subject`, `session`,
  `task` as integers.

## 1. Questions

- **Q1 (replication).** On all 27 MEG-MASC subjects, each held out in turn with pretraining on the
  other 26, is the dynamic encoder's standard-cell advantage at k=100 resolved positive, averaged over
  pretraining seeds 0–2?
- **Q2 (vanishing).** On the same 27, with the evaluated story withheld from pretraining (the Table 3
  story-0 row's configuration), is the advantage still not resolved in the dynamic encoder's favour at
  k=100 and at k=500?
- **Q3 (bridge).** On the original ten subjects with the original roster (pretraining on the other nine),
  does the rebuilt archive reproduce the sign and resolution of the published standard-cell advantage at
  k=100, averaged over seeds 0–2? And is it numerically consistent with the old-archive three-seed result?

Q3 is answered first (Stage 1). Q1 and Q2 are only run if Q3 passes (Stage 2).

## 2. Design: three stages, hard stop after each

Common to every training unit (one unit = one held-out subject × one cell × one seed):
code = a new pinned bundle `scripts/subjects27_20261004/src_pinned/` = the 09-26 strengthen bundle
(`train_mb.py` sha256 `ba615a25…`, `eval_metrics.py` `c5b94117…`) **plus one default-preserving flag
`--input_gain G`** that multiplies the built MEG segments (`Xtr`, `Xhs`) by G right after
`build_segments_with_meta`. With G=1.0 the code path is unchanged. Flags (identical to strengthen):
`--variants conv,dynamic_big --layer 6 --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000
--pre_epochs 80 --adapt_epochs 120 --seed {0,1,2} --split_seed 0 --adj_s 10 --adj_list 5,10,20`.
Backend is the historical one (no `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG` unset; the strengthen
preflight refuses the slow backend). Two units share one GPU (second starts 300 s later), as in strengthen.

Cells (flags as in the published runs):

| cell | flags | published counterpart |
|---|---|---|
| RR standard | `--guard_scope recording --split_by recording` | `mb1f_ksweep` (seed 0), `st26_rr_seed{1,2}` |
| UNSEEN-0 (Q2) | `--holdout_story 0 --guard_scope stimulus` (pretrain {1,2,3}, calibrate {1,2,3}, evaluate 0) | Table 3 story-0 row, `mb1j_story0` |
| SEEN (factorial SS cell) | `--holdout_story 3 --eval_story 0 --guard_scope stimulus` (pretrain {0,1,2}, calibrate {1,2}, evaluate 0) | `mb1k_volctl` |
| SSc clean (optional 2D) | `--guard_scope stimulus --split_by story` (four-story pretraining) | `st26_ssc_seed{0,1,2}` |

Archive views, written in Stage 0a (no copy of the 84 GB array; `epochs_data.npy` is a symlink):
`merged27_bci/` (all 27, metadata in the old archive's string format) and `merged27_bci_ten/` (metadata
restricted to the ten; `trial_index` keeps the global row numbers, so the same array and the same target
file serve both). Roster is controlled only by which metadata file is passed; `train_mb.py` pretrains on
every non-held-out subject in the metadata.

### Stage 0 — pilot. Purpose: measure cost; verify the input and target build. No estimand.

**Stage 0a (CPU job, no GPU; `scripts/slurm/subjects27_stage0a_cpu.sbatch`).** Every check writes to one
`stage0a_checks.json`. Any FAIL stops the plan before any GPU job.

- A1 Integrity: `epochs_data.npy` shape (403,368, 208, 250) float32; `trial_index == arange(N)`;
  27 subjects; 196 runs; runs per subject 8, except 4 for sub-03/12/16/20/21 (participants.tsv).
- A2 Format: build the two BCI views; `subject` zero-padded `%02d`; `session/task/run` written in the
  literal string form the old archive uses (read from the old `trial_metadata.tsv`, not assumed);
  `task` is the story id 0–3 with the same numbering as the old archive for every shared
  (subject, session) — checked by equal word sequences.
- A3 Trial selection on the ten: for each shared (subject, session, task), the word list (word_index,
  onset_s to 1 ms) equals the old archive's, row for row. (MEG checked sub-01 only.)
- A4 Gain: per shared subject, `std_old / std_new` over matched epochs. G = median over the ten.
  PASS only if max/min of the ten ratios ≤ 1.01 and, after scaling, median per-epoch r ≥ 0.999 on a
  600-epoch sample of each shared subject. G is written to the json and frozen; Stage 1 and Stage 2 use
  it unchanged. (MEG fitted 100163.23 on one run of sub-01; that number is a cross-check, not G.)
- A5 Instrument check at the input layer (planted): with seed 0 initialisation, the same 256 word
  segments from (i) the old archive, (ii) merged27 × G and (iii) merged27 unscaled go through each
  encoder's `in_bn` + spatial conv in float32 (train-mode BN). PASS if (i) vs (ii) outputs correlate
  ≥ 0.99. (iii) is recorded, not gated; it documents what an unscaled run would feed the network.
- A6 Targets: run BCI's `src/extract_contextual.py` (live sha256 `a9945966…`, layers 4 6 8 10 12,
  fp16, aligned to `trial_index`) on `merged27_bci/`, offline (`HF_HUB_OFFLINE=1`; if GPT-2 is not in
  the HF cache, STOP — no online fallback on shared scratch). Compare layer 6 with the published
  `results/m0/masc_gpt2_contextual.npy` (sha256 `ad1e9d83…`) on every matched word of the ten. PASS if
  cosine ≥ 0.999 on every matched row and the meta `layers_kept` is identical. The MEG project's
  GPT-2 file for merged27 (job 15149712) is compared descriptively only and is not used.
- A7 Counts without loading MEG: reproduce the segment binning of `build_segments_with_meta`
  (`floor(onset/1.0)` bins per recording, ≥1 word) from metadata alone, for every (cell, held-out
  subject, roster): pretraining segments N_pre, held-subject segments, calibration pool, evaluation set,
  and which k ∈ {100,500,2000} each subject has (k ≤ calibration pool − 300). For the ten under the
  ten-roster, old-archive and merged27 counts must be equal.
- A8 Storage: `myquota` before writing; the target file is 403,368 × 5 × 768 × 2 B ≈ 3.1 GB (shape
  arithmetic). Nothing else large is written.

**Stage 0b (one GPU, two units concurrently; `scripts/slurm/subjects27_stage0b_pilot.sbatch`).**
1 held-out subject (sub-01) × 1 cell (RR standard) × 1 seed (0), on both rosters:
unit A = 27-roster (pretraining on 26; the Stage 2 shape), unit B = ten-roster (pretraining on 9;
the Stage 1 shape). Both use `--input_gain G`. Measured and written to a Stage 0 report:

- M1 wall-clock per unit, phase-resolved from a timestamped log: segment build, pretraining per variant
  (seconds per epoch), adaptation per variant per k (FM + scratch), evaluation;
- M2 GPU utilisation every 15 s (job mean, min, and per phase — the build phase is CPU-only);
- M3 peak host memory per unit (`/usr/bin/time -v` MaxRSS) and peak GPU memory;
- M4 printed `train_seg`, `heldsubj_seg`, `calib_pool`, `eval_seg`, pool sizes; must equal A7's counts;
- M5 GPU-h of the job (1 GPU × job elapsed, from `sacct`, not from the job's own timer).

Stage 0b gates: both units exit 0 with receipts; M4 equals A7; **competence floor**: both arms in both
units have r ≥ 0.60 at k=500 (published per-subject minimum in these cells is ≥ 0.795, §0; 0.5 is
chance). A failed floor means the instrument is broken (scale, targets or format), not that the
architectures are equal: STOP, diagnose, rerun Stage 0. The per-arm r values are seen (one subject);
the Δ for sub-01 is not analysed, and no rule below may change after Stage 0.
Reuse: unit B is Stage 1's (sub-01, seed 0) unit and unit A is Stage 2's (sub-01, RR, seed 0) unit,
**only if** the bundle SHA256SUMS, G, the metadata views and the target file are byte-identical when
Stage 1/2 run; otherwise they are rerun.

If the cluster cancels Stage 0b for utilisation below 50%, that is a measurement: record the per-phase
utilisation, and before any Stage 1 job move the segment build to a CPU step (a cached-segment
variant, its own prereg deviation) or pair units differently. Do not resubmit unchanged.

**HARD STOP 0.** Post the Stage 0 report (A1–A8, M1–M5, the Stage 1/2 projections of §5) to the author.
Stage 1 is not submitted without the author's go (decision point D1).

### Stage 1 — bridge (Q3). 10 subjects × RR × seeds 0–2 = 30 units (29 new if unit B is reused).

Ten-roster, `merged27_bci_ten/`, gain G. Same subjects, flags and `--split_seed 0` as the strengthen RR
runs, so seeds 1 and 2 are realisation-matched to `st26_rr_seed{1,2}` (same RNG stream if A7's counts
are equal; only the archive differs). Seed 0 is **not** realisation-matched to the published `mb1f`:
its klist (100,200,500,1000,2000) and extra noflow arm change the RNG stream (strengthen run_record), so
the seed-0 comparison carries training-realisation noise of the size Table 2 shows between its SR and
RR cells (+0.034 vs +0.043).

**No-go rule, Stage 1 → Stage 2 (fixed now).** Stage 2 is run only if Stage 1's Q3 primary label is
**PASS**: the seed-averaged k=100 Δ̄ has a 95% interval entirely above zero **and** all three seed means
are > 0 (the strengthen Q2 SUPPORT rule). Any other label (FAIL_UNRESOLVED, FAIL_SIGN, REVERSED; §4)
means **do not run Stage 2**; Stage 1 is reported as a preprocessing-sensitivity finding with the §4
wording. The author cannot waive this rule after seeing Stage 1; a waiver would need a new prereg
written before Stage 1 is analysed. A DRIFT label on the bridge difference (§3) does not block Stage 2
but changes how Stage 2 may be compared (§4).

**HARD STOP 1.** Report Stage 1. Stage 2 is not submitted without the author's go (D2).

### Stage 2 — full (Q1, Q2). 27 subjects × {RR, UNSEEN-0, SEEN} × seeds 0–2 = 243 units (242 new).

27-roster, `merged27_bci/`, gain G, no subject excluded (the paper applied no exclusion; §6 for the
artifact-floor sensitivity). Block priority: 2A RR (Q1) → 2B UNSEEN-0 (Q2) → 2C SEEN (Prop. 1(iii)
difference-in-differences, paper diagonal) → optional 2D SSc (clean diagonal; +81 units; separately
approved). Within a block, all subjects at seed 0, then seed 1, then seed 2.

Completeness: a block counts only if every subject has a receipt passing the per-unit gate (rc 0; SHA
of bundle and inputs; args echo; radius-10 unrounded r within 0.0005 of the primary rounded r; printed
counts = A7). An incomplete block is reported as "not completed (x/27)" with **no point estimate**.
Cut-off: written into this file (Deviation entry) when the author approves Stage 2, before any Stage 2
job; default = 72 h before the end of the author-response window. One analysis run, at the cut-off or
when all approved blocks are complete, whichever is first. No interim look at any Stage 2 Δ.

Stop-and-re-project rule: if the first completed Stage 2 task takes more than 1.5× its projected
wall-clock (§5), pause the array, re-project from the measured task, and ask the author before
continuing.

## 3. Estimands (as in the paper)

- **Score.** Rank percentile r = `fm.meanpct` from `eval_metrics.guarded_retrieval_metrics`, rounded to
  3 dp in the result JSON (the value every published Δ uses); the unrounded radius-10 value is reported
  as a check. **Tie handling: the code's convention is used.** The code computes
  `rank = 1 + #(s_c > s_q)` and `r_q = 1 − (rank−1)/(|Π_q|−1)`, so a candidate whose score equals the
  target's counts as *below* the target. Eq. (1) as printed uses a strict `<` and counts it as *not
  below*. All published numbers come from the code, so the code is the estimand here; the discrepancy is
  disclosed (kit §E-4). Exact ties come only from candidates with identical targets (same words in the
  same 1-s bin of another presentation); they hit both encoders identically and cancel in every Δ. Levels
  shift by an unmeasured amount. Single-session subjects have no such twins.
- **Advantage.** Δ_s(cell, k, seed) = r_s(dynamic_big) − r_s(conv), per held-out subject (the unit of
  inference). Seed-averaged Δ̄_s = mean over seeds 0–2 of the subject's Δ.
- **Interval and "resolved".** Paired subject bootstrap of the mean of Δ̄_s: 20,000 draws,
  `numpy.random.default_rng(0)`, percentile 2.5/97.5 — the draws of `mb_conditions.ci` (the strengthen
  analysis reproduced them bit for bit). "Resolved" = the 95% interval excludes zero (the paper's
  definition, §3 of the tex). Per-seed means are reported with their own intervals.
- **Budgets.** k=100 is primary for the standard-cell advantage (§4 of the tex: "the standard-protocol
  advantage rests on k=100"); k=500 is reported with its label and enters Q2's rule; k=2000 is
  descriptive (labels only). A subject enters a (cell, k) entry only if it has that k; a contrast uses
  the subjects that have that k in every cell it involves. n is printed for every entry.
- **Q3 primary (Stage 1).** Δ̄ at k=100, n=10, RR cell, merged27 ten-roster.
- **Q3 bridge difference (Stage 1, label only).** b_s = Δ̄_s(new) − Δ̄_s(old), where old = published
  `mb1f` seed 0 + `st26_rr_seed{1,2}` (same ten, same split). Label CONSISTENT (interval covers 0) or
  DRIFT (resolved). Also per seed for seeds 1–2 (realisation-matched). k=500 reported the same way.
- **Q1 (Stage 2).** Δ̄ at k=100, RR cell, n=27 (k=500 labelled).
- **Q2 (Stage 2).** Δ̄ at k=100 and k=500, UNSEEN-0 cell, n=27.
- **Prop. 1(iii) difference-in-differences (Stage 2).** Both arms evaluate story 0:
  D_s = Δ̄_s(SEEN) − Δ̄_s(UNSEEN-0), the estimate of δh_0 (exposure term, Prop. 1(iii)). Sign note: the
  ELN of 2026-08-03 printed UNSEEN − SEEN; this file uses SEEN − UNSEEN so that a positive value means
  pretraining exposure adds to the dynamic encoder's advantage. Labels at k=100 and k=500, unadjusted and
  Bonferroni over the two budgets (same draws at 1.25/98.75).
- **Diagonal (Stage 2).** The paper's diagonal, Eq. (2): Δ̄_s(SS) − Δ̄_s(RR) with SS = the SEEN arm
  (as in Table 2), labels at k=100 and k=500. The **clean-minus-standard diagonal** of the strengthen
  analysis, Δ̄_s(SSc) − Δ̄_s(RR), exists only if block 2D is approved and complete; otherwise it is
  reported as "not estimated".
- **Descriptive decomposition (Stage 2, labels only, no rule).** (a) Pretraining-roster effect on the
  original ten: Δ̄_s(RR, 27-roster) − Δ̄_s(RR, ten-roster from Stage 1), s ∈ ten (paired; same archive,
  same subjects; only the 17 added pretraining subjects differ). (b) Mean Δ̄ for the 17 added subjects vs
  the original ten (unpaired, no test). (c) Two-session (n=22) vs one-session (n=5) subjects in the RR
  cell (§6: for one-session subjects the recording split never puts one story on both sides).
  (d) Q1/Q2 labels recomputed without sub-03, sub-13, sub-19 (the MEG project's artifact floor).

## 4. Decision rules and wording (fixed now)

Numbers are filled in from the analysis JSON only. "Seed means a / b / c" are the per-seed means.

**Q3 (Stage 1), on Δ̄ at k=100:** PASS = interval > 0 and all three seed means > 0;
REVERSED = interval < 0; FAIL_SIGN = Δ̄ ≤ 0 and not REVERSED; FAIL_UNRESOLVED = Δ̄ > 0 but not PASS.
**Q1 (Stage 2):** REPLICATES = interval > 0 and all three seed means > 0; REVERSED = interval < 0;
WEAKEN = otherwise.
**Q2 (Stage 2):** VANISH_HOLDS = neither k=100 nor k=500 resolved positive; VANISH_FAILS = either
resolved positive. (Unadjusted intervals at two budgets make VANISH_FAILS easier to reach; that is the
conservative direction for the paper's claim.)

Branches. The why-10 sentence uses the fact base F, which is **AUTHOR TO CONFIRM** (kit §C):

> F: [AUTHOR TO CONFIRM] The ten subjects are the entire preprocessed MEG-MASC archive this project
> inherited from earlier work; we recorded its contents at our first data check, before any model in this
> paper was trained, and used all of it. The rule by which that earlier work chose the ten is not recorded.

**Branch INSTRUMENT** (a Stage 0 check or the competence floor fails and is not fixed in time):
> We began a 27-subject rerun on an archive rebuilt from the raw release, but it did not pass its own
> input checks before the discussion period ended, so we make no claim from it. F.

**Branch NOGO** (Stage 1 label ≠ PASS; Stage 2 not run):
> We rebuilt MEG-MASC from the raw release with new epoching code (identical trial selection; median
> per-epoch $r = 0.9999$ with the archive once its global scale is restored) and reran the standard
> protocol on the same ten subjects over three pretraining seeds. The $k{=}100$ advantage [is not resolved
> | is not positive | reverses]: $\bar\Delta = a$ $[l, u]$ (seed means $a_0$ / $a_1$ / $a_2$), against
> $+0.048$ $[+0.024,+0.071]$ on the original archive. The standard-protocol advantage is therefore
> sensitive to preprocessing differences below the level the paper describes, and we did not extend a
> comparison that does not survive a rebuild of its own archive to all 27 subjects. We will report this
> sensitivity next to Table 2.
> Why ten: F. A 27-subject run was planned but stopped by this bridge check.

**Stage 1 PASS sentence** (used inside every Stage 2 branch):
> On the original ten subjects, an archive rebuilt from the raw release (new epoching code, identical
> trial selection, global scale restored) reproduces the standard-protocol advantage at $k{=}100$ over
> three seeds ($\bar\Delta = a$ $[l, u]$; original archive $+0.048$ $[+0.024,+0.071]$; paired difference
> $d$ $[l_d, u_d]$).
If the bridge label is DRIFT, add: "The rebuild moves the estimate (paired difference resolved), so we
compare the 27-subject results only with the ten-subject results on the rebuilt archive, not with the
published numbers."

**Branch R — replicates** (Q1 REPLICATES, Q2 VANISH_HOLDS):
> On all 27 MEG-MASC subjects (each held out in turn, pretraining on the other 26, three pretraining
> seeds) the standard-protocol advantage at $k{=}100$ replicates ($\bar\Delta = a$ $[l, u]$, $n{=}27$;
> seed means …), and with the evaluated story withheld from pretraining it is not resolved at $k{=}100$
> ($b$ $[l, u]$) or $k{=}500$ ($c$ $[l, u]$). Unlike Table 3's story rows these cells have three seeds.
> Not resolved at $n{=}27$ is still not equality: the interval bounds the remaining advantage at $u$.
> These runs change subject sampling and pretraining volume together (26 instead of 9 pretraining
> subjects); on the original ten, pretraining on 26 moves the advantage by $e$ $[l, u]$ (descriptive).
> Why ten: F. Rerun on all 27, the two claims that rest on this comparison hold.

**Branch M — mixed: standard replicates, withholding does not remove it** (Q1 REPLICATES, Q2 VANISH_FAILS):
> On all 27 subjects the standard-protocol advantage replicates ($a$ $[l, u]$), but withholding the
> evaluated story from pretraining no longer removes it: the story-0 cell resolves in the dynamic
> encoder's favour at $k{=}\ldots$ ($b$ $[l, u]$, $n{=}27$, three seeds). The paper's statement that the
> advantage disappears once the story leaves pretraining (TL;DR, abstract, Table 3 title) is a ten-subject,
> single-seed result that does not hold at $n{=}27$; we will narrow it to a reduction [if DiD resolved
> positive] / to "not detected at $n{=}10$" [otherwise].
> Why ten: F. The standard-protocol advantage holds on all 27; its disappearance under story withholding
> does not, so that part of our claim depends on the ten.

**Branch N — does not replicate** (Q1 WEAKEN or REVERSED; Q2 reported, not interpreted):
> On all 27 subjects the standard-protocol advantage at $k{=}100$ is [not resolved | reversed]
> ($a$ $[l, u]$; seed means …), although the same rebuilt pipeline reproduces it on the original ten
> (Stage 1). The low-budget advantage in Table 2 is therefore a property of the ten-subject sample and/or
> its nine-subject pretraining corpus, not of MEG-MASC [descriptive split: on the original ten,
> pretraining on 26 instead of 9 moves it by $e$ $[l, u]$; the 17 added subjects average $f$]. With no
> standard-protocol advantage left at $n{=}27$, the story-withheld comparison has nothing to remove and we
> draw no inference from it. We will lead with the 27-subject result and narrow the abstract's first
> clause to the ten-subject sample.
> Why ten: F. On all 27 the comparison comes out differently, so the ten are not representative for it,
> whatever the reason they were chosen.

**Added to R and M, by DiD label (k=100 primary; k=500 reported):**
- resolved positive: "The difference-in-differences of Proposition 1(iii) now resolves ($D$ $[l, u]$):
  on the same story, keeping it in pretraining raises the dynamic encoder's advantage relative to
  withholding it — the paired test the ten-subject data could not resolve."
- not resolved: "The difference-in-differences of Proposition 1(iii) still does not resolve at $n{=}27$
  ($D$ $[l, u]$), so attributing the loss of the advantage to pretraining exposure remains a comparison of
  separately tested cells, not a paired result."
- resolved negative: "The difference-in-differences resolves in the other direction ($D$ $[l, u]$); we
  do not attribute the loss of the advantage to pretraining exposure."

**Added to every Stage 2 branch:** the paper-diagonal label; the clean-diagonal label or "not
estimated (block 2D not run)"; the descriptive (c) and (d) labels if either changes a Q1/Q2 label
("…; without the three subjects an artifact floor would exclude, Q1 is …").

**What no branch may say:** that the ten were not outcome-selected (only "fixed before any model in this
paper"); that a non-resolved interval shows equal architectures; that the 27-subject result tests the
OSF "Part 1 only" explanation of the ten (it does not fit the roster, §6); anything about the RS/SR cells
or the guard (not rerun here).

**Camera-ready (only if the paper is accepted; not this PDF):** R → one sentence + appendix table;
M → narrow TL;DR, abstract and Table 3 title; N → abstract's first clause restricted to the ten-subject
sample with the 27-subject row added to Table 2. Every edit goes with its `check_paper_numbers.py` pin.

## 5. Compute

**Per-unit cost: unknown — Stage 0 measures it.** No GPU-h figure is given here. Stage 1/2 are projected
from Stage 0's measurements only, as follows (all inputs are Stage 0 outputs):

- From unit A's timestamped log (27-roster, RR, sub-01): pretraining seconds per epoch per segment for
  each variant, c_v = (pretraining wall-clock of v) / (80 × N_pre,A); adaptation wall-clock per variant
  T_adapt,v (3 budgets × FM + scratch; roster-independent to first order); build time per loaded word
  window, t_w, from units A and B (two points).
- For every planned unit u, N_pre,u and the number of loaded word windows W_u come from A7 (exact
  counts, no GPU). Projected unit wall-clock:
  T_u = t_w · W_u + Σ_v (80 · c_v · N_pre,u + T_adapt,v).
- Two units share a GPU, so task wall-clock = max(T_u1, T_u2) × (1 + s), where s is the contention
  factor measured in Stage 0b: unit A's seconds per pretraining epoch while unit B was running, divided by
  the same after unit B finished, minus 1 (both are in A's log).
- GPU-h(stage) = Σ_tasks task wall-clock × 1 GPU. Reported separately for Stage 1 (15 tasks),
  Stage 2 blocks 2A, 2B, 2C (each ⌈81/2⌉ = 41 tasks before reuse), and optional 2D.
- Wall-clock floor = GPU-h / 2 (per-user cap of 2 concurrent GPUs); queue time is not predictable and
  is not projected.
- Memory: whether two 27-roster units fit one node comes from unit A's MaxRSS (M3) and the node limit;
  if not, Stage 2 runs one unit per GPU and the utilisation plan changes (decision point).
- Consistency check (not a projection): Stage 1 has the shape of the 09-26 strengthen block B, whose
  per-task elapsed times are in its receipts (`elapsed_s_this_invocation`). If Stage 0's unit B differs
  from those by more than 2×, find out why before Stage 1.
- The paper's compute statement (41 GPU-h, App.) is not used for any projection (audit M5: aggregation
  not recorded, stale).

Stage 0 itself: CPU job ≤ 6 h walltime; GPU job 1 GPU with a 4 h walltime **cap** (a ceiling, not an
estimate; the units checkpoint per unit and the job cannot exceed 4 GPU-h).

## 6. Disclosures that go with any result from this plan

1. **The archive is rebuilt, not the published one.** `meg_masc_merged27` was produced by the MEG
   project with newly written epoching code from the raw OSF release (2026-07-28/30; JID 14991342 for
   the 27 per-subject shards, then a merge): 403,368 epochs over 196 runs, 2,058.0 per run, as in the
   old archive (156,408 / 76). On one run of sub-01 it matches the old archive's trial selection exactly
   and its signal at median per-epoch r = 0.9999 (min 0.998, n = 654) **after a global gain of about
   1.0016×10⁵** (old = new × gain); residual after gain ≈ 0.8% of signal SD; not bit-exact.
2. **Neither archive is z-scored.** Only the no-z-score variant reproduces the old archive (z-scored
   variants: r 0.72–0.96); the old `metadata.json` flag `zscore_normalized: true` is false. Appendix I's
   "baseline-corrected and z-scored" (`appendix_models.tex:18–20`) is wrong for the published runs too.
3. **The gain is not absorbed by the encoders.** Both encoders start with `BatchNorm1d` at `eps=1e-5`.
   The old archive's SD is about 2.7×10⁻⁸ (variance ≈ 7×10⁻¹⁶ ≪ eps), so that layer does not normalise
   the input; it multiplies it by about 1/√eps ≈ 316. merged27 is about 10⁵ smaller again. An unscaled
   merged27 run would therefore feed the network a signal ~10⁵ smaller relative to every bias term —
   a different instrument, not a preprocessing variant. This plan multiplies by the single constant G
   fitted in Stage 0a and checks it at the input layer (A4, A5). (This also means the kit's paste
   sentence E-1, "batch normalisation … absorbs a global scale, so we expect no change to any Δ", is not
   supported for the published runs and should not be used as written.)
4. **Targets are rebuilt** with this repo's `extract_contextual.py` and checked against the published
   target file on the ten (A6). If A6 fails, nothing runs.
5. **Roster facts.** MEG-MASC has 27 subjects: 22 with two sessions, 5 with one (sub-03, 12, 16, 20, 21;
   participants.tsv; 196 recordings fetched, as expected). The published ten are sub-01, 02, 03, 05, 07,
   08, 09, 10, 11, 13 (BCI ELN 2026-07-07T07:55; "10 subjects (NOT 27)" 07-08). The release is split
   across OSF projects: ag3kj = sub-01–11, h2tzn = sub-12–23, u5327 = sub-24–27. **sub-04 and sub-06 are
   in the first project but not among the ten, while sub-13 (second project) is.** So "only the first
   OSF project was pulled" (MEG ELN 07-28, worded "plausibly") does not explain the ten and is not used.
   On participants.tsv the ten differ from the seventeen only in age before correction (28.0 vs 22.6 y,
   MWU p = 0.042; Holm 0.125); all ten are native English speakers; the one non-native speaker is sub-21.
6. **The published ten were the inherited archive — AUTHOR TO CONFIRM** (sentence F, §4). Until the
   author confirms, F is not pasted anywhere.
7. **Single-session subjects change the standard cell.** With one session, the recording split puts each
   story on one side only, so the "standard" cell is story-disjoint for 5/27 subjects (1/10 in the
   published sample). Reported as descriptive (c).
8. **No exclusion is applied.** The MEG project's artifact floor (median + 3 MAD = 41.3% artifact rate)
   would drop sub-03 and sub-13 (both in the ten) and sub-19; the floor's n=24 sample is not a superset of
   the ten. Primary = all 27; the floor is a descriptive sensitivity (d).
9. **Stage 2 changes two things at once**: who is evaluated (27 vs 10) and how much pretraining data each
   model sees (26 vs 9 subjects, the exact segment ratio from A7). Descriptive (a) separates them on the
   original ten only.
10. Calibration diversity differs between UNSEEN-0 ({1,2,3}) and SEEN ({1,2}) exactly as in the paper's
    published arms (App. calmatch addresses it on the ten; not redone here).

## 7. Author decision points

- **D0 — approve Stage 0** (CPU checks ≤ 6 h; one GPU job capped at 4 GPU-h; two units on sub-01).
  Preconditions before submit, all to be written *after* this commit and logged: the pinned bundle with
  `--input_gain` and its tests (G=1.0 byte-identical to the strengthen bundle on a planted small archive;
  planted archive scaled by 1/G and run with G equal to the unscaled run within 1e-3 in r); the Stage 0a
  check script with a planted-failure selftest for each of A1–A7; the analysis script with planted Δ for
  every branch in §4 and the bootstrap asserted equal to `mb_conditions.ci`. the cluster must be reachable
  (VPN → route → DNS → auth, in that order).
- **D1 — approve Stage 1** after the Stage 0 report (measured cost, projected Stage 1 GPU-h). Hard stop;
  the author may pre-approve in writing with a GPU-h ceiling.
- **D2 — approve Stage 2** (and separately block 2D) after seeing Stage 0's measured cost, the Stage 2
  projection, and Stage 1's result. Only possible if Stage 1 = PASS. The cut-off date is written here
  as a Deviation before any Stage 2 job.
- **D3 — confirm or correct sentence F** (why ten) against the author's own records. Independent of D0–D2.
- **D4 — whether any of this is used in the rebuttal at all.** The kit's current position ("not
  committed") stands until D2 is taken.

## Deviations

- **2026-10-05 04:54 +08:00, D0 recorded, before any job exists.** D0 (approve Stage 0 only) reached this session relayed by the
  coordinator ("Author approved D0 (2026-10-05, chat: 批)"). Stage 1/2 still need D1/D2. Deviations 1–8 below
  were written after reading the remote metadata/headers only (no check, no model run) and before submitting:
  1. merged27 stores **251** time samples (`metadata.json` n_times = 251, −0.2…0.8 s inclusive), the old archive
     250. A1 checks the array against its own metadata instead of (…, 250); `train_mb` reads samples [50:250)
     from both, so the window is the same if both start at −0.2 s. A4 records per-epoch r at offset 0 and at
     offset 1 and gates on offset 0 (the MEG re-epoch validation also compared the first 250 samples).
  2. GPT-2 is cached under `$SCRATCH/hf_cache/hub` (snapshot 607a30d…); `$SCRATCH/.cache/huggingface`
     holds only a lock. HF_HOME / HF_HUB_CACHE in both sbatch files point there (offline, one per line).
  3. `myquota` is not installed on the cluster and `quota` cannot read the /vast quota; A8 records filesystem free
     space and inodes (`df`) and gates on ≥ 50 GB free.
  4. A7 is first validated against the segment counts that published and strengthen runs *printed*
     (strengthen receipts' design lines for RR/SR/RS/SSc; `logs/mb/bci-mb1[kjqm]-*.out` for SEEN and
     UNSEEN-0), counted on the old archive with the published targets; any mismatch fails A7.
  5. The analysis script with planted branches (§7 D0 list) is deferred to before Stage 1: Stage 0 computes
     no estimand.
  6. A5's metric is made precise: signal = conv(bn(x)) − conv(bn(0)) at seed-0 init (train-mode BN, float32),
     because the conv bias dominates raw outputs. PASS = corr(signal_old, signal_G·new) ≥ 0.99 **and**
     SD ratio within 1%. The unscaled case is recorded (SD ratio, fraction of outputs equal to the zero-input
     response).
  7. The 0a job first runs the check script's planted-failure selftest and the bundle tests (T0–T2) on the
     the cluster env (pandas 3.0.2, unchanged since 2026-05-01), and stops if either fails. Locally (pandas 2.3.3):
     selftest 15/15 PASS; bundle T0/T1/T2 PASS (T2 max |Δr| 7.1e-6).
  8. Stage 0b's competence floor reads `fm.meanpct` at k=500 for conv and dynamic_big in both units.
- **2026-10-05T22:02:46+08:00, Stage 0a result: A3 FAIL → INSTRUMENT branch; D0 closed.** Job 19176514 (cpuA_partition, cl015,
  2026-10-04 17:00–17:03 EDT, sacct FAILED 1:0, 3 min 32 s) ran every registered step: the selftest passed
  15/15 and bundle tests T0–T2 passed on the cluster env (T2 max |Δr| 5.82e-6); A1 PASS (403,368 × 208 × 251,
  27 subjects, 196 runs, 2,058.0 epochs per run); A2 PASS; A8 PASS. **A3 FAIL**: 19 of 76 shared runs
  mismatch. They are all task 2 (easy_money), i.e. every recording of that story in the ten; the other 57 runs
  match row for row. Read-only diagnosis (login node, three runs): row counts equal (2,482), word_index,
  sequence_id and onset_s identical (max |Δonset| 0). In each of those runs exactly one word_text differs:
  the old archive has "None", the rebuild has an empty field (e.g. trial_index 4246, sub-01 ses-0,
  sequence 220, word 0, onset 1036.471 s). A4–A7 did not run because A3 gates them, so no G and no target
  comparison. Not a script or environment error: the check compared what it was registered to compare and
  found a real difference between the archives (the rebuild almost certainly lost the literal word "None" to
  NA parsing). Per the stop rule nothing else runs: 0b not submitted, 0 GPU. The check is not relaxed here.
  Any rerun (e.g. treating "None" and "" as the same word, which would also have to be checked against how
  the published targets encoded that word, A6) needs a new author decision and a new deviation, written
  before resubmission.
- **2026-10-05T22:07:02+08:00, Deviation 9 — registered diagnostic for the A3 failure; decision rule fixed before it runs.**
  The coordinator's default is to keep A3 as it is and run a diagnostic first; the author may override.
  Nothing below has been run when this is committed.
  1. *Listing.* For each of the 19 easy_money runs of the ten, list the single row whose word_text differs
     (old trial_index, merged27 trial_index, subject, session, sequence_id, word_index, onset_s, old and new
     word_text). Also count, over all 156,408 old rows, every word_text equal to "None" and every merged27
     row of the ten with an empty word_text. If those sets are not exactly the 19 listed rows, A3 stays
     FAILED.
  2. *Read-only CPU diagnostic on the cluster* (`scripts/subjects27_dev9_none_diag.py`; login node, GPT-2 small
     offline from $SCRATCH/hf_cache, seconds of CPU):
     (a) whether each row is in the published per-run trial selection (present in the old trial_metadata and
         in the published target file's filled rows) and whether its published layer-6 vector is zero;
     (b) recompute layer 6 for that row's whole sentence group exactly as `src/extract_contextual.py` builds
         it (sha256 a9945966…: group by subject/session/task/run/story/sequence_id, ordered by word_index,
         words joined by single spaces, last overlapping sub-token, fallback to the previous word's token),
         twice: encoding E_NA (word_text read as NA, so fillna gives "") and encoding E_None (the literal
         "None"). Compare with the published row by cosine. **Control (must pass, or no conclusion is
         drawn):** under the encoding that matches, *every* word of the group must match its published row
         (cos ≥ 0.999). Later words attend to this word, so the wrong encoding is expected to fail on them.
     (c) the word that the MEG-MASC source metadata carries at that position. The raw OSF events.tsv are no
         longer on scratch. The diagnostic reports whatever source copy it can find (the old archive's
         `stimuli_words.txt`, the MEG project's metadata), names the file it used, and writes "not
         available" otherwise.
  3. *Decision rule (fixed now).* Each of the 19 rows is classified as:
     - MISSING: present in the selection, and either its published vector is zero, or E_NA matches (cos ≥
       0.999 for that row and every word of its group) while E_None does not;
     - LITERAL: E_None matches by the same criterion while E_NA does not;
     - AMBIGUOUS: anything else, including a failed control.
     A3 may treat old-"None" ≡ new-"" **only if all 19 rows are MISSING**, and then only at those exact
     row pairs, through an explicit allowlist file written by the diagnostic. Every other comparison stays
     strict, and A6 still compares the rebuilt targets with the published ones. If any row is LITERAL, the
     two archives differ in content: A3 stays FAILED (INSTRUMENT, D0 closed), and the kit reports this as a
     reproducibility caveat of the published archive itself. If any row is AMBIGUOUS, A3 stays FAILED and the
     question goes back to the author. Only under all-MISSING is 0a resubmitted, with .
     0b waits for 0a PASS.
- **2026-10-05T22:14:25+08:00, Deviation 9 result: ALL_MISSING (19/19) → A3 uses the allowlist; 0a resubmitted.**
  Ran `scripts/subjects27_dev9_none_diag.py` (sha256 2ecef5c1…, commit a5e620b) read-only on the the cluster login
  node. Receipt: `results/subjects27_20261004/dev9/none_diag.json` (sha256 aec0a4bc…).
  (1) Listing OK. There are exactly 19 differing rows, one per easy_money recording of the ten (old
  trial_index 4246, 12478, …, 152422, i.e. every 8,232 rows; merged27 trial_index is identical). Each is
  sequence 220, word 0, onset 1036.471 s; old "None", new "". The old archive has 19 "None" words in total,
  and the ten-subject view has 19 empty words in total: the same rows.
  (2a) Every row is in the published selection and its published vector is non-zero.
  (2b) The published reader (pandas defaults + fillna) reads all 19 as "", and these are its only NA rows.
  E_NA reproduces the published layer-6 vectors for the row (cos 1.0000) **and for all 16 words of its group**
  (min 0.99999998). E_None does not: the row itself is at 0.99996 because it sits at position 0, where
  GPT-2's layer-6 states barely depend on the token, but the group minimum is 0.506. This is why the
  whole-group control was registered.
  (2c) The raw OSF events.tsv are no longer on scratch. The old archive's `stimuli_words.txt` carries the
  real English word: "… small backpack **None** said and …". So the published pipeline (and the rebuild,
  in the same way) turned the story word "None" into an empty word before GPT-2. That is one word in
  story 2, a reproducibility caveat of the published targets themselves, not a difference between the
  archives.
  (3) Under the rule fixed above all 19 rows are MISSING. A3 now takes `--a3_allowlist
  dev9/a3_allowlist.json` (sha256 102b1d42…, 19 exact pairs): only those pairs and only the text pair
  "None" → "" are accepted; any other difference, and any unused pair, still fails. Selftest extended with 4
  planted cases (19/19 PASS locally). 0a is resubmitted unchanged otherwise. A6 still compares the
  rebuilt targets with the published ones.
- **2026-10-05T22:29:04+08:00, Deviation 10 — 0a resubmission 19225034 failed on an environment error after A1–A4 and A8
  PASS; script-only fix.** With the allowlist, A3 passed (76/76 runs; 19/19 pairs used). A4 passed: G =
  100196.25, spread of the ten per-subject ratios 1.0012, per-subject median per-epoch r at offset 0 ≥ 0.99988
  (offset 1: 0.79–0.93, so the windows align at the start), residual after G 0.8–1.7% of the old SD. Raw SDs:
  old 2.0–5.1e-8, new 2.0–5.1e-13. Then `extract_contextual.py` crashed on import:
  `ImportError: cannot import name 'Tokenizer' from 'tokenizers' (…/src/tokenizers.py)`. Python puts the
  script's own directory first on sys.path, so running it from `src/` makes the repo's encoder module shadow
  HF `tokenizers`. Fix (sbatch only): copy the script byte-for-byte (sha256 checked in the job) to
  `$S27/stage0a/extract_bin/` and run it there. Smoke on the cluster: from such a directory transformers imports and
  GPT-2 tokenises offline. No check changed; selftest 19/19 and T0–T2 PASS re-run locally. 0a resubmitted
  whole (pre_targets is recomputed; G is re-fitted from the same deterministic sample).
- **2026-10-06T14:24:11+08:00, Deviation 11 — 0b job 19230245 failed on an environment error before any training; sbatch-only fix.**
  The job ran on gpu-node (NVIDIA H200) from 2026-10-05 15:33:35 to 15:39:13 EDT; sacct FAILED 127:0, elapsed 5:42.
  Gate checks reached: G was read from the 0a json, which says PASS, and the preflight passed for both units.
  Then both units exited rc=127 at once: `/usr/bin/time: No such file or directory` (the H200 nodes do not have
  GNU time). Training never started, so no 0b check (counts, floor) was reached and nothing was measured.
  What it did record: 23 utilisation samples, all 0% (preflight plus the 300 s stagger), and a peak of 945 MiB
  GPU memory (the preflight's synthetic conv steps). Fix, sbatch only: a dependency-free python wrapper
  (`subprocess.call` + `getrusage(RUSAGE_CHILDREN).ru_maxrss`, KiB on Linux) writes the same "Maximum resident
  set size (kbytes)" line the report parser reads, plus wall-clock. The job first self-tests the wrapper: a
  child that allocates 200 MiB and exits 3 must report rc 3 and ≥ 200,000 KiB. On the cluster the wrapper reported
  211,664 KiB and rc 3. This replaces the per-unit MaxRSS of /usr/bin/time -v; it is the largest single
  descendant process, which here is the train_mb process. sacct MaxRSS is kept as a cross-check. No check or
  threshold changed; T0–T2 re-run locally: PASS. The failed run's logs are archived under
  `results/subjects27_20261004/stage0/failed_19230245/` and moved aside on the cluster, so the report parser sees
  only the new run.
- **2026-10-07T01:06:52+08:00, Stage 0b result (JID 19267362, gpu-node, H200; 2026-10-06 03:03–03:57 EDT; COMPLETED, sacct 54:35)
  → Stage 0 PASS. HARD STOP 0.** Receipt: `results/subjects27_20261004/stage0/stage0_report.json`
  (sha256 3a7e6774…); report script sha256 c9985512….
  - Gates. The timing-wrapper self-test passed (rc 3, 211,032 KiB). The printed counts equal A7 in both units
    (A: train_seg 153,976 / heldsubj 6,555 / eval 3,642 / calib_pool 2,613; B: 55,753 / 6,555 / 3,642 / 2,613).
    Competence floor r at k=500: A conv 0.940, dynamic_big 0.940; B conv 0.870, dynamic_big 0.891 → PASS. The
    per-subject Δ is not analysed (sub-01 re-enters in Stages 1/2). The pretraining validation curve has the
    same shape as the old-archive strengthen run (chance until ~ep 64, then rises), read from
    `logs/strengthen_20260926/train_st26_rr_seed1_s01.log`.
  - Measured cost. Job 0.910 GPU-h (sacct elapsed 3,275 s × 1 GPU). Unit wall-clock: A 3,214 s, B 1,828 s.
    Segment build A 1,230 s, B 869 s. Pretraining A 976 s (conv) + 879 s (dynamic_big), B 397 s + 410 s.
    Adaptation per k, A: 8 / 14 / 38 s (k = 100 / 500 / 2000). MaxRSS A 183.2 GB, B 71.9 GB; batch step
    250.2 GB of the 300 GB request; peak VRAM 5,677 MiB.
  - **GPU utilisation: job mean 40.7%, below the cluster's 50% cancellation line.** It was not cancelled. By phase:
    0.9% during A's segment build, 67% during A's pretraining, 41% while both units overlapped, 53% after B
    ended; 0% in the first 15 min. The build is CPU/IO-bound and dominates the idle time.
  - Projection (prereg §5 formula, registered two-units-per-GPU design; in-sample check on B: predicted
    1,660 s vs measured 1,828 s, so the formula runs about 9% low): **Stage 1 9.4 GPU-h** (29 units, 15 tasks;
    floor 4.7 h wall-clock at 2 GPUs); **Stage 2 three cells 120.2 GPU-h** (2A RR 45.0, 2B UNSEEN-0 43.6,
    2C SEEN 31.7; floor 60 h wall-clock); optional 2D SSc 46.0. Contention factor s = 0.139; overhead
    61 s per task.
  - Consistency check (§5): unit B / median strengthen block-B unit = 1,828 / 927.5 = 1.97 — just under the 2×
    stop line. B was paired with the IO-heavy 27-roster build, which the strengthen units never were, so the
    Stage 1 figure is probably high. The check passes at the edge and is reported as such.
  - Not the registered design, for D1/D2 only: with the segment build moved to a CPU step, the same measured
    rates give Stage 1 5.3 GPU-h and Stage 2 76.6 GPU-h (3 cells). This scenario is also the obvious remedy
    for the <50% utilisation. Pairing two 27-roster units needs about 2 × 183 GB of host RAM (nodes have
    ~2 TB; request ≥ 400 GB).
- **2026-10-07T01:16:19+08:00, Deviation 12 — segment building moves to a CPU job; written before any Stage 1 job; nothing submitted;
  D1 (yes/no + GPU-h cap) still pending.**
  *Why (d), the utilisation argument.* In 0b the GPU sat at 0.9% while unit A built its segments (1,230 s of
  3,214 s), and none of the first 15 min was above 0%. It ran at 67% during pretraining. The build is CPU/IO
  work on the 84 GB memmap, and it is what pulled the job mean down to 40.7%, under the cluster's 50% cancellation
  line.
  *Design.* `scripts/subjects27_segcache.py build` runs on cpuA_partition. It builds every segment of every recording
  of a view with the pinned `build_segments_with_meta`, one chunk of 8 recordings at a time, in the
  metadata's groupby(sort=False) order. It writes X.npy (float32), Y.npy, rec.npy, on.npy and a
  manifest with the params, input SHAs and per-file sha256. GPU units use a new bundle
  `scripts/subjects27_20261007_segcache/src_pinned` (train_mb sha256 2820f780…). It is the Stage 0 bundle plus a
  default-off `--seg_cache` that slices the selected recordings' rows out of the cache, in cache order. It
  refuses: any manifest param or trial_metadata sha that differs, any mask that is not recording-level, any
  recording order that differs from the build order, and `--dose`. Segments are not rebuilt on the GPU node.
  *(a) Bit-identity gate (registered; the GPU array is dependency-chained afterok on it).* Stage 0b did not save
  its segment arrays, so the gate compares against the same function on the same inputs. It rebuilds, on the
  CPU node, the pretraining and held-subject segments of 0b unit A (27-roster, sub-01, RR) and unit B
  (ten-roster, sub-01, RR) with `build_segments_with_meta` exactly as train_mb does, plus a SEEN-cell unit
  (27-roster, sub-01) for the story masks Stage 2 needs. It requires the sha256 of X, Y, rec and onsets to be
  identical to the cache slices, the cache files to match their manifest, and the counts to equal 0b's printed
  ones (A 153,976 / 6,555; B 55,753 / 6,555). In addition, Stage 1 reruns unit (01, seed 0) from the cache, and its
  result JSON must equal 0b unit B's apart from tag/timestamp/seg_cache. A mismatch holds all Stage 1 receipts
  until it is explained. Local tests (planted archive): segcache selftest 5/5 (chunked == one-shot bytes;
  take == build for RR/UNSEEN0/SEEN; a flipped last bit, a partial-recording mask and a count mismatch are
  caught). Bundle tests S0–S2 PASS: without the flag the result equals the Stage 0 bundle's; with the cache it
  equals on-the-fly building for RR and story-holdout, with G.
  *(b) Memory.* Stage 1 GPU tasks (two ten-roster units) request 200 GB; 0b unit B peaked at 72 GB including
  the build this design removes. Stage 2 pairs of 27-roster units request ≥ 400 GB (0b unit A 183 GB with the
  build), or run one unit per node. Revisit with Stage 1's measured MaxRSS. The CPU job requests 300 GB (a
  fresh 27-roster reference build is the 183 GB case; cpuA_partition nodes have 513 GB).
  *(c) Re-projection (same measured 0b rates, build removed).* Stage 1 ≈ 5.3 GPU-h; Stage 2 (3 cells) ≈ 76.6
  GPU-h; 2D SSc ≈ 30.0. Disclosed with it:
    - the formula under-predicted 0b unit B by about 9% (1,660 vs 1,828 s);
    - the strengthen consistency ratio is 1.97× (limit 2×);
    - reading the slice from the cache is not free, and that time is not in the projection;
    - the first Stage 1 task measures it, and the >1.5× pause rule applies.
  Hard ceiling of the prepared Stage 1 array: 15 tasks × 1 h walltime = 15 GPU-h. The array refuses to start
  unless the author's cap is ≥ 15 (otherwise trim the array).
  *Files (prepared, not submitted):* `scripts/slurm/subjects27_stage1_segcache_cpu.sbatch` (cpuA_partition, 5 h;
  self-test, both caches, gate (a)), `scripts/slurm/subjects27_stage1_gpu.sbatch` (gpuA_partition,gpuB_partition,
  array 0-14%2, 1 h, 200G, HF vars one per line, 15-s utilisation, timing wrapper),
  `scripts/subjects27_stage1_submit.sh <cap>` (probe, rsync, sha diff, sbatch CPU, sbatch GPU
  --dependency=afterok). Both sbatch files require `SUBJ27_APPROVED=D1`.
- **2026-10-08T02:51:57+08:00, D1 recorded; Deviation 13 — Stage 1 under the author's 12 GPU-h cap; analysis script committed before
  any Stage 1 job.** D1 reached this session relayed by the coordinator ("Author approved D1 (2026-10-08, chat:
  GPU都批) … GPU-h cap 12"). The prepared array (Deviation 12) had a 15 GPU-h ceiling (15 tasks × 1 h). It now
  has a walltime of 48 min, so 15 × 0.8 h = 12.0 GPU-h, and it refuses to start unless the cap is ≥ 12.0. The
  build-removed projection is 0.355 h per task on average, so 48 min leaves ≥ 2× headroom. Nothing else changes.
  The GPU family is pinned to H200: the job exits on any other GPU, because 0b and every published run were on
  H200 and builds differ across GPU families. The 1.5× pause rule (§2) is applied to Stage 1 as well: if the
  first completed GPU task takes more than 1.5 × 1,277 s = 1,916 s, the pending array tasks are held
  (`scontrol hold`), the projection is redone from that task, and the author is asked before release. The
  Stage 1 analysis (`scripts/subjects27_stage1_analysis.py`) is committed now, before any job. It implements
  §3/§4 exactly: G0 (old three-seed k=100 reproduces +0.048 [+0.024, +0.071], checked: PASS); G1 (30/30 units,
  args echo, input_gain = G, seg_cache, per-k n and pool equal to st26_rr_seed1); the Q3 labels and the GO/NOGO
  rule; the bridge CONSISTENT/DRIFT label; the E2E comparison of unit (01, seed 0) with 0b unit B. Selftest 8/8:
  the bootstrap equals mb_conditions.ci bit for bit, all four Q3 branches are hit, a negative seed blocks PASS,
  and the bridge labels behave as specified.
- **2026-10-08T03:21:12+08:00, Deviation 14 — CPU segment job moved from cpuA_partition to QoS QOS_B (partition cs), before anything was
  queued.** The first `subjects27_stage1_submit.sh 12` stopped at sbatch: "partition 'cpuA_partition' is not valid
  for this job". `sacctmgr` shows the cpuA_partition QoS caps each user at cpu=32, mem=120G. Test-only runs: 64G
  and 120G are accepted, 128G and 300G are rejected. The registered 300G is kept (gate (a) rebuilds the
  27-roster unit A, which peaked at 183 GB in 0b), so the job uses `--qos=QOS_B` with no partition line.
  Test-only placed it on partition cs. Nothing else changes; no job had been created.
- **2026-10-08T03:50:01+08:00, Deviation 15 — CPU segment job 19364750 failed on a filesystem error; script-only fix; the
  dependent GPU array 19364751 was cancelled before any task ran.** The selftest passed. Then the 27-roster cache
  build wrote all 25 chunks (160,621 segments in 364 s) and failed while removing its chunk directory:
  `OSError: [Errno 39] Directory not empty: …/segcache/merged27_bci/_chunks`. One chunk was still open as a
  memmap (to read its shape), and on /scratch (VAST, NFS) deleting an open file leaves a silly-rename
  `.nfsXXXX` entry. The directory was empty once the process exited, which confirms the cause. Fix in
  `subjects27_segcache.py`: drop the memmap before writing, gc before deleting, and retry rmdir for up to 60 s,
  then fail loudly. No check changed. Gate (a) never ran, so no segment was compared and no GPU time was used.
  The partial outputs were deleted on the cluster and the failed logs archived under
  `results/subjects27_20261004/stage1/failed_19364750/`. Segcache selftest 5/5 and bundle tests S0–S2 were re-run,
  and the pair was resubmitted with the same script and cap.
- **2026-10-09T16:52:20+08:00, Stage 1 first pass and Deviation 16 (limit change only).** CPU job 19368959 COMPLETED (1,403 s, cpu-node):
  the caches were built (27-roster 160,531 segments in 372 s; ten-roster 62,308 in 78 s) and **gate (a) PASSED**.
  A_rr_s01 153,976 / 6,555, B_rr_s01 55,753 / 6,555 and SEEN_s01 83,585 / 6,555 are byte-identical to a fresh
  pinned build, the cache files match their manifests, and the A/B counts equal 0b's.
  GPU array 19368966 (all H200): 11 tasks COMPLETED in 1,031–1,655 s. Tasks 2, 3, 5 and 6 hit TIMEOUT at the
  48-min cap (2,882–2,902 s): units (s07, s08, s09, s10; seed 0) and (s01, s02, s03, s05; seed 1), all ten-roster
  RR, 8 units with no result JSON. Cause, as measured: in those tasks conv pretraining alone took 1,860–2,350 s,
  against about 360 s in a normal unit. GPU utilisation was 7–16% (normal tasks 47–72%); the printed counts and
  the cache load were normal. So the GPU was starved on the host side. Batch CPU time was not higher, and nodes
  gpu-node and gpu-node also ran normal tasks (12, 13). The cause is not identified and is not a code-path
  difference. **The 1.5× rule did not fire.** The watcher process died at 2026-10-08 05:31 CST, before the first
  GPU task started. Had it been alive, it would still not have fired: the rule looks at the first completed task,
  task 0, which took 1,125 s < 1,916 s. GPU used so far (sacct ElapsedRaw, 15 tasks): 25,540 s = **7.094 GPU-h**;
  4.906 GPU-h of the 12 GPU-h cap remain. Deviation 16: re-run exactly tasks 2, 3, 5, 6 (same units, sbatch,
  bundle, cache and G) with --time=01:13:00, so 4 × 1.217 h = 4.867 GPU-h and the worst-case total is 11.96 ≤ 12.
  Nothing else changes. The timed-out logs are archived (`results/subjects27_20261004/stage1/timeout_19368966/`).
  The Stage 1 analysis is not run until all 30 units are in.
- **2026-10-10T01:00:34+08:00, Stage 1 RESULT: Q3 = FAIL_UNRESOLVED → NOGO. Stage 2 is not run** (no-go rule §2, not waivable).
  Inputs: 30/30 units (sha256 identical local/remote). All 15 GPU tasks ran on H200. Re-run 19476016 tasks
  2/3/5/6 completed in 3,118 / 1,099 / 1,707 / 1,039 s. GPU total from sacct: 25,540 + 6,963 s = **9.03 GPU-h**
  (cap 12). Analysis run once: `scripts/subjects27_stage1_analysis.py` → `results/subjects27_20261004/stage1/stage1_analysis.json`
  (sha256 a211d1c5…). Gates: G0 PASS (old three-seed k=100 +0.048 [+0.024, +0.071]); G1 PASS (args echo,
  input_gain = G, seg_cache, per-k n and pool = st26_rr_seed1). E2E: unit (01, seed 0) from the cache equals
  Stage 0b unit B exactly (true).
  - **k=100 (primary), rebuilt archive, ten subjects, three seeds: +0.035 [−0.001, +0.069], not resolved.**
    Seed means +0.025 / +0.023 / +0.057, all > 0. That is not PASS: the interval covers 0 → FAIL_UNRESOLVED.
  - Bridge k=100 (label): paired new − old −0.013 [−0.034, +0.007] → CONSISTENT. Realisation-matched seeds:
    seed 1 −0.032 [−0.095, +0.020], seed 2 +0.010 [−0.024, +0.049], both not resolved.
  - k=500: +0.018 [+0.007, +0.030], resolved (old +0.018 [+0.005, +0.029]); bridge −0.000 [−0.013, +0.012],
    CONSISTENT.
  - k=2000 (descriptive): +0.003 [−0.002, +0.007] (old +0.001 [−0.006, +0.008]).
  **Branch NOGO wording (§4, numbers filled):**
  > We rebuilt MEG-MASC from the raw release with new epoching code (identical trial selection; median
  > per-epoch $r = 0.9999$ with the archive once its global scale is restored) and reran the standard protocol
  > on the same ten subjects over three pretraining seeds. The $k{=}100$ advantage is not resolved:
  > $\bar\Delta = +0.035$ $[-0.001, +0.069]$ (seed means $+0.025$ / $+0.023$ / $+0.057$), against $+0.048$
  > $[+0.024, +0.071]$ on the original archive. The standard-protocol advantage is therefore sensitive to
  > preprocessing differences below the level the paper describes, and we did not extend a comparison that
  > does not survive a rebuild of its own archive to all 27 subjects. We will report this sensitivity next to
  > Table 2.
  Registered labels that go with it: "The paired difference between the two archives is itself not resolved
  ($-0.013$ $[-0.034, +0.007]$), and the $k{=}500$ advantage reproduces ($+0.018$ $[+0.007, +0.030]$)."
  Why ten: sentence F (AUTHOR TO CONFIRM) + "A 27-subject run was planned but stopped by this bridge check."
  Reading (not a claim): the point estimate stayed positive in every seed, and the archive difference is
  not resolved. The fragile part is the *resolution* of the k=100 advantage at n=10, not its sign. The bridge
  cannot tell whether the rebuild moved it or whether the original interval sat near its edge.
- **2026-10-10T04:13:57+08:00, erratum to the Stage 1 RESULT wording above (rounding only; no label, rule or verdict changes).**
  Three filled-in numbers were rounded from the ELN's 4-dp digest instead of the receipt. Read from
  `stage1_analysis.json` and rounded half-up: bridge k=100 paired difference **−0.014** [−0.034, +0.007]
  (stored −0.013533; printed −0.013 above), rebuilt k=500 **+0.018 [+0.006, +0.030]** (lower 0.0064667;
  printed +0.007), rebuilt k=2000 **+0.003 [−0.002, +0.006]** (upper 0.0064815; printed +0.007). The paper
  (R1, paper_iclr.tex) prints the receipt values; `check_paper_numbers.py` bans the double-rounded ones.
