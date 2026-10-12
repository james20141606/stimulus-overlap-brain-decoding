# PREREG — strengthening runs for #60432 (bci_split), 2026-09-26

Written 2026-09-26 03:15 +08:00 (first draft started 03:05). Committed before any job exists; no new result has been seen.
[Two lines naming internal review files removed for the public release.]
The tex is not touched by this work; the repo's writer agent integrates whatever branch fires.

## 0. What I have already seen (so the reader can discount it)

- Published Table 2 (seed 0, n=10, rank percentile, Δ = dynamic − static): standard cell (recording guard,
  recording split; `mb1f_ksweep`) +0.043 [+0.009,+0.081] at k=100, +0.014 [+0.001,+0.026] at k=500;
  stimulus/story cell = SEEN arm `mb1k_volctl` (story 3 withheld) +0.020 / −0.006; RS `mb1t_cellC`
  +0.013 / +0.010; SR `mb1t_cellD` +0.034 / +0.024; M_split k=500 −0.017 [−0.031,−0.001] (not after Bonferroni).
- ELN 2026-07-31: on `mb1f` (standard cell, seed 0) in ×chance, the endpoint ("flow") term dyn−noflow
  resolves only at k=100 (+3.49 [+0.31,+7.15]) and the trunk term noflow−conv resolves at k=200…2000 but
  not at k=100 (+4.76 [−1.02,+10.58]). **The standard-cell half of Q5 is therefore not blind** (different
  metric, same runs). The rank-percentile version has never been computed.
- ELN 2026-08-01: `mb1g_clean` (the clean four-story stimulus/story cell), n=3, ×chance: k=100 +1.09 ns,
  k=500 +3.19 ns. Rank percentile never computed at n=3; the other seven subjects were never run.
- No seed replicate of any factorial cell exists (the paper's "three-seed check" is the SEEN/UNSEEN arms).

## 1. Questions and estimands

Metric everywhere: rank percentile r (`fm.meanpct`), Δ_s = r_dyn − r_static per held-out subject s,
ten subjects {01,02,03,05,07,08,09,10,11,13}, k ∈ {100, 500}. Intervals: paired subject bootstrap,
20,000 draws, `default_rng(0)`, percentile (same draws as `mb_conditions.ci`, i.e. the published generator).
Bonferroni over the factorial's four contrasts = the same draws at 0.625/99.375.
"Resolved" = 95% interval excludes zero (the paper's usage).

- **Q1 Clean stimulus/story cell (W10).** Δ in a stimulus-guard + story-split cell whose pretraining keeps
  all four stories (the `mb1g_clean` configuration), seed 0. Plus the clean factorial: Eq. (factorial)
  contrasts with this cell in place of the SEEN arm (RR, RS, SR = the published seed-0 runs).
- **Q2 Seed stability of the standard-cell premise (W9).** Δ_RR at pretraining seeds 0 (published), 1, 2.
  Estimands: per-seed subject means with intervals; seed-averaged Δ per subject then subject bootstrap;
  within-subject across-seed SD vs between-subject SD of seed means.
- **Q3 Clean cell across seeds; the diagonal over seeds (W9, W7).** Δ_SSc at seeds 0,1,2; seed-averaged
  Δ_SSc − Δ_RR. Q3b: the full factorial averaged over three seeds (needs block D).
- **Q4 Guard radius (W28).** In every new run, Δ at radii 5, 10, 20 s evaluated on the same fine-tuned
  embeddings (primary stays 10 s). Seed-averaged per subject; Δ(5)−Δ(10), Δ(20)−Δ(10).
- **Q5 Single-ingredient ablation (W6).** `dynamic_big_noflow` = the dynamic encoder's residual-free trunk
  without the endpoint-displacement branch (same pooling and head). Trunk term T = r_noflow − r_static,
  endpoint term E = r_dyn − r_noflow, so Δ = T + E. Standard cell: published `mb1f` (seed 0, has the
  noflow arm, n=10). Clean stimulus/story cell: block A (seed 0). Also T_SSc − T_RR, E_SSc − E_RR.

## 2. Runs (sample)

Code: `scripts/strengthen_20260926/src_pinned/` = the calmatch bundle that produced the 09-24/25 primary-cluster
runs (train_mb.py 1a4953ec, eval_metrics.py c5b94117) plus three default-preserving changes to train_mb.py
(`--split_seed`, `--adj_list`, seed/radius echo in the result JSON); SHA256SUMS in the bundle. The local
test (`scripts/test_strengthen_20260926.py`) checks that with historical flags the result numbers equal the
calmatch bundle's, that `--adj_list` leaves every primary number unchanged, and that `--split_seed 0` keeps
seed 0's calibration/evaluation partition. Flags per run: `--variants conv,dynamic_big[,dynamic_big_noflow]
--layer 6 --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 --pre_epochs 80 --adapt_epochs 120
--seed {0,1,2} --split_seed 0 --adj_s 10 --adj_list 5,10,20`, no `--holdout_story` (all four stories in
pretraining). A seed changes initialisation, batch order and the calibration/validation draw; the
protocol's partition (which recordings or stories are calibration vs test) stays the published one.

| Block | Slurm tasks | Cell | Seeds | Arms | Serves |
|---|---|---|---|---|---|
| A | 0–4 | SSc: stimulus guard + story split, four-story pretraining | 0 | conv, dyn, noflow | Q1, Q4, Q5 |
| B | 5–14 | RR: recording guard + recording split | 1, 2 | conv, dyn | Q2, Q4 |
| C | 15–24 | SSc | 1, 2 | conv, dyn | Q3, Q4 |
| D | 25–44 | RS, SR | 1, 2 | conv, dyn | Q3b, Q4 |

Order = priority. Two subjects per GPU (`scripts/strengthen_20260926/units.sh`). Expected ~0.3 GPU-h per
two-subject task (calmatch history); A+B ≈ 4.5 GPU-h, +C ≈ 7.5, +D ≈ 13.5.

**Platforms.** The primary cluster (the one that ran every published run; its shared training env is used
read-only) runs the blocks in priority order. A secondary cluster (own env) is used only if the author
enables it, and then only for blocks the primary is not running; the allocation is written to the ELN at
submit time, before any result exists. The same unit never runs on both. One site-agnostic sbatch
(`scripts/slurm/strengthen_20260926.sbatch`); account, partition, memory, walltime and paths come from a
local, untracked site file via `scripts/strengthen_20260926/runbook.sh`. Every receipt records platform,
host and GPU; a block may mix platforms (the platform difference then counts as seed-level noise and makes
intervals wider, not narrower); a primary-only recomputation is printed as a descriptive check when mixed.

## 3. Completeness and timing

- **Cut-off 2026-09-26 10:15 +08:00.** Runs whose receipt is written later are excluded. The analysis
  (`scripts/strengthen_analysis_20260926.py`) runs once, at the cut-off or as soon as every submitted
  block is complete, whichever is earlier.
- A block counts only if all ten subjects pass gate G1 (receipt; SHA; design echo; radius-10 re-evaluation
  equal to the primary r to rounding; per-k evaluation n and pool identical to the same subject's reference
  run: RR→`mb1f`, RS→`mb1t_cellC`, SR→`mb1t_cellD`, SSc seeds 1–2→SSc seed 0) with every arm at k=100 and
  500. An incomplete block is reported as "not completed (x/10)" with **no point estimate**, in the paper
  or anywhere else.
- Gate G0: the published Table 2 per-subject advantages recomputed from the pulled JSONs must equal
  `results/interaction_receipt.json`; if not, Q1, Q3b are not analysed.
- If the primary cluster is unreachable at the cut-off, the analysis uses what has been pulled (SHA-checked); the rest
  is "not completed".

## 4. Decision rules (fixed now; thresholds are not changed after results exist)

**Q1** (block A; SSc Δ at k=100 and 500):
- WEAKEN if the SSc interval is resolved positive at k=100 or at k=500.
- SUPPORT if neither is resolved positive **and** at both budgets the SSc point estimate is below the
  published standard cell's (+0.043, +0.014).
- INCONCLUSIVE otherwise (not resolved, but a point estimate at or above the standard cell's).
- Q1b (label, clean factorial M_split at k=500): BONFERRONI_RESOLVED / UNADJUSTED_ONLY / NOT_RESOLVED.

**Q2** (block B; k=100 is primary because the paper rests the standard-protocol advantage on k=100):
- SUPPORT if the seed-averaged interval at k=100 is resolved positive **and** all three per-seed means > 0.
- REVERSED if the seed-averaged interval at k=100 is resolved negative.
- WEAKEN otherwise. k=500 is reported with its label; no rule.

**Q3** (blocks A+C; B for the diagonal): WEAKEN if the seed-averaged SSc interval is resolved positive at
either budget; SUPPORT if not and Q2 = SUPPORT; else INCONCLUSIVE. The seed-averaged diagonal
Δ_SSc − Δ_RR is reported with its label (resolved_neg is the only label that adds a claim).
Q3b (all blocks): the four contrasts with unadjusted and Bonferroni intervals, labels only.

**Q4** (every complete block): INSENSITIVE if, in every analysed cell and k, the "resolved positive"
status is the same at 5, 10 and 20 s **and** both mean shifts |Δ(r)−Δ(10)| < 0.01; else SENSITIVE (the
failing cells are listed).

**Q5** (block A + `mb1f` noflow arm): at k=100, an ingredient *carries* the standard-cell advantage if its
term is resolved positive in RR; it is *split-sensitive* if it carries and is not resolved positive in SSc.
Verdict: ONE_INGREDIENT:{trunk|endpoint}:{split_sensitive|not_split_sensitive}, BOTH_INGREDIENTS,
NEITHER_ALONE. k=500 labels reported alongside.

All results are reported whichever branch fires. A weakening branch narrows the paper with the sentence
below; no threshold, cell, seed or subject is changed after the fact. Deviations, if any, are appended
below with a timestamp and a reason, before the analysis is run.

## 5. Paper wording per branch (writer agent pastes the matching one; numbers from `analysis.json` only)

Q1
- SUPPORT: "A clean stimulus/story cell that keeps all four stories in pretraining does not resolve the
  advantage either (k=100: $a$ [$l$, $u$]; k=500: …; App. X), so Table 2's loss of the advantage does not
  depend on withholding story 3."
- INCONCLUSIVE: "A clean stimulus/story cell with all four stories in pretraining does not resolve the
  advantage, but its point estimate is not below the standard cell's, so the loss of resolution there is
  not evidence that the advantage is gone (App. X)."
- WEAKEN: "With all four stories in pretraining, the stimulus guard and story split leave the advantage
  resolved at k=… (… [.., ..]; App. X). The loss of the advantage in Table 2's stimulus/story cell
  therefore needs story 3 withheld from pretraining: the change that removes it is pretraining exposure,
  not the evaluation split." And every "no stimulus-held-out cell favours the dynamic encoder" becomes
  "no cell that withholds the evaluation story from calibration and pretraining favours …".
- Q1b NOT_RESOLVED adds: "With the clean cell the k=500 interaction does not resolve even before
  correction; the split sensitivity rests on which cells resolve, not on the interaction contrast."
  UNADJUSTED_ONLY: "With the clean cell the k=500 interaction again resolves before but not after
  correction." BONFERRONI_RESOLVED: "With the clean cell the k=500 interaction survives Bonferroni
  correction over the four contrasts."

Q2
- SUPPORT: "Across three pretraining seeds the standard-cell advantage at k=100 is positive in every seed
  (seed means $a$, $b$, $c$) and its seed-averaged interval excludes zero (App. X)."
- WEAKEN: "Across three pretraining seeds the standard-cell advantage at k=100 [is not positive in every
  seed | has a seed-averaged interval that covers zero] (App. X); the premise holds at the published seed,
  so the split sensitivity is shown for that run, not for the pair across training randomness." §7's
  seed item is rewritten to say this.
- REVERSED: as WEAKEN, with "reverses sign when averaged over seeds".

Q3
- SUPPORT: "Averaged over three seeds, the clean stimulus/story cell does not resolve the advantage at
  either budget, while the standard cell does at k=100."
- Diagonal resolved_neg adds: "Averaged over seeds, the change from the standard to the fully corrected
  cell is itself resolved at k=…"
- WEAKEN / INCONCLUSIVE: the Q1 sentences with "averaged over three seeds".

Q4
- INSENSITIVE: "Guard radii of ±5 and ±20 s leave every analysed cell's resolution unchanged and move the
  advantage by less than 0.01 rank percentile (App. X)."
- SENSITIVE: "The comparison depends on the guard radius in [cells]; ±10 s was fixed in advance and the
  other radii are in App. X."

Q5
- ONE_INGREDIENT:X:split_sensitive: "A single-ingredient ablation (seed 0) places the standard-cell
  advantage at k=100 on the [residual-free trunk | endpoint-displacement branch]; that ingredient's
  advantage is not resolved in the clean stimulus/story cell (App. X)." §7 item "neither … the
  contribution of either architectural ingredient is established" is narrowed accordingly.
- ONE_INGREDIENT:X:not_split_sensitive: "… places it on X, whose advantage persists in the clean cell."
- BOTH_INGREDIENTS: "Each ingredient alone carries a resolved share of the standard-cell advantage at
  k=100 (App. X)."
- NEITHER_ALONE: "Neither ingredient alone resolves at n=10; the sensitivity remains a property of the pair."

## 6. Limits stated in advance

n=10 subjects; three seeds is the minimum that gives a spread, not a variance estimate; one corpus.
Seed replicates hold the calibration/test partition fixed, so they price training randomness, not
partition randomness. Q5's standard-cell half re-reads runs whose ×chance decomposition was seen (§0).
The clean SSc cell has a different pool from the SEEN-arm cell (both halves of the story split are
scored); rank percentile is used for that reason (Lemma 1), and pool composition is not controlled.

## Deviations

- 2026-09-26 04:40 +08:00, before any job exists: cluster names, accounts and paths removed from this file
  and from the sbatch (anonymity; requested by the coordinator). The two per-site sbatch files became one
  site-agnostic sbatch fed by an untracked site file. No question, estimand, run, cut-off or rule changed.
- 2026-09-26 08:45 +08:00, before any job exists: the primary cluster was unreachable from 02:53 to 08:45
  (no VPN route), so no block can finish by the 10:15 cut-off. Per §3 this round is recorded as **not
  completed before submission**; the submitted paper gets no text from it. If the cluster returns later,
  the same runs, gates and rules are used unchanged for rebuttal evidence only; for that use the cut-off
  becomes the moment every submitted block has a receipt, or 2026-09-27 12:00 +08:00, whichever is earlier.
