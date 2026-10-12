# Lab-notebook excerpts: criteria and reading rules fixed before their results

Verbatim entries of the append-only lab notebook, in notebook order, except that identity
strings are replaced by placeholders (USER, $SCRATCH, CLUSTER_ACCOUNT, gpuA_partition, ...)
and lines about logging in to the compute cluster are removed. Job IDs are kept. The full
notebook is not released.

## 2026-07-29  PRE-REGISTRATION: stimulus-scoped guard re-run (criteria fixed BEFORE the job runs)
**Why pre-register:** my logged failure mode is framing outrunning evidence, and it fired three times on
2026-07-27/28 alone (acoustic arm, granularity, "near chance"). If I decide the reading after seeing the
numbers I will read any outcome as favourable. So the decision rule is committed here first.

**Decision taken (grilling session, user-approved):** guard on STIMULUS identity + stimulus time, not on
recording (option (a)); additionally report the collapsed-pool variant (option (c)) as a robustness
check. Rationale in docs/adr/0001-stimulus-scoped-guard.md.

**Background fact that forces this:** 9/10 subjects heard story 0 in TWO sessions, word-for-word
identical (position-by-position word_text agreement = 1.0000), with a ~77 s offset and only ~2 s of
per-word jitter. The current guard masks `rec == rec[i]` where rec includes session, so every masked
neighbour keeps an unmasked twin with an identical target.

**Metric:** pool-invariant percentile (`mean_meanpct`). NOT xchance -- the stimulus-scoped guard shrinks
the pool substantially, and xchance is not pool-invariant (our own §poolsize).

**Statistic:** timing minus brain percentile, paired over the 10 story-0 subjects, at stimulus-scoped
adj = 240 s (with the full sweep 10/30/60/120/240 reported).

**Committed decision rule:**
- If the paired difference at adj=240 s has a 95% CI EXCLUDING 0  ->  the claim "the confound is not
  merely short-range proximity" is EARNED. Restore it to the abstract, stating it was measured under a
  stimulus-scoped guard, and note the earlier recording-scoped sweep could not have established it.
- If the CI INCLUDES 0, or the difference collapses toward 0 as the guard widens  ->  the confound is
  substantially a proximity effect. Write that plainly: our own guard defect had masked it; §sweep's
  conclusion REVERSES; the thesis narrows to "shared-stimulus settings with a recording-scoped guard".
- If the stimulus-scoped guard leaves too few candidates to score (pool < 20 for any subject)  ->
  report as inconclusive at that width and fall back to the widest width that admits >= 20; do NOT
  quietly pick the width that looks best.
**Unchanged either way:** every head-to-head result, because both arms are scored in the same pool and
none of them relied on the guard removing anything. This is stated in the paper already.
**No claim will be written from this run until the criterion above is applied as written.**

## 2026-07-29  RESULT: stimulus-scoped guard re-run -- the claim is EARNED (criterion applied as written)
**Job 14991752**, both arms, held-out story 0, 10 subjects, guard widths 10/30/60/120/240, stimulus
scope (docs/adr/0001). Criteria were committed BEFORE submission, same day, above.
**Pool verification first** (`pool_per_width.py`, computed from metadata alone -- the mask depends only
on onsets and width, not the model, so this needed no re-run):
```
 subject n_seg   adj10  adj30  adj60  adj120  adj240
   01     194     181    153    118      52       1
   03      95      89     75     58      26       1   <- single-session subject, smallest pool
  ...
 min over subjects:      89     75     58      26       1
```
**adj=240 is DISCARDED**: under stimulus scope the minimum pool collapses to 1, so the metric is
meaningless there. That is also what produced the spurious percentile jump at 240 (timing 0.784->0.864,
brain 0.662->0.792): with a pool of ~1 the percentile is trivially high. The pre-registered rule covered
this case -- fall back to the widest width admitting pool>=20, and do NOT shop for a favourable width.
That width is **adj=120 s** (min pool 26).
**Result (pool-invariant percentile, timing - brain, paired over 10 subjects):**
```
 adj    timing   brain     diff        95% CI          min pool
  10    0.7837  0.6495   +0.1342  [+0.1215,+0.1468]      89
  30    0.7833  0.6512   +0.1321  [+0.1196,+0.1447]      75
  60    0.7836  0.6537   +0.1299  [+0.1172,+0.1427]      58
 120    0.7844  0.6623   +0.1221  [+0.1091,+0.1351]      26   <- decision width
 240      --      --        --     discarded (pool=1)      1
```
**VERDICT per the pre-registered rule: EARNED.** At the widest guard this dataset can support, with the
guard scoped to the stimulus so every twin is masked, the timing arm still beats the brain by +0.122
percentile with a CI excluding zero. Widening the stimulus-scoped guard 12x (10 -> 120 s) moves the gap
only 0.134 -> 0.122. So the confound is **not merely short-range proximity** -- and this is now measured
under a guard that can actually refute it, which the recording-scoped sweep could not.
**Retroactive insight:** the old sweep's widest point (recording-scoped 240 s, pool ~97) was only viable
BECAUSE it ignored the twin presentation. Under the correct scope, 240 s exhausts the pool. That is a
cleaner statement of the original defect than the one I wrote yesterday.
**Scope of the claim (to state in the paper, not to bury):** held-out story 0, 10 subjects, one dataset,
percentile metric, single seed per arm. It licenses "not merely local", not "global" in any absolute sense.

## 2026-07-30  MB VERDICT: H1 HOLDS -- dynamic tokens win in the low-shot regime (10/10 subjects)
**Job 15007389 completed the k-sweep to 10/10 subjects.** Criterion applied: paired dynamic - static
few-shot x-chance per held-out subject, at each calibration budget k. Within a k all arms share the same
pool and eval, so the paired comparison is valid.
```
     k     conv      dyn   noflow |  dyn-conv            95% CI          verdict
   100    15.26    23.51    20.02 |    +8.26   [+2.05,+14.46]   DYNAMIC WINS
   200    11.37    19.43    17.71 |    +8.06   [+3.61,+12.52]   DYNAMIC WINS
   500    21.18    26.71    27.26 |    +5.53   [+1.74, +9.33]   DYNAMIC WINS
  1000    25.38    28.77    29.39 |    +3.39   [-0.04, +6.83]   ties
  2000    28.26    30.48    31.48 |    +2.21   [-0.51, +4.94]   ties  (n=9)
```
**H1 HOLDS, with the predicted structure:** the dynamic advantage is largest at the smallest calibration
budget and decays monotonically (+8.26 -> +2.21), converging to a tie by k>=1000. That is exactly
MANIFOLD_BRIDGE.md's H1 ("in LOW-SHOT cross-subject transfer a dynamics-aware token transfers better
than a static amplitude patch") and it is the mechanistically sensible signature: a prior helps when
target data is scarce and stops mattering once there is enough of it to fit anything.
Ordering is dynamic_big > noflow > conv at every k, consistent with MB-1e's zero-calibration result
(flow 0.909 > noflow 0.897 > conv 0.894), so the flow feature contributes on top of the dynamic
architecture rather than being the whole effect.
**I WAS WRONG EARLIER, and this is why the criterion was pre-specified.** On 2026-07-27 I reported, from
ONE subject's partial data (k=100: conv 37.2 / dyn 38.9 / noflow 39.0), that the arms looked "basically
tied" and that H1 "may be failing". With all 10 subjects the conclusion reverses decisively. Reading a
trend off one subject was the error; waiting for the full sweep was correct.
**Consequence for the program:** per the Q6 criterion, H1 partially holds, so the CROSS-DEVICE arm
(MASC<->LibriBrain) is now worth funding -- it is no longer a case of spending compute to confirm a known
negative. the architecture study has a positive headline, not a negative one.
**Still missing for the architecture study to be a paper:** the cross-device direction (zero results), and the headline
figure needs both directions per the doc. No draft exists.

## 2026-08-10T22:20:00   arms D/C' built, dry-run verified, submitted (15599825); top-1 eval queued (15599772)

**Why:** The adversarial audit named one experiment as the gap between 7 and strong accept: a one-bit
exposure contrast INSIDE the external pipeline, at matched volume, matched evaluation stimulus,
matched pool. And the reproduction had never been validated against Défossez's published >41% top-1.
Both were built and launched by agents; this entry records their state-changing work (they do not
commit).

**Design (patches/bm_mixed_split.py + scripts/slurm/repro_mixed.sbatch):**
```
  C'  train = lw1 + easy_money                    118,310 ex   } same valid (cable_spool_fort),
  D   train = lw1 + easy_money + BW-train-blocks, 119,502 ex   } same test  (BW's block-hash test
      thinned by BM_TRAIN_FRAC=0.5657                          }  blocks, byte-identical, 32,340 ex)
  D - C' = does having OTHER blocks of the evaluation narrative in training help, and by how much.
```

**Dry-run evidence (CPU, all 196 recordings):** per-story assignment correct in both modes
(lw1 28->train, easy_money 113->train, cable_spool_fort 71->valid, BW 31->test with 100->train in D /
122 dropped in C'); `# Examples` Cprime 118310|53933|32340 -- train EXACTLY equals arm C's 118,310,
confirming the lw1+easy_money identity; D natural 202,360 -> calibrated frac 0.5657 lands 119,502
(+1.0%, vs arm B's +1.8% precedent). Valid and test bit-identical across modes.

**Bug found and fixed during composition:** the BM_TRAIN_FRAC stage located "train" as
`split.max()`, which on an all-valid cable_spool_fort recording is the VALID index -- it would have
silently thinned arm D's validation set. Now `_train_ind = len(cdf)-1` under mixed mode. Default
(env unset) behaviour byte-unchanged; idempotence verified; backup events.py.orig_mixedsplit.

**Dora signatures:** 4401(Cprime)=8baf1b03, 4402(D)=38605352, both fresh -- no checkpoint reuse.
Seed<->mode pairing hard-coded in the sbatch (env vars do not enter dora signatures).

**Disclosed limitation for the write-up:** the shared test set is BW-only, so the retained-query
count (and hence pool) will be whatever it naturally is -- identical between D and C' by construction
but NOT pool-comparable to arms A/B/C. Only D vs C' is admissible; the sbatch header says to record
the "negatives selected" line.

**top-1 eval (job 15599772):** new xp b78e9a79 from `dora info seed=3102 test.wer_topx=1`, seeded
with a patched copy of 1e87bf54/checkpoint.th (`last_test_epoch=-1` to force the test stage,
`best_loss=-1e30` so the one unavoidable resume epoch cannot displace the epoch-4 best_state),
BM_WER_MAX_SEGMENTS=7140, story/mixed/frac env unset. Readout: top-1 = 100 - wer, to sit next to
Défossez's published >41%.

**Queue:** 15599772 PD(Priority), 15599825 PD(Priority). scratch 93.12%.

## 2026-08-11T20:30:00   seed replication of the one-bit contrast submitted (15639245)

**Why:** The final external review's stated gap between accept and strong accept: "the external
isolation remains single-run." Seeds 4403-4406 give the D-C' paired difference at n=3 (with 4401/4402).

**Command(s):**
```bash
# derive repro_mixed_seeds.sbatch from repro_mixed.sbatch (job bci-mixseeds, sentinel
# mixed_seeds_ALL_DONE, MAXCHUNKS=40, arms INTERLEAVED per seed: Cprime 4403 / D 4404 /
# Cprime 4405 / D 4406, so a partial chain yields complete PAIRS)
sbatch code/scripts/slurm/repro_mixed_seeds.sbatch    # -> 15639245 PD
```


**Design note:** interleaving Cprime/D per seed means a culled or stalled chain still leaves complete
seed-pairs for the paired difference, rather than three of one arm and one of the other.

**Expected:** ~4 runs x ~6-8 epochs x ~35 min ≈ 14-18 GPU-hours through 1h45m chunks. Readout: paired
D-C' at n=3 with a sign test / range, upgrading "point estimate without an interval" in §5 and the
Limits bullet. Reading fixed now: if the three paired differences share the +1.8-point sign, the
exposure estimate gains a crude interval and the Limits bullet narrows; if they straddle zero, §5's
estimate is demoted to "not resolved at n=3" and the ordering claim (price of fix >> leak) is
retained only if it survives all three pairs.

## 2026-08-17T01:10:00   A/B/C decomposition seed replication launched: B+2 seeds, C+2 seeds, parallel chains

**Why:** Round-5 review (8.8, strong accept) names one remaining computable gap: the A/B/C
decomposition is single-run -- "the paper still cannot estimate the volume-versus-exposure
decomposition with inferential confidence". Inventory on cluster: arm A already has n=3
(58.41/57.79/57.21, the quoted 1.2-point spread; seeds default/3101/3102); B has only 4301
(51.96), C only 4101 (47.69). The missing variance is B and C.

**Design:** two parallel self-resubmitting chains, 2 GPUs each, exact arm-env reuse:
- `repro_b_seeds.sbatch` -> arm B seeds **4302, 4303** (env == repro_volmatch: block split,
  BM_TRAIN_FRAC=0.5824, TRAIN_FRAC_SEED=0 so the retained-train subset is FIXED across model
  seeds -- the seed moves only optimization randomness; pool 7140). Sentinel b_seeds_ALL_DONE.
- `repro_c_seeds.sbatch` -> arm C seeds **4102, 4103** (env == repro_storysplit: BM_STORY_SPLIT=1,
  SPLIT_SEED=9 fixed partition; pool 7140, cap binds). Sentinel c_seeds_ALL_DONE.
- Seed ranges are load-bearing: env vars do not enter the dora signature, so B and C at the SAME
  seed would share a checkpoint dir. B stays 43xx, C 41xx. Verified 4302/4303/4102/4103 unused,
  both patch families still applied in events.py (TRAIN_FRAC x5, STORY_SPLIT x6 hits), sentinels
  and chunk counters clean.
- Two chains instead of one interleaved script because two instances of ONE script would race:
  .DONE appears only at completion, so both would start the same first arm.

**Command(s):**
```bash
scp repro_{b,c}_seeds.sbatch cluster:.../code/scripts/slurm/
ssh cluster 'sbatch .../repro_b_seeds.sbatch; sbatch .../repro_c_seeds.sbatch'
```

**Output digest:** B-chain 15883646, C-chain 15883647, both PD (Priority) on gpuA_partition.

**Observation:** Wall clock ~2 runs per chain in parallel; B peaked at epoch 4 historically, C ran
longer. Known chain failure mode: successor instant-death exit 0:53 on a bad node (twice before);
recovery is external resubmission of the same script (idempotent via .DONE/sentinel). Check state:
`ls $SCRATCH/repro/{b,c}_seeds_ALL_DONE` and the LOGD .DONE files. Pre-registered use of the
result: recompute A-B and B-C with per-arm n=3 means and ranges; upgrade S5's "resolved only
coarsely, to roughly +-15 percentage points" to whatever the spread actually supports, and revise
the Limits bullet "Multi-seed replication of the A/B/C arms is the obvious next run" to report it.
If the B/C spreads come out wider than arm A's 1.2 points, the +-15pp coarseness statement gets
WIDER, not narrower -- the paper follows the data either way.

## 2026-08-19T06:10:00   Designs 1+2 LAUNCHED: pool-identity rescore (15981568) and D thinning-realizations chain (15981569)

**Why:** cluster back after the network outage; queue idle. The two highest value-per-GPU-h designs
from the experiment scout go in.

**Pre-flight completed:** patches/bm_wer_pool_seed.py applied on the cluster (bm/wer.py now saves/reseeds/
restores CPU RNG around the WER loader and logs "wer pool identity sha1=..." -- backup
.orig_poolseed). Sig discovery verified live: `dora info seed=4401` -> Sheep(8baf1b03), exactly the
ELN-recorded 4401 signature; the sbatch parses Sheep(<8hex>. Rescore time budget raised to 5h;
per-arm .DONE files make a resubmit continue where a killed chunk stopped.

**Command(s):**
```bash
sbatch code/scripts/slurm/repro_onebit_rescore.sbatch      # 15981568, ~6-10 GPU-h
sbatch code/scripts/slurm/repro_d_realizations.sbatch      # 15981569, chain, 2 runs
```

**Pre-registered readings (in the sbatch headers, fixed before submission):**
- Rescore: hashes match across arms -> qualification (ii) deleted, D-C' re-quoted on the common
  pool; hashes differ -> identity matching impossible, constraint recorded; any pair <= 0 under the
  common pool -> separation demotes and the paper says so. Only the blind BM_WER_POOL_SEED=0 draw
  is ever reported.
- D realizations (seeds 4407/4408 = thinning realizations 1/2): both above every C' -> the
  one-realization clause is deleted and the permutation floor tightens to 1/C(8,5)~0.018; any at or
  below the best C' -> separation demotes to majority ordering, the lottery enters Limits with its
  measured spread.

**Sentinels:** onebit_rescore_ALL_DONE, d_realizations_ALL_DONE. Both scored under
BM_WER_MAX_SEGMENTS=7140 + BM_WER_POOL_SEED=0.

## 2026-08-30T13:35:00   DESIGN 4 LAUNCHED: second evaluation narrative (easy_money), full-dose pair, calibrated to -0.39%

**Why:** the last unlaunched on-corpus design -- kills the "for that assignment" single-narrative
scope of every S5 number. Post-design-3 improvement over the scout spec: D2 runs at FULL dose
(BM_TRAIN_FRAC_EVALSTORY=1.0), so the second narrative replicates the full-dose leak quantity
directly against BW's measured +3.00.

**Enabling patch:** patches/bm_mixed_story_choice.py -- BM_MIXED_EVAL_STORY / BM_MIXED_VALID_STORY
make the narrative assignment selectable; defaults byte-identical (backup .orig_storychoice).

**Calibration (CPU dry runs):**
```
Cprime2 natural: 163,189 | 53,933 | 24,843     D2 natural: 218,773 (EM train blocks = 55,584)
D2full: naive 0.6594 -> 160,840 (-1.44%); calibrated 0.6741 -> 162,548 (-0.39%) ACCEPTED
(err-low is the conservative direction: an under-fed D2 can only understate the leak)
```
Test = easy_money block-hash TEST blocks, byte-identical within the pair; ~12.4k retained -> the
7,140 cap BINDS; valid stays cable_spool_fort in both arms. Absolute top-10 NOT comparable to the
BW pair (different narrative, 163k vs 118k natural volume); admissible cross-narrative statement =
sign + relative size of the within-pair leak. Pre-registered reading in the sbatch header (same
sign as +3.00 -> replication clause; zero/opposite -> print non-replication, BW numbers stand;
escalate only if |leak2| < 1.2).

**Chain submitted; seeds 4601/4602; sentinel story2_pair_ALL_DONE.**

## 2026-09-12T01:26:17   Matched-volume CLOCK arm launched (the last control the paper names as unrun); reading pre-registered
**Why:** §5 of both variants ends "the matched-volume \emph{clock} arm is the control we have not run." The decoder's volume term is measured (arm B); the clock's is not. One run closes it before the ICLR deadline (abstract 9/18, paper 9/25). Cluster reachable today with no auth gate; BCI queue empty.
**Command(s):** ```bash
# new: scripts/slurm/repro_clock_volmatch.sbatch = repro_b_seeds.sbatch env (block split, BM_TRAIN_FRAC=0.5824,
#      BM_TRAIN_FRAC_SEED=0 -> arm B's exact retained subset, BM_WER_MAX_SEGMENTS=7140) + BM_TIMING_ONLY=1, seed 4801
#      (48xx unused anywhere; env vars are not in the dora signature). Chunked 1h45m chain, successor queued first,
#      .DONE on rc==0, sentinel clock_volmatch_ALL_DONE, MAXCHUNKS=10.
scp scripts/slurm/repro_clock_volmatch.sbatch cluster:$SCRATCH/project/code/scripts/slurm/
ssh cluster 'bash -lc "cd $SCRATCH/project && sbatch --parsable code/scripts/slurm/repro_clock_volmatch.sbatch"'
# generator: scripts/abc_decomposition.py gained a CLOCK VOLUME CONTROL section (reads volmatch/raster_4801.log,
#            asserts pool 7140, prints A/B/C clock contrasts and the pre-registered branch).
```
**Output digest:**
```
JID=17378448   bci-clockvm  PD  (gpuA_partition, gpu:2, 1h45m chunks)
existing clock arms: A = 89.52 (R2 14377929) / 89.57 (raster_3201) / 92.00 (raster_3202); C = 91.79 (storysplit/raster_4201)
arm B realised 120,389 train examples; the raster arm must log the same count (same frac, same thinning seed)
```
**PRE-REGISTERED READING (fixed before any result):** A's seed range 89.52-92.00, width 2.48, half 1.24 -> boundary 88.28.
- Branch 1: clock B >= 88.28 -> the clock is insensitive to the volume cut; the unrun-control clause is replaced by the measurement in both variants, and the convergence caveat's "insensitivity to volume near ceiling" becomes measured rather than argued.
- Branch 2: clock B < 88.28 -> the clock does lose to volume, so the story split GAINS at matched volume (C-B > 0); the "does not drop" sentence is re-scoped to the decomposition (volume loss + split gain). Not deleted.
- Escalate to seeds 4802/4803 only if clock B lands within 1.0 of 88.28. No decoder number moves in either branch.
**Observation:** Expected cost ~11 GPU-h (raster epoch at 58% volume ~30 min; best-valid at epoch 1 in every clock arm so far; patience 10 -> ~11 epochs, 4 chunks). First check: the train-example count in the log must equal arm B's 120,389, else the run is not the matched control and is discarded.
