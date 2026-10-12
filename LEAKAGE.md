# Reporting cross-subject decoding on shared stimuli

Naturalistic neuroimaging datasets present each stimulus to many participants — in MEG-MASC and the
Brennan EEG corpus, the same stimuli to all of them — because functional alignment methods such as
hyperalignment and shared response models need time-locked common stimulation. A consequence is easy to miss: **holding out the subject does not hold
out the stimulus.** A model fitted on other participants has usually already been fitted to the
material it is evaluated on, through their recordings.

This is not benchmark contamination in the usual sense. In language models, evaluation text leaks into
a training corpus by accident and the remedy is decontamination. Here the overlap is required by the
design. Remove the shared stimulus and the method it supports disappears with it. The transfer signal
and the contamination are the same quantity, so the question is not how to remove the overlap but how
much of the reported effect it accounts for.

## What we measured

On MEG-MASC, holding the evaluation narrative fixed and varying only whether it had appeared in
pretraining (story count matched, not volume: the UNSEEN arms pretrain on 32–69% more segments than SEEN, and
calibrate on stories {1,2,3} against SEEN's {1,2} at the same k; volume is exactly pinned only in the
dose-response; matched candidate pools):

| encoder | calibration budget | above-chance signal retained |
|---|---|---|
| static conv | 100 | 71.7% [60.2, 83.6] |
| static conv | 500 | 54.1% [51.4, 56.6] |
| static conv | 2000 | 53.9% [52.2, 55.8] |
| dynamics-aware | 2000 | 50.3% [48.5, 52.4] |

At equal k, UNSEEN calibrates on three stories and SEEN on two; restricting UNSEEN to SEEN's two changes the static retained fraction by −0.8 and −1.6 points at k=500 and 2000 (paper appendix "Calibration-diversity control").

Retained fraction is the mean over subjects of `(unseen_i − 0.5)/(seen_i − 0.5)` on mean percentile
rank, with a paired subject bootstrap for the interval. Regenerate with
`scripts/retained_fraction.py`; `--estimator ratio-of-means` gives the other reading, which differs by
at most 0.2 points at k=500 and k=2000 and by 4 at k=100, where per-subject denominators are noisy.

On Narratives fMRI, under a plain cross-subject design with no pretraining stage, holding out the
evaluation segments leaves roughly a quarter of the above-chance signal; withholding the whole
narrative leaves an interval covering chance.

What survives is real: it reaches mean percentile 0.66–0.71 against 0.5 chance on a narrative no
brain in the training set has heard, and a from-scratch encoder tuned over a learning-rate and
weight-decay grid does not reach it.

**The effect is not confined to the size of the number.** A pre-specified architecture comparison in
our own work favoured a dynamics-aware tokenizer under the standard protocol at both low-calibration
budgets (pool-size-invariant rank advantage +0.043 [+0.009, +0.081] at k=100 and +0.014 [+0.001, +0.026]
at k=500, n=10; the pre-specified ×chance criterion read +8.26 [+2.40, +14.19]; over three pretraining
seeds the k=100 advantage is +0.048 [+0.024, +0.071]; on a rebuild of the same archive the k=100
interval reaches zero, +0.035 [−0.001, +0.069], while k=500 reproduces, +0.018 [+0.006, +0.030]). At that published seed (seed 0), once calibration
and evaluation stop sharing a story the advantage is undetectable at both low budgets with every story
still in the pretraining corpus (+0.013 [−0.011, +0.037]; +0.010 [−0.004, +0.024]), but averaged over
three pretraining seeds the same kind of cell still resolves at k=100 (+0.035 [+0.013, +0.056]), so
story-disjoint calibration alone does not remove the advantage. It is also undetectable in the cell that
additionally withholds story 3 from pretraining (+0.020 [−0.019, +0.055]; −0.006 [−0.022, +0.011]) — that
cell's k=100 interval contains the standard-protocol +0.043, so these cells cannot distinguish a licensed
advantage from an undetected one; it is never significantly positive in any of four narratives at any
budget when the evaluation story is also withheld from pretraining, and at k=2000 all four withheld
narratives reverse (three of four under a Bonferroni correction). Every cell that withholds the story
from pretraining is a single seed whose replicates trained the static encoder only, and no paired
contrast that would attribute the loss of the advantage to pretraining exposure resolves, so we attribute
it to no single change. At seed 0 the static encoder also leads at k=2000 under the standard protocol;
averaged over three seeds that lead is not resolved (+0.001 [−0.006, +0.008]), so whether the top-budget
reversals are leakage is open. The advantage that disappears with the shared stimulus is the low-budget
one. At fixed model the adjacency guard changes rank percentile by under 1e-4, so of the two protocol
changes only the split can carry an effect. Comparisons across protocol conditions are
made in mean percentile rank because the candidate pools differ by up to 7× between conditions; each
stimulus is reported separately rather than pooled, since the cells reuse subjects and stories and a
pooled interval would be too narrow. At this sample size the architecture comparison is sensitive to the split (the architecture×split
interaction at k=500 excludes zero before, not after, multiplicity correction); it is not an
established change in the difference between the encoders.

## What we recommend

1. **Hold the stimulus out of pretraining, not just the subject**, and report both numbers. The gap
   between them is the part of the score that shared exposure explains.
2. **Report a stimulus-derived baseline** — a model given the timing or other stimulus-side features
   your protocol already assumes, and no brain data — alongside every retrieval result. Noise and
   shuffle controls do not substitute: they test whether the recording carries information, not whether
   that information is redundant with what the experimenter supplied.
3. **Compare across conditions with a rank statistic.** Candidate pools change when the protocol
   changes, and both accuracy and ×chance move with pool size at fixed skill — accuracy *rises* as the
   pool shrinks. Mean percentile rank is invariant to pool size (not composition); use it, or report the pool alongside every number.

## The diagnostic

`scripts/stimulus_leakage_check.py` takes a three-column manifest and reports how much of your
evaluation material appears in your fitting partition.

```bash
python scripts/stimulus_leakage_check.py --manifest manifest.csv
python scripts/stimulus_leakage_check.py --manifest manifest.csv --propose-split --out clean.json
```

```csv
subject,stimulus,split
sub-01,pieman,pretrain
sub-27,pieman,test
```

It distinguishes *held-out subjects with shared stimuli* from *no leakage*, which is the distinction
that matters and the one a subject-level split hides. Exit status is 1 when any evaluation stimulus
appears in a fitting split, so it can run in CI.

Status 1 is not a verdict that a result is wrong. For many designs the overlap is unavoidable. It is a
prompt to report both numbers.

## A pool trap in a widely used pipeline

`patches/bm_wer_pool_match.py` applies to `brainmagick`, the reference decoding pipeline. Its
evaluation draws the candidate pool as the first `wer_negatives` entries of a permutation of the
retained queries, with `wer_negatives` defaulting to 10,000 — so the pool is
`min(10_000, N_queries)`. On MEG-MASC the cap does *not* bind: the run retains 7,140 queries, so the
pool is 7,140, a number set by how the split happened to size the test partition rather than by
anything the experimenter chose.

That matters the moment you change the protocol. Moving from the shipped block-level splits to
story-level splits enlarges the test partition enough that the cap does bind and the pool becomes
10,000; chance moves from 0.140% to 0.100% and top-10 accuracy moves with it at unchanged skill, in
the direction that flatters the corrected protocol. Comparing the two numbers directly would attribute
a pool change to the split. The patch pins the query set so both conditions share one pool.

The general form: a configuration knob named as a *ceiling* is a pool-size determinant whenever the
quantity it caps can fall below it. Grep your own logs for what the pool actually was in each run
before comparing them.
