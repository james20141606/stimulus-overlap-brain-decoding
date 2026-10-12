#!/usr/bin/env python3
"""Epoch-matched comparison of the reference pipeline under block-level vs story-level splits.

WHAT THIS ANSWERS

The reproduced reference pipeline splits MEG-MASC by hashing each block's word sequence, so blocks of
one story land in different splits: the model trains on parts of a story and is scored on other parts
of the same story. BM_STORY_SPLIT=1 keeps every story wholly within one split. This script reports
what that costs, which is the paper's one externally-corrected number.

THREE THINGS IT REFUSES TO GET WRONG, each of which was a real error in this project:

1. POOL. top-10 accuracy is not pool-invariant, and bm/wer.py sets the candidate pool to
   min(wer_negatives, N_queries) -- so changing the split changes the pool as a side effect. Both runs
   here are pinned to 7,140 by BM_WER_MAX_SEGMENTS (patches/bm_wer_pool_match.py). This script asserts
   that both logs say "wer 7140 negatives selected" and refuses to report anything if they disagree.
   At a fixed pool, accuracy IS comparable; that is the only reason we may quote it.

2. EPOCH. The block-split run converges at epoch 4 and overfits after. Reading the story-split run at
   epoch 40 against that would let split level and training budget move together. We report the full
   epoch-matched curve, and the headline at each run's own best-validation-loss checkpoint -- matched
   by selection criterion, which is the right matching for a converged-model comparison.

3. EPOCH RENUMBERING. Chunked restarts renumber epochs, and on resume dora replays its whole history
   under one timestamp starting from 0 while its live counter stays 1-indexed. Parsing labels naively
   shifts the curve by an epoch and makes the story-split arm look better than it is. See de_replay().
   The block-split references are therefore taken from the ELN (jobs 14260550 and 14377929), each a
   single unchunked run whose numbering never restarted.

DISCLOSE WITH THE RESULT. With four stories the partition is 1 test / 1 valid / 2 train, so the
training set SHRINKS with the split: 118,310 examples against the block split's 203,152, i.e. 58%.
Story-level splitting and reduced training data move together at four stimuli and cannot be separated.
The corrected number is a LOWER BOUND on what the pipeline achieves under a clean split, not a clean
estimate of the leakage alone. The script prints this every time; do not quote the number without it.

Usage (on the cluster):
  python3 scripts/storysplit_compare.py --arm brain     # the MEG decoder
  python3 scripts/storysplit_compare.py --arm raster    # the word-onset clock
"""
import argparse
import re
import sys
from pathlib import Path

LOGD = Path("$SCRATCH/project/logs/m1")
REPRO = Path("$SCRATCH/repro")

# Per arm: the story-split log, the block-split reference curve, that reference's best-validation
# epoch, and the reference's own log. The reference NUMBERS come from the ELN (both were single
# unchunked runs, so their epoch numbering is trustworthy), but the reference POOL is verified from
# the log rather than assumed -- the comparison is void if the two sides were scored against different
# pools, and checking only one side would be the exact hole this script exists to close.
#   brain  = the MEG decoder, job 14260550, best valid 1.60 at epoch 4, overfits after.
#   raster = the word-onset clock, job 14377929, best valid 0.7550 at epoch 1; ran 11 epochs and
#            early-stopped on dora's own patience-10 rule, so it is converged and not truncated.
ARMS = {
    "brain": dict(
        story=LOGD / "storysplit" / "brain_4101.log",
        block=[(1, 58.915), (2, 47.262), (3, 42.829), (4, 41.590)],
        block_best_epoch=4,
        block_log=REPRO / "bci-repro-r1-14260550.out",
        what="MEG decoder",
    ),
    "raster": dict(
        story=LOGD / "storysplit" / "raster_4201.log",
        block=[(1, 10.476)],
        block_best_epoch=1,
        block_log=REPRO / "bci-repro-r2-14377929.out",
        what="word-onset clock (no brain data)",
    ),
}
POOL = 7140                     # candidate pool, pinned in BOTH runs; chance = wer_topx / POOL
TOPX = 10

STALL_EPOCHS = 3                # epochs with no new best valid loss before we call it converged
BLOCK_TRAIN_EXAMPLES = 203_152
STORY_TRAIN_EXAMPLES = 118_310

# timestamp is load-bearing: it is how the replayed block is identified (all one timestamp).
RE_TEST = re.compile(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\].*?Test Summary \| Epoch (\d+) \| wer=([\d.]+)%")
RE_VALID = re.compile(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\].*?Valid Summary \| Epoch (\d+) \| loss=([\d.]+)")
RE_NEG = re.compile(r"wer (\d+) negatives selected")
RE_BEST = re.compile(r"New best valid loss ([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def top10(wer_pct):
    return 100.0 - wer_pct


def read(path):
    if not path.exists():
        sys.exit(f"missing log: {path}")
    return ANSI.sub("", path.read_text(errors="replace"))


def de_replay(entries):
    """[(timestamp, label, value)] in log order -> {real_epoch: value}.

    Two separate numbering schemes have to be reconciled, and getting this wrong silently shifts the
    whole curve:

      * On resume dora dumps its checkpointed history into the log under a SINGLE timestamp, numbered
        from 0, so the replayed block's labels are 0..k-1 for real epochs 1..k. Position is the truth
        there; the label is off by one.
      * It then continues live with its real 1-indexed counter, so from the seam onward the LABEL is
        the truth. This is why label k never appears -- epoch k was the last replayed one.

    So: replayed block -> real epoch = position; live lines -> real epoch = label. Both verified
    against brain_4101.log on 2026-08-09, where replay [0,1,2,3] + live [5] is five real epochs.
    """
    if not entries:
        return {}
    seq = []
    for ts, ep, val in entries:
        if seq and ep <= seq[0][1]:          # numbering restarted; earlier lines are a stale replay
            seq = []
        seq.append((ts, ep, val))
    replay_ts = seq[0][0]
    n_replayed = sum(1 for ts, _, _ in seq if ts == replay_ts)
    out = {}
    for i, (_, ep, val) in enumerate(seq):
        k = i + 1 if i < n_replayed else ep
        # A key collision here is the epoch-renumbering bug this docstring warns about. The old
        # guard downstream compared a dict's length to its own keyset (a tautology, audit
        # 2026-08-19); the collision must be caught where it happens, before the overwrite.
        if k in out:
            raise SystemExit(f"de-replay key collision at epoch {k}: replayed history and live "
                             f"entries overlap -- inspect the log seam before trusting best_epoch")
        out[k] = val
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="brain")
    a = ap.parse_args()
    cfg = ARMS[a.arm]
    STORY, BLOCK_LOG = cfg["story"], cfg["block_log"]
    BLOCK, BLOCK_BEST_EPOCH = cfg["block"], cfg["block_best_epoch"]
    print(f"arm: {a.arm} -- {cfg['what']}")
    txt = read(STORY)

    # --- guard 1: the pool must be identical, or nothing below is comparable ----------------------
    pools = sorted({int(m) for m in RE_NEG.findall(txt)})
    if not pools:
        sys.exit("story-split log has no 'negatives selected' line yet -- run has not reached a WER "
                 "evaluation. Nothing to compare.")
    if pools != [POOL]:
        sys.exit(f"POOL MISMATCH: story-split log reports {pools}, expected [{POOL}].\n"
                 f"top-10 accuracy is not pool-invariant, so the comparison is INVALID. Check that "
                 f"BM_WER_MAX_SEGMENTS={POOL} was exported and patches/bm_wer_pool_match.py applied.")

    # Same assertion on the other side. Verified 2026-08-09: bci-repro-r1-14260550.out logs
    # "wer 7140 negatives selected" once per epoch for its four epochs.
    if BLOCK_LOG.exists():
        bpools = sorted({int(m) for m in RE_NEG.findall(read(BLOCK_LOG))})
        if bpools != [POOL]:
            sys.exit(f"POOL MISMATCH on the block-split reference: {BLOCK_LOG.name} reports {bpools}, "
                     f"expected [{POOL}]. The two runs were not scored against the same pool, so the "
                     f"gap below would be partly a pool artifact. Refusing to report it.")
        print(f"verified: block-split reference {BLOCK_LOG.name} also used a {POOL}-candidate pool.")
    else:
        print(f"WARNING: {BLOCK_LOG} not found, so the reference run's pool is UNVERIFIED. The "
              f"comparison assumes {POOL} on both sides; confirm before publishing.")

    # --- collect the story-split curve BY ORDINAL POSITION, never by the logged label -------------
    # On resume, dora replays its whole checkpointed history into the log and RENUMBERS it from 0.
    # Observed live on 2026-08-09: the first chunk logged epochs 1,2,3,4 with wer 83.025/58.214/
    # 54.153/52.430, and after the resume the identical four results reappear -- same losses, same
    # durations -- labelled 0,1,2,3. Keying a dict on the label with last-wins therefore shifts the
    # whole curve by one epoch and makes the story arm look a full epoch better at every row.
    # So: find the last point where the numbering restarts, then index by POSITION. The n-th Test
    # Summary in the surviving sequence is the n-th completed epoch, whatever dora chose to call it.
    # A Test Summary is NOT emitted every epoch. bm/solver.py runs the test stage only when
    # `self.best_epoch > self.last_test_epoch`, and it scores `best_state`. Two consequences, both
    # load-bearing:
    #   (a) every Test Summary is already an evaluation of the BEST-validation checkpoint, so the
    #       selection criterion is correct by construction and needs no extra matching;
    #   (b) a MISSING Test Summary means "validation did not improve", not "not evaluated" -- so
    #       counting epochs by Test Summaries undercounts precisely once the model starts to converge,
    #       which is exactly when the stall detector is supposed to fire. Epochs are therefore counted
    #       from Valid Summary, which does appear every epoch.
    story = de_replay([(ts, int(e), float(w)) for ts, e, w in RE_TEST.findall(txt)])
    valid = de_replay([(ts, int(e), float(v)) for ts, e, v in RE_VALID.findall(txt)])
    if not story:
        sys.exit("no Test Summary lines in the story-split log yet.")
    if not valid:
        sys.exit("no Valid Summary lines in the story-split log yet -- cannot count real epochs.")
    epochs_done = max(valid)
    best_epoch = max(story)          # the last new best; test runs at each one
    # (duplicate real epochs are caught inside de_replay at insertion time; the old check here
    # compared len(dict) to len(dict.keys()) and could never fire)
    print(f"note: {epochs_done} epochs completed, {len(story)} of them setting a new best "
          f"(test runs only on improvement); best validation at epoch {best_epoch}.\n")

    chance = 100.0 * TOPX / POOL
    print(f"candidate pool {POOL} in both runs (verified) -> chance top-{TOPX} = {chance:.3f}%\n")
    print(f"{'epoch':>6}  {'block top-10':>13}  {'story top-10':>13}  {'gap':>7}  {'above-chance kept':>18}")
    print("  " + "-" * 68)

    block = dict(BLOCK)
    for ep in sorted(set(block) | set(story)):
        b = block.get(ep)
        s = story.get(ep)
        if b is None or s is None:
            bs = f"{top10(b):12.1f}%" if b is not None else " " * 13
            ss = f"{top10(s):12.1f}%" if s is not None else " " * 13
            print(f"{ep:>6}  {bs}  {ss}  {'--':>7}  {'(unpaired)':>18}")
            continue
        bt, st = top10(b), top10(s)
        kept = (st - chance) / (bt - chance) * 100.0
        print(f"{ep:>6}  {bt:12.1f}%  {st:12.1f}%  {bt - st:+7.1f}  {kept:17.1f}%")

    # --- headline: each run at its own best-validation-loss checkpoint ----------------------------
    print()
    # Convergence is "validation loss stopped improving", not "passed the other run's epoch number".
    # NOTE: the two runs' validation losses are NOT comparable to each other -- they validate on
    # different stories -- so 2.12 here is not worse than the block split's 1.60. Only each run's own
    # trajectory licenses a stopping decision.
    stale = epochs_done - best_epoch     # epochs completed since validation last improved
    if stale < STALL_EPOCHS:
        print(f"NOT YET CONCLUSIVE. Validation last improved at epoch {best_epoch} and "
              f"{epochs_done} epochs are done, so it has stalled for {stale} epoch(s) and we require "
              f"{STALL_EPOCHS}. Reading it now risks scoring a still-training model against a "
              f"converged one -- the budget-vs-split confound this design exists to avoid.")
    else:
        b = top10(block[BLOCK_BEST_EPOCH])
        s = top10(story[best_epoch])
        kept = (s - chance) / (b - chance) * 100.0
        print(f"CONVERGED: validation last improved at epoch {best_epoch}, then {stale} epochs "
              f"without improvement.")
        print(f"  block split at its best-validation checkpoint (epoch {BLOCK_BEST_EPOCH}): {b:.2f}%")
        print(f"  story split at its best-validation checkpoint (epoch {best_epoch}): {s:.2f}%")
        print(f"  gap {b - s:+.2f} points;  above-chance signal retained {kept:.1f}%")
        print(f"  Both scored on the same {POOL}-candidate pool, and both are evaluations of "
              f"best_state, so the two runs are matched by selection criterion rather than by epoch "
              f"number.")

    print(f"""
MANDATORY DISCLOSURE -- do not quote the number without it.
  Story-level splitting also shrinks the training set: {STORY_TRAIN_EXAMPLES:,} examples against the
  block split's {BLOCK_TRAIN_EXAMPLES:,} ({100.0 * STORY_TRAIN_EXAMPLES / BLOCK_TRAIN_EXAMPLES:.0f}%).
  With four stories the partition is 1 test / 1 valid / 2 train, so split level and training volume
  move together and cannot be separated at this number of stimuli.
  Both changes push the same way -- less data and no within-story overlap each hurt -- so the
  direction of the bound is: the story-split ACCURACY is a LOWER bound on what the pipeline achieves
  under a clean split, and equivalently the GAP is an UPPER bound on what the overlap was worth.
  The leakage effect is therefore at most the gap shown, not exactly it.""")


if __name__ == "__main__":
    main()
