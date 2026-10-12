#!/bin/bash
#SBATCH --job-name=bci-storysplit
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
# WALL CLOCK: chunking failed BEFORE the WER cap and works AFTER it, which is why this file reversed
# itself. The old reasoning was sound on its own terms -- one epoch cost ~47 min (18 train + 10 valid
# + 19 WER), so a 1h50m chunk died partway through epoch 2, before the epoch boundary where dora writes
# its checkpoint, and every chunk restarted at epoch 1 with no checkpoint.th ever written. What changed
# is the WER stage: capping the query set at 7,140 (below) cut it from ~19 min to ~3 min, so an epoch
# now costs ~29 min and a 1h45m chunk clears three of them with room to spare. Confirmed empirically --
# job 15540582 was culled during epoch 5 and left outputs/xps/45613291/checkpoint.th (153 MB, written
# 07:44, immediately after epoch 4). The pool fix and the wall-clock fix turn out to be the same fix.
#
# THE ~2h CULL IS REAL, AND IT IS GROUP CONTENTION -- NOT A PER-USER CAP.
# Corrected 2026-08-09 (second revision). Job 15540582 was CANCELLED by 0 at 02:17:49 with
# Reason=QOSGrpGRES while it was my only GPU job. Four kills now sit at 02:05:29, 02:15:25, 02:17:49
# and 02:23:19 -- an 18-minute band -- so a ~2h reclamation is real and chunking is needed. That
# much stands, and the chunked architecture below is unchanged.
# WHAT I GOT WRONG, AND THE REPO HAD ALREADY GOT WRONG ONCE. I attributed the kills to a
# two-GPU-per-user cap, having read `sacctmgr show qos` as MaxTRESPU=gres/gpu=2 for QOS QOS_A. It
# reports gres/gpu=24; I dropped the final digit. There is no two-GPU per-user limit, so the rule
# "hold back your other GPU jobs or one will be killed" was wrong and is removed. Read sacctmgr with
# `where name=<qos>` and one field at a time -- this exact field has now been misread twice here.
# WHAT FITS THE EVIDENCE. QOSGrpGRES names a GROUP limit: partition QOS gpuA_partition carries GrpTRES
# gres/gpu=112 across all of ENG-SCHOOL. Jobs pend when that pool is full and get reclaimed under
# sustained contention; when it is quiet they run to their wall clock -- which is why bci-acoustic
# (07:26:54), bci-r2-shuf (06:53:16) and bci-repro-r2 (05:46:21) all COMPLETED far past 2h. You
# cannot tell at submission which regime you are in, so design against ~2h.
# CONSEQUENCE: chunks, not one long run. 1h45m leaves 20 minutes below the earliest observed kill.
#
# MEMORY: job 15508620 died OUT_OF_MEMORY at 500G (MaxRSS 524251404K) after 01:34:45; three earlier
# chunks died at 192G. Raising the ceiling bought eleven minutes and hit the same wall, because this
# is one allocation and not a drift. Both arms died at the identical line, "WER | 450/453", which is
# the last progress print before `torch.cat(estimates_list)` in bm/wer.py. Estimates are
# (N, 768, 361) float32; the story split makes the test partition larger than the train partition
# (118310 | 53933 | 141946), so the WER loop grows from 56 batches to 453 at batch_size=256. The block
# -split run retained 7,140 queries from its 56 batches, so at the same ~50% reject rate 453 batches
# retain ~58,000 -- eight times the arrays, held as lists plus their concatenated copies at once.
# Fixed by capping the WER query set (see BM_WER_MAX_SEGMENTS below), which is also the fix for a much
# more serious problem.
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# THE CORRECTED EXTERNAL NUMBER. Re-run the reproduced reference pipeline with STORY-LEVEL splits.
#
# WHY THIS IS THE HIGHEST-VALUE REMAINING EXPERIMENT. Every number in the paper is currently our own.
# The accepted precedents for evaluation-critique papers all re-run somebody else's published
# comparison and publish the corrected value beside the original; that is what turns a census into a
# re-analysis. We reproduced this pipeline end to end (brain 58.4% top-10, clock 89.5%), so it is the
# one external result we can correct rather than estimate.
#
# WHAT CHANGES. `assign_blocks` hashes each block's uid, and for MEG-MASC that uid is the block's WORD
# SEQUENCE. Blocks from one story therefore hash independently and land in different splits: the model
# trains on parts of a story and is scored on other parts of the same story. Blocks are disjoint --
# which is what "stimulus-disjoint splits" is usually taken to mean -- but the STORY is shared. That is
# exactly the calibration-level overlap our own experiments show can reverse an architecture
# comparison. BM_STORY_SPLIT=1 keeps every story wholly within one split.
#
# DISCLOSE WITH THE RESULT: with four stories the partition is 1 test / 1 valid / 2 train, so the
# training set also SHRINKS relative to the original run. Story-level splitting and reduced training
# data move together here and cannot be separated at four stimuli. The corrected number is therefore a
# lower bound on what the pipeline achieves under a clean split, not a clean estimate of the leakage
# alone -- state it that way.
#
# COMPARE EPOCH-MATCHED, NOT END-TO-END. The block-split run this corrects reached epoch 14, and the
# three seed replicates the paper quotes (58.40 / 57.79 / 57.21% top-10) are early-epoch values --
# `Test Summary | Epoch 3 | wer=42.794%` is 57.21% exactly. Scoring a story-split run at epoch 40
# against a block-split run at epoch 3 would let split level and training budget move together, which
# is the two-variable design this project keeps rebuilding by accident. Both runs log a Test Summary
# per epoch, so the comparison is a curve over epochs 1-14 and the corrected number is read off at a
# matched epoch. Extra epochs here are headroom, not the claim.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF          # torch 2.0.1 in this env rejects expandable_segments
export BM_STORY_SPLIT=1                # <-- the whole point
export BM_STORY_SPLIT_SEED=9           # verified: 1 test / 1 valid / 2 train over the four stories
                                       # (49/49/98 recordings). Most seeds are degenerate at four
                                       # stimuli, so this is a disclosed choice, not a random draw.
# THE CANDIDATE POOL MUST NOT MOVE. This is the reason the OOM was worth stopping over rather than
# routing around with more memory. bm/wer.py builds the candidate pool from the retained queries:
# `perm[:wer_negatives]` with wer_negatives=10_000. The block-split run being corrected logged
# "wer 7140 negatives selected" -- 7,140 < 10,000, so it used ALL of its rank-local queries and its
# pool is 7,140, the figure the paper discloses. The story split accumulates roughly 58,000 queries, so
# the 10,000 cap WOULD have bound and the corrected number would have been scored against a 10,000-
# candidate pool. Chance moves 0.140% -> 0.100% and top-10 moves with it at unchanged skill. A paper
# whose subject is that this metric is not pool-invariant cannot publish an external correction that
# silently changes the pool. Capping the query set at 7,140 makes the pool identical by construction,
# and drops the peak from ~514 GB to ~32 GB as a side effect. Per rank, which is the right scope:
# 7,140 is itself rank-local and both runs use --gres=gpu:2. See patches/bm_wer_pool_match.py.
export BM_WER_MAX_SEGMENTS=7140
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/storysplit_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/storysplit
# TMPDIR MUST BE NODE-LOCAL. multiprocessing removes its temp dir at process exit; on NFS that hits
# silly-rename (.nfsXXXX "Device or resource busy"), rmtree raises, and the process exits non-zero
# even though TRAINING SUCCEEDED -- the last chunk reached 35 epochs with valid loss still improving
# and was recorded as a failure purely on this shutdown path. Only the multiprocessing scratch space
# moves; everything else stays on /scratch.
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} BM_STORY_SPLIT=$BM_STORY_SPLIT date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

# --- self-resubmitting chain -------------------------------------------------------------------
# Queue the successor NOW, before doing any work, so a mid-epoch cull cannot stop the chain.
# afterany (not afterok) because a culled chunk exits non-zero and is exactly the case we resume from.
# The dependency also keeps the successor PENDING while this chunk runs, which matters: two chunks
# running at once would duplicate work and fight over the same dora checkpoint, not because of any
# per-user GPU cap (there is none at this scale -- MaxTRESPU is gres/gpu=24).
CHUNKS=$CACHE/storysplit_chunk_count
MAXCHUNKS=40                      # ~70h of chunks; a guard against a chain that fails instantly and
                                  # resubmits forever. 40 epochs at ~29 min needs ~14 chunks per arm.
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted. The chain is not converging -- inspect the logs rather"
  echo "[chain] than raising MAXCHUNKS, because 40 chunks is already well past what 2 arms x 40"
  echo "[chain] epochs at ~29 min should need."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_storysplit.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT (afterany:${SLURM_JOB_ID})" \
                 || echo "[chain] WARNING: could not queue successor"
fi

run_one () {
  local arm=$1 S=$2
  [ -f "$LOGD/${arm}_${S}.DONE" ] && { echo "[skip] ${arm} ${S} finished"; return 0; }

  # DO NOT START AN ARM THAT CANNOT REACH A CHECKPOINT. dora writes checkpoint.th in commit(), which
  # runs AFTER the epoch's test stage -- so an epoch that trains and validates but is killed before
  # commit leaves nothing behind and the next chunk starts that arm at epoch 1 again.
  # This bit us on 2026-08-09: chunk 15551567 ran the brain arm first, the raster arm inherited only
  # 37 minutes, completed train (20 min) and valid (14 min), and was killed at 15:43:37 three minutes
  # into its test stage. A whole epoch of compute, discarded, and the log showed "Epoch 1" twice with
  # DIFFERENT losses (1.2304 then 1.1995) -- the signature of a real restart rather than dora's replay,
  # which repeats losses exactly. Two more chunk boundaries would have looked like a stuck run.
  # A raster epoch costs ~20 min train + ~14 min valid + ~4 min test, so demand 45 minutes of headroom.
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2}
    NF==2 {print $1}
    NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${arm}: only ${left_min} min left in this chunk, need 45 to reach a checkpoint."
    echo "[skip] Leaving it for the successor rather than burning an epoch that cannot commit."
    return 0
  fi

  echo "[$(date -Is)] === ${arm} seed=${S} (story-level split, ${left_min:-?} min of chunk left) ==="
  if [ "$arm" = raster ]; then export BM_TIMING_ONLY=1; else unset BM_TIMING_ONLY; fi
  "$ENV/bin/dora" run -d seed="$S" >>"$LOGD/${arm}_${S}.log" 2>&1
  local rc=$?                       # must be the next line; a $(...) here would clobber $?
  unset BM_TIMING_ONLY
  # DONE ON rc==0 ONLY. The previous rule also accepted "the log contains a Test Summary", which was
  # safe for a single long run and is CATASTROPHIC for a chunked one: dora logs a Test Summary EVERY
  # epoch, so chunk 1 would mark the arm finished after ~3 epochs and chunk 2 would skip it entirely.
  # Training would stop at a fifth of the budget and nothing in the output would say so -- the run
  # would look complete and report a number from an undertrained model. That fallback existed for a
  # TMPDIR-on-NFS shutdown error which is now fixed at the source (TMPDIR is node-local, below), so it
  # is no longer earning its keep. If dora ever again exits non-zero after genuinely finishing, the
  # chain will loop and the chunk budget will stop it -- loud and inspectable, which is the failure
  # mode to prefer over a silent truncation.
  if [ "$rc" -eq 0 ]; then
    touch "$LOGD/${arm}_${S}.DONE"
  fi
  echo "[$(date -Is)] ${arm} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

run_one brain  4101
run_one raster 4201

DONE=$(ls "$LOGD"/*.DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/2"
[ "$DONE" -eq 2 ] && touch "$SENT" && echo "[chain] ALL DONE"
echo "[$(date -Is)] chunk complete"
