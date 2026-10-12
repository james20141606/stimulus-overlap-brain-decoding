#!/bin/bash
#SBATCH --job-name=bci-volmatch
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# THE VOLUME CONTROL FOR THE CORRECTED EXTERNAL NUMBER.
#
# WHY. Our corrected external result is 58.41% -> 47.69% guarded top-10 under story-level splits, and
# it is confounded in a way the paper discloses: with four narratives the corrected partition is
# 1 test / 1 valid / 2 train, so training volume falls from 203,152 to 118,310 examples (58%) at the
# same time. An external reviewer named exactly this as the highest-value remaining experiment --
# "a training-volume-matched rerun of the reference decoder that isolates story overlap from the 42%
# reduction in training examples". This is that rerun.
#
# THE DESIGN. Three arms, one candidate pool:
#     A  block split, full train   203,152   58.41%   (have, job 14260550)
#     B  block split, train @ 58%  ~118,310  ???      (this job)
#     C  story split, full train   118,310   47.69%   (have, chain 15540582->)
# B shares VOLUME with C and SPLIT with A, so A-B is the volume contribution and B-C is what the
# shared story is worth once volume is held. If B lands near 58.41%, volume is not carrying the drop.
# If B falls most of the way to 47.69%, it is, and our headline needs restating.
#
# WHAT IS DELIBERATELY NOT MATCHED, and must be disclosed with the result: B trains on blocks drawn
# from all four stories, C on two whole stories. B matches C in volume but not in stimulus
# composition -- the same volume-matched / composition-matched distinction the paper already draws for
# its own MB experiments, where the two designs have opposite confounds and bracket the answer.
# B bounds the volume contribution; it does not make the comparison composition-free.
#
# WHY NOT remove_ratio. bm/wer.py sets the candidate pool to min(wer_negatives, N_queries) and the
# block split retains exactly 7,140 queries. remove_ratio shrinks TEST as well as train, which would
# drop the retained queries to ~4,100, un-bind the BM_WER_MAX_SEGMENTS cap, and silently change the
# pool -- the precise error this paper is about. patches/bm_train_subsample.py thins the train split
# only, so all three arms keep one pool. Verify in the log: "wer 7140 negatives selected".
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT                   # arm B is the ORIGINAL block-level split
unset BM_STORY_SPLIT_SEED
export BM_TRAIN_FRAC=0.5824            # 118310/203152; matches arm C's training volume
export BM_TRAIN_FRAC_SEED=0
export BM_WER_MAX_SEGMENTS=7140        # the pool, pinned identically to arms A and C

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/volmatch_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/volmatch
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} BM_TRAIN_FRAC=$BM_TRAIN_FRAC date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

# Chunked chain, same structure as repro_storysplit: the ~2h reclamation under group GPU contention is
# real (see that file's header), so queue the successor before doing any work.
CHUNKS=$CACHE/volmatch_chunk_count
MAXCHUNKS=20
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted; inspect rather than raising MAXCHUNKS."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_volmatch.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local arm=$1 S=$2
  [ -f "$LOGD/${arm}_${S}.DONE" ] && { echo "[skip] ${arm} ${S} finished"; return 0; }
  # Do not start an arm that cannot reach a checkpoint: dora commits AFTER the epoch's test stage, so
  # an epoch killed before commit leaves nothing and the next chunk restarts at epoch 1. See
  # repro_storysplit.sbatch, where this cost a full epoch on 2026-08-09.
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${arm}: only ${left_min} min left, need 45 to reach a checkpoint. Leaving to successor."
    return 0
  fi
  echo "[$(date -Is)] === ${arm} seed=${S} volume-matched (${left_min:-?} min left) ==="
  "$ENV/bin/dora" run -d seed="$S" >>"$LOGD/${arm}_${S}.log" 2>&1
  local rc=$?
  # rc==0 ONLY: dora logs a Test Summary every improving epoch, so a log-based DONE rule would mark
  # the arm finished after the first chunk and silently truncate training.
  [ "$rc" -eq 0 ] && touch "$LOGD/${arm}_${S}.DONE"
  echo "[$(date -Is)] ${arm} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

run_one brain 4301

DONE=$(ls "$LOGD"/*.DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/1"
[ "$DONE" -eq 1 ] && touch "$SENT" && echo "[chain] ALL DONE"
echo "[$(date -Is)] chunk complete"
