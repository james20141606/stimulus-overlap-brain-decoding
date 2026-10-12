#!/bin/bash
#SBATCH --job-name=bci-bseeds
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# SEED REPLICATION OF ARM B (block split, train thinned to arm C's volume).
#
# WHY. Round-5 external review (8.8, strong accept): the one remaining computable gap is that the
# A/B/C decomposition is single-run -- "the paper still cannot estimate the volume-versus-exposure
# decomposition with inferential confidence". Arm A already has three seeds (58.41/57.79/57.21,
# the 1.2-point spread the paper quotes); B and C have one each (B 4301 = 51.96, C 4101 = 47.69).
# This chain adds B at seeds 4302 and 4303; repro_c_seeds.sbatch adds C at 4102 and 4103. The two
# chains run in parallel (different arms, different sentinels -- no shared .DONE race).
#
# ENV = exactly repro_volmatch.sbatch (the arm-B definition), only the seeds differ:
#   no story split; BM_TRAIN_FRAC=0.5824 (118310/203152, achieved 120389 = +1.8% vs target);
#   BM_TRAIN_FRAC_SEED=0 so the SAME retained-train subset across model seeds -- the seed must move
#   only optimization randomness, not the data; BM_WER_MAX_SEGMENTS=7140 pins the pool.
#
# SEED RANGES ARE LOAD-BEARING: env vars do not enter the dora signature, so arm B seed S and arm C
# seed S would share a checkpoint directory. B stays in 43xx, C in 41xx. 4302/4303 are unused.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT                   # arm B is the ORIGINAL block-level split
unset BM_STORY_SPLIT_SEED
unset BM_MIXED_SPLIT
export BM_TRAIN_FRAC=0.5824            # 118310/203152; matches arm C's training volume
export BM_TRAIN_FRAC_SEED=0
export BM_WER_MAX_SEGMENTS=7140        # the pool, pinned identically to arms A and C

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/b_seeds_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/volmatch
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} BM_TRAIN_FRAC=$BM_TRAIN_FRAC date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

# Chunked chain: the ~2h reclamation under group GPU contention is real; queue the successor
# BEFORE doing any work (see repro_storysplit.sbatch's header for the history).
CHUNKS=$CACHE/b_seeds_chunk_count
MAXCHUNKS=40
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted; inspect rather than raising MAXCHUNKS."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_b_seeds.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local arm=$1 S=$2
  [ -f "$LOGD/${arm}_${S}.DONE" ] && { echo "[skip] ${arm} ${S} finished"; return 0; }
  # dora commits AFTER the epoch's test stage; an epoch killed before commit leaves nothing.
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
  # rc==0 ONLY: dora logs a Test Summary every improving epoch; a log-based rule truncates training.
  [ "$rc" -eq 0 ] && touch "$LOGD/${arm}_${S}.DONE"
  echo "[$(date -Is)] ${arm} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

run_one brain 4302
run_one brain 4303

DONE=$(ls "$LOGD"/brain_430[23].DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/2"
[ "$DONE" -eq 2 ] && touch "$SENT" && echo "[chain] ALL DONE"
echo "[$(date -Is)] chunk complete"
