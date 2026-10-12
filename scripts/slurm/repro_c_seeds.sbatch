#!/bin/bash
#SBATCH --job-name=bci-cseeds
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# SEED REPLICATION OF ARM C (story-level split, full train). Companion of repro_b_seeds.sbatch;
# see that header for the WHY (round-5 review: A/B/C decomposition single-run is the last
# computable gap). Arm C has one run (4101 = 47.69%); this adds seeds 4102 and 4103.
#
# ENV = exactly repro_storysplit.sbatch (the arm-C definition), only the seeds differ:
#   BM_STORY_SPLIT=1 with BM_STORY_SPLIT_SEED=9 (the verified non-degenerate 1 test / 1 valid /
#   2 train partition -- FIXED across model seeds so the seed moves only optimization randomness);
#   no train thinning; BM_WER_MAX_SEGMENTS=7140 pins the pool (the cap BINDS here: the story split
#   retains 115,874 queries, and letting the pool float was the original pool bug).
#
# SEED RANGES ARE LOAD-BEARING: env vars do not enter the dora signature -- C stays in 41xx,
# B in 43xx. 4102/4103 are unused (4201 is the raster arm, distinct).
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

export BM_STORY_SPLIT=1
export BM_STORY_SPLIT_SEED=9
unset BM_TRAIN_FRAC
unset BM_TRAIN_FRAC_SEED
unset BM_MIXED_SPLIT
export BM_WER_MAX_SEGMENTS=7140

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/c_seeds_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/storysplit
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} BM_STORY_SPLIT=$BM_STORY_SPLIT date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

CHUNKS=$CACHE/c_seeds_chunk_count
MAXCHUNKS=40
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted; inspect rather than raising MAXCHUNKS."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_c_seeds.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local arm=$1 S=$2
  [ -f "$LOGD/${arm}_${S}.DONE" ] && { echo "[skip] ${arm} ${S} finished"; return 0; }
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${arm}: only ${left_min} min left, need 45 to reach a checkpoint. Leaving to successor."
    return 0
  fi
  echo "[$(date -Is)] === ${arm} seed=${S} (story-level split, ${left_min:-?} min left) ==="
  "$ENV/bin/dora" run -d seed="$S" >>"$LOGD/${arm}_${S}.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && touch "$LOGD/${arm}_${S}.DONE"
  echo "[$(date -Is)] ${arm} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

run_one brain 4102
run_one brain 4103

DONE=$(ls "$LOGD"/brain_410[23].DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/2"
[ "$DONE" -eq 2 ] && touch "$SENT" && echo "[chain] ALL DONE"
echo "[$(date -Is)] chunk complete"
