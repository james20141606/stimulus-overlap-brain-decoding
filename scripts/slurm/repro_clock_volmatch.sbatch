#!/bin/bash
#SBATCH --job-name=bci-clockvm
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
# MATCHED-VOLUME CLOCK ARM ("clock B"). Launched 2026-09-12.
#
# WHY. paper_iclr.tex §5 / paper_transfer.tex: "For the decoder that separation is arm B (the
# matched-volume decoder loses 6.45 points under the block split); the matched-volume clock arm is
# the control we have not run." This is the last control the paper names as unrun. One run closes it.
#
# WHAT. Exactly arm B's environment (repro_b_seeds.sbatch / repro_volmatch.sbatch) -- original
# block-level split, train thinned to arm C's volume with BM_TRAIN_FRAC=0.5824 and
# BM_TRAIN_FRAC_SEED=0 (so the retained subset is the SAME one arm B trained on), pool pinned to
# 7,140 -- plus BM_TIMING_ONLY=1, which replaces the MEG input with the word-onset raster (the
# clock). Seed 4801: the 48xx range is unused anywhere (env vars are not in the dora signature, so a
# reused seed would resume another arm's checkpoint).
#
# EXISTING CLOCK ARMS, for the reading:
#   clock A (block split, full volume):  89.52 (R2 14377929), 89.57 (raster_3201), 92.00 (raster_3202)
#   clock C (story split, full volume):  91.79 (storysplit/raster_4201)
#
# PRE-REGISTERED READING (fixed before launch; generator section in scripts/abc_decomposition.py):
#   A's seed range is 89.52-92.00, width 2.48; half of it is 1.24, so the boundary is 88.28.
#   Branch 1: clock B >= 88.28 -> the clock is insensitive to the volume cut; the "control we have
#             not run" clause is replaced by the measurement in both variants.
#   Branch 2: clock B <  88.28 -> the clock DOES lose to volume, hence the story split GAINS at
#             matched volume (C - B > 0); the §5 "the timing arm ... does not drop" sentence is
#             re-scoped to the decomposition (volume loss + split gain), not deleted.
#   Escalate to seeds 4802/4803 ONLY if clock B lands within 1.0 of 88.28.
#   Either branch retires the sentence; neither touches any decoder number.
#
# VOLUME CHECK: arm B realised 120,389 train examples (+1.8% vs 118,310). The raster arm must log the
# same count (same frac, same thinning seed); if it does not, the run is not the matched control.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
unset BM_STORY_SPLIT                   # ORIGINAL block-level split, as arm B
unset BM_STORY_SPLIT_SEED
unset BM_MIXED_SPLIT
export BM_TRAIN_FRAC=0.5824            # arm B's thinning: 118310/203152
export BM_TRAIN_FRAC_SEED=0            # arm B's retained subset
export BM_WER_MAX_SEGMENTS=7140        # the pool, pinned as in every other arm
export BM_TIMING_ONLY=1                # the clock: raster replaces the MEG input
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/clock_volmatch_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/volmatch
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
grep -q BM_TIMING_ONLY bm/dataset.py || { echo "[FAIL] timing-only patch not applied"; exit 1; }
grep -q BM_TRAIN_FRAC bm/events.py bm/dataset.py 2>/dev/null || { echo "[FAIL] train-frac patch not applied"; exit 1; }
grep -q BM_WER_MAX_SEGMENTS bm/wer.py || { echo "[FAIL] pool cap patch not applied"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} BM_TRAIN_FRAC=$BM_TRAIN_FRAC TIMING_ONLY=$BM_TIMING_ONLY date=$(date -Is)"
[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

CHUNKS=$CACHE/clock_volmatch_chunk_count
MAXCHUNKS=10
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted; inspect rather than raising MAXCHUNKS."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_clock_volmatch.sbatch \
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
  echo "[$(date -Is)] === ${arm} seed=${S} matched-volume clock (${left_min:-?} min left) ==="
  "$ENV/bin/dora" run -d seed="$S" >>"$LOGD/${arm}_${S}.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && touch "$LOGD/${arm}_${S}.DONE"
  echo "[$(date -Is)] ${arm} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

run_one raster 4801
DONE=$(ls "$LOGD"/raster_4801.DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/1"
[ "$DONE" -eq 1 ] && touch "$SENT" && echo "[chain] ALL DONE"
echo "[$(date -Is)] chunk complete"
