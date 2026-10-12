#!/bin/bash
#SBATCH --job-name=bci-dose
#SBATCH --output=$SCRATCH/project/logs/m1/mixed/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/mixed/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# DESIGN 3 (experiment scout, 2026-08-19): the EXTERNAL DOSE LADDER. Converts §6's "full-dose
# extrapolation is point-estimate algebra, not resolved" into a measurement, and completes a
# four-point external dose-response with the existing C' (dose 0, n=3) and D (dose 0.59, n=5):
#
#     D_full  seed 4501  BM_TRAIN_FRAC_EVALSTORY=1.0   BM_TRAIN_FRAC=<CAL_FULL>   dose 1.0
#     D_mid   seed 4502  BM_TRAIN_FRAC_EVALSTORY=0.30  BM_TRAIN_FRAC=<CAL_MID>    dose 0.30
#
# CALIBRATION (CPU dry runs, logs/m1/mixed/dryrun_Dfull*.log / dryrun_Dmid*.log):
#   D_full: 0.2896 -> 119,134 (+0.70%); calibrated 0.2828 -> 118,385 (+0.06%). ACCEPTED.
#   D_mid:  0.7869 -> 116,798 (-1.28%); calibrated 0.7997 -> 118,440 (+0.11%). ACCEPTED.
#   Both an order of magnitude tighter than the accepted D precedent (+1.0%).
#
# PRE-REGISTERED READING (scout report, fixed before launch):
#   D_full - C'mean < 6.45  -> §6's price-vs-leak upgrades from extrapolation to measurement.
#   D_full - C'mean >= 6.45 -> the §6 pricing sentence and the abstract DEMOTE to
#                              parity-or-reversal and must be rewritten.
#   D_mid out of order (>= the 0.59-dose mean or below C'mean) -> the external curve is
#                              non-monotone; the monotone-understates argument loses its premise
#                              and retreats to the internal sweep.
#   Escalate D_full to 3 seeds (+2 runs) ONLY if its point lands within ~1.2 of 6.45.
#
# CONFOUNDS (disclosed in the same words as the existing arms): dose moves jointly with
# displacement (at full dose ~71% of training is the evaluation narrative); validation stays
# cable_spool_fort but its relative training weight falls as dose rises -- repeat the free
# selection check (selected epoch == best logged eval) per arm and report it; volume pinned <=
# target per precedent (err low). Pools: BM_WER_POOL_SEED=0 from birth; pool membership remains
# run-dependent (fitted-scaler mechanism), so cross-arm reads follow the measured-draw-noise
# framing already in the paper.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT BM_STORY_SPLIT_SEED
export BM_WER_MAX_SEGMENTS=7140
export BM_WER_POOL_SEED=0
CAL_FULL=0.2828   # dry-run round 2: 118,385 train (+0.06% vs 118,310)
CAL_MID=0.7997    # dry-run round 2: 118,440 train (+0.11%)

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/dose_ladder_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/mixed
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
grep -q BM_TRAIN_FRAC_EVALSTORY bm/events.py || { echo "[FAIL] evalstory patch not applied"; exit 1; }
grep -q BM_WER_POOL_SEED bm/wer.py || { echo "[FAIL] pool-seed patch not applied"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} CAL_FULL=$CAL_FULL CAL_MID=$CAL_MID date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

CHUNKS=$CACHE/dose_ladder_chunk_count
MAXCHUNKS=30
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_dose_ladder.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local name=$1 S=$2 evfrac=$3 frac=$4
  [ -f "$LOGD/${name}_${S}.DONE" ] && { echo "[skip] ${name} ${S} finished"; return 0; }
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${name}: only ${left_min} min left. Leaving to successor."; return 0
  fi
  echo "[$(date -Is)] === ${name} seed=${S} EVALSTORY=${evfrac} FRAC=${frac} (${left_min:-?} min left) ==="
  (
    export BM_MIXED_SPLIT=D BM_TRAIN_FRAC_EVALSTORY="$evfrac" BM_TRAIN_FRAC="$frac" BM_TRAIN_FRAC_SEED=0
    exec "$ENV/bin/dora" run -d seed="$S"
  ) >>"$LOGD/${name}_${S}.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && touch "$LOGD/${name}_${S}.DONE"
  echo "[$(date -Is)] ${name} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${name}_${S}.log"
  return 0
}

run_one Dfull 4501 1.0  "$CAL_FULL"
run_one Dmid  4502 0.30 "$CAL_MID"

DONE=$(ls "$LOGD"/Dfull_4501.DONE "$LOGD"/Dmid_4502.DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/2"
[ "$DONE" -eq 2 ] && touch "$SENT" && echo "[chain] ALL DONE"
