#!/bin/bash
#SBATCH --job-name=bci-dreal
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# DESIGN 2 (experiment scout, 2026-08-19): two further D arms under NEW REALIZATIONS of the
# thinning draw. Deletes "one shared realization of the thinning draw, so the spread prices
# optimization noise, not the thinning lottery", and -- if separation holds -- tightens the exact
# permutation floor from 1/C(6,3)=0.05 to 1/C(8,5)~0.018 (5 D vs 3 C').
#
# ENV = the D arm of repro_mixed_seeds.sbatch exactly, except BM_TRAIN_FRAC_SEED, which IS the
# experimental variable here: seed 4407 runs realization 1, seed 4408 realization 2 (the existing
# three D runs all share realization 0). Model seed and realization seed move together BY
# CONSTRUCTION -- env vars do not enter the dora signature, so a same-model-seed/new-realization
# run would collide with an existing checkpoint dir; the paper must say the two variances are
# jointly priced, not decomposed. The kept-count per recording is round(len*frac), seed-invariant,
# so the realized 0.59 dose and the 119,502 volume are IDENTICAL across realizations; only WHICH
# blocks vary.
#
# PRE-REGISTERED READING: both new D above every C' (scored under the design-1 pinned pool) ->
# the realization sentence is deleted, permutation p tightens to 0.018, spread prices optimization
# + lottery jointly. Any new D at or below the best C' -> separation demotes to majority ordering
# and the thinning lottery enters Limits with its measured spread.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT BM_STORY_SPLIT_SEED
export BM_WER_MAX_SEGMENTS=7140
export BM_WER_POOL_SEED=0            # score under the common pinned pool from birth (design 1)
D_FRAC=0.5657

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/d_realizations_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/mixed
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
grep -q BM_WER_POOL_SEED bm/wer.py || { echo "[FAIL] pool-seed patch not applied"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} D_FRAC=$D_FRAC date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

CHUNKS=$CACHE/d_realizations_chunk_count
MAXCHUNKS=30
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_d_realizations.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local S=$1 REAL=$2
  [ -f "$LOGD/D_${S}.DONE" ] && { echo "[skip] D ${S} finished"; return 0; }
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] D ${S}: only ${left_min} min left. Leaving to successor."; return 0
  fi
  echo "[$(date -Is)] === D seed=${S} BM_TRAIN_FRAC_SEED=${REAL} (${left_min:-?} min left) ==="
  (
    export BM_MIXED_SPLIT=D BM_TRAIN_FRAC="$D_FRAC" BM_TRAIN_FRAC_SEED="$REAL"
    exec "$ENV/bin/dora" run -d seed="$S"
  ) >>"$LOGD/D_${S}.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && touch "$LOGD/D_${S}.DONE"
  echo "[$(date -Is)] D ${S} dora_rc=${rc}"
  tail -3 "$LOGD/D_${S}.log"
  return 0
}

run_one 4407 1
run_one 4408 2

DONE=$(ls "$LOGD"/D_440[78].DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/2"
[ "$DONE" -eq 2 ] && touch "$SENT" && echo "[chain] ALL DONE"
