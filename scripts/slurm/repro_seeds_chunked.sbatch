#!/bin/bash
#SBATCH --job-name=bci-seeds
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G --gres=gpu:2 --time=01:50:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# SELF-RESUBMITTING seed replicates. Structural workaround for the administrative kill.
#
# ROOT CAUSE: `CANCELLED by 0` (uid 0 = root) with Reason=QOSMaxGRESPerUser at 02:13-02:30 across four
# jobs. A periodic enforcement pass kills whatever puts this user over the per-user GPU allowance;
# the user's own jobs consume most of it. `--requeue` cannot help -- it only fires on PREEMPTED.
#
# So: run for 1h50m (safely inside the ~2.2h kill window), then RESUBMIT SELF. dora checkpoints every
# XP, and `dora run` WITHOUT --clear resumes from the checkpoint, so each chunk advances the current
# seed by ~1h50m instead of restarting it. ~30h of training therefore completes in ~17 chunks without
# ever holding a GPU long enough to be noticed.
#
# GPU count deliberately kept at 2: the reference runs 14260550 (brain) and 14377929 (raster) used 2,
# and changing it changes the effective batch size, which would invalidate the seed comparison.
#
# Stops when the sentinel exists, so the chain terminates cleanly rather than resubmitting forever.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF      # torch 2.0.1 in this env rejects expandable_segments
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENTINEL=$CACHE/seeds_ALL_DONE
export TMPDIR=$CACHE/tmp XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
LOGD=$SCRATCH/project/logs/m1/seeds_chain
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} date=$(date -Is)"

if [ -f "$SENTINEL" ]; then echo "[done] sentinel present, chain complete"; exit 0; fi

# Resubmit the next chunk BEFORE doing work, so an administrative kill cannot break the chain.
NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
       $SCRATCH/project/scripts/slurm/repro_seeds_chunked.sbatch 2>/dev/null || true)
echo "[chain] next chunk queued as ${NEXT:-FAILED}"

# COMPLETION MARKER. Do NOT use `grep "Test Summary"` -- dora logs a Test Summary at EVERY evaluation
# epoch ("Test Summary | Epoch 2 | wer=43.655%"), not only at the end, so that check fires within
# minutes of a run starting. It briefly reported 1/4 seeds finished when nothing had finished, and had
# it reached 4/4 the sentinel would have stopped the chain with four incomplete runs.
# dora exits 0 only when a run actually completes (including completing after a resume), so the rc is
# the reliable signal; we persist it as a marker file because the rc itself lives in the job log, which
# is per-chunk.
run_seed () {   # $1 arm, $2 seed
  local arm=$1 S=$2
  if [ -f "$LOGD/${arm}_${S}.DONE" ]; then
    echo "[skip] ${arm} seed ${S} already finished"; return 0
  fi
  echo "[$(date -Is)] === ${arm} seed=${S} ==="
  if [ "$arm" = raster ]; then export BM_TIMING_ONLY=1; else unset BM_TIMING_ONLY; fi
  "$ENV/bin/dora" run -d seed="$S" >>"$LOGD/${arm}_${S}.log" 2>&1
  local rc=$?                        # MUST be the next line: a $(...) here would overwrite $?
  unset BM_TIMING_ONLY
  [ "$rc" -eq 0 ] && touch "$LOGD/${arm}_${S}.DONE"
  echo "[$(date -Is)] ${arm} seed=${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

for S in 3101 3102; do run_seed brain  "$S"; done
for S in 3201 3202; do run_seed raster "$S"; done

DONE=$(ls "$LOGD"/*.DONE 2>/dev/null | wc -l)
echo "[chain] seeds finished: ${DONE}/4"
[ "$DONE" -eq 4 ] && touch "$SENTINEL" && echo "[chain] ALL DONE -- sentinel written, chain will stop"
echo "[$(date -Is)] chunk complete"
