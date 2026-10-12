#!/bin/bash
#SBATCH --job-name=bci-seeds
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G --gres=gpu:2 --time=44:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --requeue

# Seed replicates for the two HEADLINE arms of sec:brainmagick, so the 89.5-vs-58.4 comparison can
# carry an interval instead of being two point estimates. Both AC reviews named this as the change
# that most moves the paper; the protocol is already matched (40-epoch cap, patience 10, best-valid
# selection, identical stimulus-disjoint split and 7,140 pool), so seeds are the missing piece.
#
# We already hold seed 2036 (brain, JID 14260550) and 2037-equivalent (raster, 14377929). This adds two
# more seeds per arm -> n=3 each. dora's XP signature includes the seed, so these do not collide with
# the existing runs or with each other.
set -uo pipefail
export PYTHONUNBUFFERED=1
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here. The brainmagick env pins torch 2.0.1
# and that option only exists from 2.1, so every worker dies in torch.cuda.set_device with
# "Unrecognized CachingAllocator option: expandable_segments". That is exactly what killed JID
# 15007388: four XPs created, all workers crashed in <60s, job reported COMPLETED 0:0. See
# experiments_detail.md 2026-07-31.
unset PYTORCH_CUDA_ALLOC_CONF
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
export TMPDIR=$CACHE/tmp XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }

LOGD=$SCRATCH/project/logs/m1/seeds_${SLURM_JOB_ID:-manual}
mkdir -p "$LOGD"

# run_one <arm> <seed>: keep dora's FULL output in a per-seed file and capture dora's OWN exit status.
# Two bugs in the previous version made a total no-op look like success:
#   1. `echo "[$(date -Is)] ... exit=$?"` reports the exit status of `date`, not of the pipeline,
#      because the command substitution runs first and overwrites $?. Capture rc on its own line.
#   2. piping dora through `grep` discarded every traceback, so the crash left no trace anywhere.
run_one() {
  arm=$1; S=$2
  echo "[$(date -Is)] === ${arm} seed=${S} ==="
  "$ENV/bin/dora" run -d seed="$S" >>"$LOGD/${arm}_${S}.log" 2>&1
  rc=$?                                   # MUST be the very next line; no substitutions before it
  grep -E "Test Summary|New best|Train Summary" "$LOGD/${arm}_${S}.log" | tail -20
  if [ "$rc" -ne 0 ]; then
    echo "[FAIL] ${arm} seed=${S} dora_rc=${rc} -- last 15 lines:"
    tail -15 "$LOGD/${arm}_${S}.log"
  fi
  echo "[$(date -Is)] ${arm} seed=${S} dora_rc=${rc}"
  return 0
}

# ROOT CAUSE of three failed attempts (2026-08-01). The cluster runs PreemptMode=REQUEUE with
# preempt/partition_prio, so gpuA_partition jobs are preempted after ~2.5h. dora checkpoints (a 153MB
# checkpoint.th appeared under outputs/xps/<sig>/ each time), BUT this script called
# `dora run --clear`, which WIPES the XP folder before running. So every restart began from zero, and
# a ~9h brain seed could never survive a 2.5h preemption cycle. Fixed: dropped --clear so dora resumes,
# added --requeue so SLURM restarts the job automatically, and the per-seed log now appends (>>).
#
# ARM selects which half to run. NOTE (2026-07-31): the two-job split is now DISCOURAGED. Running
# brain and raster as separate 2-GPU jobs held 4 GPUs at once and both were killed at 01:02 with
# Reason=QOSMaxGRESPerUser, alongside the user's own concurrent jobs. The QOS_A QoS allows
# MaxWall 2 days, so all four runs (~29.6h) fit in ONE 2-GPU job -- which is also the smallest
# footprint against other work on the account. Default ARM=all and submit once.
# Sizing (from sacct on the originals): brain 14260550 took 8h55m, raster 14377929 took 5h46m.
# All four sequentially is ~29.6h against a 24h wall, so a single job would lose the last raster seed
# after burning a full day. Split: brain 2x9h = 18h, raster 2x5.8h = 11.6h, each comfortably inside 24h,
# and the two run in parallel.
ARM=${ARM:-all}
if [ "$ARM" = brain ] || [ "$ARM" = all ]; then
  for S in 3101 3102; do
    unset BM_TIMING_ONLY
    run_one brain "$S"
  done
fi
if [ "$ARM" = raster ] || [ "$ARM" = all ]; then
  for S in 3201 3202; do
    export BM_TIMING_ONLY=1
    run_one raster "$S"
    unset BM_TIMING_ONLY
  done
fi

echo "[$(date -Is)] DONE -- per-seed logs in $LOGD"
