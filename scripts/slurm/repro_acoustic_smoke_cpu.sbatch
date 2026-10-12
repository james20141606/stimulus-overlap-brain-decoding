#!/bin/bash
#SBATCH --job-name=bci-ac-smoke
#SBATCH --output=$SCRATCH/repro/%x-%j.out
#SBATCH --error=$SCRATCH/repro/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT_B --partition=cpuA_partition

# R1 SMOKE (CPU): a tiny end-to-end plumbing check of the training loop on the CACHED data, before
# committing the expensive 2-GPU multi-hour R1. Catches any torchaudio.info-style API/version error at
# model-build / cache-load / first forward-backward / metric, in minutes, on CPU (no GPU competition).
# NOT the real R1 -- 2 recordings, 1 epoch, 3 batches, device=cpu.
set -uo pipefail
export PYTHONUNBUFFERED=1
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
export TMPDIR=$CACHE/tmp XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} (R1 CPU smoke)"

cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[$(date -Is)] === R2 dora run (TIMING-ONLY smoke: cpu, 2 recordings, 1 epoch, 3 batches) ==="
export BM_TIMING_ONLY=1 BM_TIMING_ACOUSTIC=1
"$ENV/bin/dora" run --clear device=cpu num_workers=0 dset.n_recordings=2 optim.epochs=1 optim.max_batches=3 2>&1
echo "[$(date -Is)] SMOKE dora exit=$?"
