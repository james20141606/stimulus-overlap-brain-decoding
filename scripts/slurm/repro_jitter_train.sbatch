#!/bin/bash
#SBATCH --job-name=bci-jitter
#SBATCH --output=$SCRATCH/repro/%x-%j.out
#SBATCH --error=$SCRATCH/repro/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G --gres=gpu:2 --time=48:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# Defossez R1: reproduce their PUBLISHED gwilliams2022 number with the UNMODIFIED pipeline, BEFORE any
# timing arm. STAGED -- do not submit until (a) the download/preprocess job finished (data present)
# and (b) the 2-GPU multi-hour cost is re-confirmed. R1 is the kill-gate: if we cannot reproduce their
# ~41% top-1, we have no standing to critique the metric and we STOP.
#
# CRITICAL (README): they trained on 2 GPUs, and contrastive negatives are taken only WITHIN a GPU, so
# the GPU count changes the result. Reproduce with 2 GPUs + `dora run -d` (batch 256 = 128/GPU).
# Target: "more than 41% top-1 accuracy" on Gwilliams (README line 20 / Nature MI 2023, Table 1).
set -uo pipefail
export PYTHONUNBUFFERED=1
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
export TMPDIR=$CACHE/tmp XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[$(date -Is)] === R2: dora run --clear -d TIMING-ONLY (word-onset raster) ==="
# -d = distributed over both GPUs (required to match their negatives-per-GPU setup).
export BM_TIMING_ONLY=1 BM_TIMING_JITTER=1
"$ENV/bin/dora" run --clear -d seed=2041 2>&1
echo "[$(date -Is)] JITTER dora exit=$?  -- check test top-1 vs the published ~41%"
