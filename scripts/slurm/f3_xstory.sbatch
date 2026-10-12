#!/bin/bash
#SBATCH --job-name=bci-f3-xstory
#SBATCH --output=$SCRATCH/project/logs/fmri/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/fmri/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=08:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

# CROSS-STORY holdout: the strongest version of the F2 design, and what paper 2's limits ask for.
# Train on piemanpni (Pie Man), test on bronx (Running from the Bronx) -- two distinct PNI narratives
# sharing ~46 subjects. With LOSO this holds out BOTH subject and stimulus, so the timing model has no
# position-to-target lookup available: the test story's positions map to targets it never saw.
#
# Prediction recorded in experiments_detail.md BEFORE running: if F2's reading is right, the clock
# should fall to near chance here. If it stays well above chance on an unseen story, it is exploiting
# more than memorisation and F2's interpretation is incomplete.
#
# Both directions are run: bronx->piemanpni as well, since a one-directional result could be a property
# of one story's segment count (129 vs 172) rather than of the design.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/rs-qwen-env/bin/python
CODE=$SCRATCH/project/src
OUT=$SCRATCH/project/results/fmri
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

run_dir() {
  A=$1; B=$2
  echo "[$(date -Is)] === train ${A} -> test ${B} ==="
  $PY "$CODE/train_fmri_xstory.py" --out_dir "$OUT" \
      --train_task "$A" --test_task "$B" \
      --subjects 20 --folds 5 --lags 0,4,8,10,12,14,20 \
      --epochs 40 --n_vox 5000 --seg_s 3.0 --tag "f3_${A}_to_${B}"
  rc=$?
  echo "[$(date -Is)] ${A}->${B} rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] ${A}->${B} exited ${rc}"
}

run_dir piemanpni bronx
run_dir bronx piemanpni
echo "[$(date -Is)] DONE"
