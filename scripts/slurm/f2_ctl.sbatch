#!/bin/bash
#SBATCH --job-name=bci-f2-ctl
#SBATCH --output=$SCRATCH/project/logs/fmri/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/fmri/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

# TRAINING-SIZE CONTROL for F2 (15044963).
# --seg_blocks 6 shrinks training from ~129 to ~56 segments AT THE SAME TIME as it holds segments out,
# so the observed collapse (clock 12.33 -> 1.99; brain 2.01 -> 1.25) has two candidate causes and the
# blk6-vs-blk0 comparison cannot separate them.
# This arm trains on the SAME ~56 segments and evaluates on THOSE SAME segments. So:
#   ctl vs blk0  = effect of training-set SIZE alone (both evaluate on seen segments)
#   ctl vs blk6  = effect of the HOLDOUT alone (both train on 56)
# If ctl reproduces blk0's inflated numbers, the collapse is generalisation. If ctl is already
# collapsed, the story is largely data volume and F2 cannot support the claim I was about to make.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/rs-qwen-env/bin/python
CODE=$SCRATCH/project/src
OUT=$SCRATCH/project/results/fmri
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

for TASK in piemanpni pieman; do
  TAG="f2_${TASK}_ctl"
  echo "[$(date -Is)] === ${TAG} ==="
  $PY "$CODE/train_fmri_f1.py" --out_dir "$OUT" --task "$TASK" \
      --subjects 20 --lags 0,4,8,10,12,14,20 \
      --epochs 40 --n_vox 5000 --seg_s 3.0 \
      --seg_blocks 6 --seg_buffer 2 --eval_on_train --tag "$TAG"
  rc=$?
  echo "[$(date -Is)] ${TAG} rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] ${TAG} exited ${rc}"
done
echo "[$(date -Is)] DONE"
