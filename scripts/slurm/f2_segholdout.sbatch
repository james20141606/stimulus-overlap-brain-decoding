#!/bin/bash
#SBATCH --job-name=bci-f2-segho
#SBATCH --output=$SCRATCH/project/logs/fmri/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/fmri/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=12:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

# THE EXPERIMENT THAT DECIDES WHETHER PAPER 2 EXISTS.
#
# F1 (result_f1_{pieman,piemanpni}.json) reported the clock at 12.33x vs the brain at 2.01x and the
# draft was built on that ~6x gap. The 2026-07-31 code audit found the comparison was never valid:
# the timing arm was trained AND evaluated on the same rows (train_fmri_f1.py, old lines 189-191)
# while the brain arm was a real LOSO. Because the timing feature vector contains position-in-story --
# a unique id per fixed-width segment -- the clock could memorise an id->target lookup, and 12.33x is
# 97.1% top-10, which is a memorisation signature rather than a timing effect.
#
# This run applies --seg_blocks, a blocked segment holdout applied IDENTICALLY to both arms, so each
# must generalise to segments it never saw. Alternating contiguous blocks (not one half/half cut) keep
# the train and test `t` ranges interleaved, so the clock is not crippled by extrapolation instead --
# that would be the opposite error. --seg_buffer 2 drops 2 segments each side of every boundary, i.e.
# a >=5-segment (>=15 s) gap, comfortably wider than the 10 s adjacency guard, so BOLD autocorrelation
# cannot carry across the split.
#
# READ THE OUTPUT THIS WAY. The test pool shrinks from ~127 to ~53 segments, so top-10 chance rises
# and x chance is NOT comparable to the F1 numbers -- only the brain-vs-clock contrast WITHIN this run
# is meaningful. Both outcomes are publishable and both are worth having:
#   * clock still clearly ahead  -> paper 2's headline is restored, now on a matched comparison.
#   * clock collapses toward the brain -> the F1 gap was memorisation, the headline stays retracted,
#     and what remains is the positive control plus an honest account of how the baseline fooled us.
# The seg_blocks=0 arm reruns the ORIGINAL setting so the two appear side by side in one file.
set -uo pipefail
export PYTHONUNBUFFERED=1
# NOTE: deliberately NOT setting PYTORCH_CUDA_ALLOC_CONF -- see repro_seeds.sbatch. rs-qwen-env is
# newer torch so it would be valid here, but there is no reason to vary it between runs.
PY=$SCRATCH/envs/rs-qwen-env/bin/python
CODE=$SCRATCH/project/src
OUT=$SCRATCH/project/results/fmri
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

for TASK in piemanpni pieman; do
  for NB in 6 0; do
    TAG="f2_${TASK}_blk${NB}"
    echo "[$(date -Is)] === ${TAG} ==="
    $PY "$CODE/train_fmri_f1.py" --out_dir "$OUT" --task "$TASK" \
        --subjects 20 --lags 0,4,8,10,12,14,20 \
        --epochs 40 --n_vox 5000 --seg_s 3.0 \
        --seg_blocks "$NB" --seg_buffer 2 --tag "$TAG"
    rc=$?                       # MUST stay on the line right after the command; no substitutions
    echo "[$(date -Is)] ${TAG} rc=${rc}"
    [ "$rc" -ne 0 ] && echo "[FAIL] ${TAG} exited ${rc}"
  done
done

echo "[$(date -Is)] DONE"
