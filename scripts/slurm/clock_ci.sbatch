#!/bin/bash
#SBATCH --job-name=bci-clockci
#SBATCH --output=$SCRATCH/project/logs/fmri/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/fmri/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT_B --partition=cpuA_partition

# Bootstrap CI for the CROSS-STORY clock arm.
# train_fmri_xstory.py saved timing_x as a bare point estimate and discarded the top10_ci95 that
# guarded_retrieval_metrics already computes. That number (1.62x / 1.74x) is load-bearing -- it is the
# evidence that timing features generalise to an unseen narrative -- so it cannot stay intervalless in
# a paper about unsupported numbers. The clock needs no BOLD, so this is CPU-only.
# Batch rather than interactive: two 40-epoch fits over a 19x-tiled training set take longer than a
# login-node session should hold.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $SCRATCH/project
$SCRATCH/envs/rs-qwen-env/bin/python scripts/xstory_clock_ci.py
rc=$?
echo "[done] rc=${rc}"
