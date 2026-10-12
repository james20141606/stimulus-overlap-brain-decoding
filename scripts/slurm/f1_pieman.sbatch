#!/bin/bash
#SBATCH --job-name=bci-f1
#SBATCH --output=$SCRATCH/project/logs/fmri/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/fmri/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=12:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

# F1 (PLAN_FMRI kill gate): can an fMRI decoder clear chance at all on Narratives `pieman`?
# Cross-subject LOSO, 20 subjects, full lag sweep, timing-only clock run alongside in the SAME
# protocol so the fMRI numbers are read the way main.tex insists MEG numbers must be.
#
# The lag sweep is the point, not a hyperparameter search: the meta/scan onset convention is
# ambiguous (meta spans 8.7-429.5 s, events.tsv puts the story at 15-437 s) and the haemodynamic
# delay adds ~4-6 s. Guessing wrong manufactures a null that looks like "fMRI cannot decode". We
# report the whole curve; a flat-at-chance curve across all lags is an honest negative.
#
# Smoke (3 subj / 5 epochs / 2k vox, CPU) already ran clean: 127 segments, brain x1.1-1.4,
# clock x6.89. Undersized on purpose -- this job is the real test of the brain arm.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/rs-qwen-env/bin/python
CODE=$SCRATCH/project/src
OUT=$SCRATCH/project/results/fmri
mkdir -p "$OUT" $SCRATCH/project/logs/fmri
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

for TASK in pieman piemanpni; do
  if [ -f "$OUT/result_f1_${TASK}.json" ]; then echo "[skip] $TASK"; continue; fi
  echo "[$(date -Is)] === F1 $TASK ==="
  $PY "$CODE/train_fmri_f1.py" --out_dir "$OUT" --task "$TASK" \
      --subjects 20 --lags 0,2,4,6,8,10,12,14,16,18,20 \
      --epochs 40 --n_vox 5000 --seg_s 3.0 --tag "f1_${TASK}"
done

echo "[$(date -Is)] DONE"
