#!/bin/bash
#SBATCH --job-name=bci-lb-long
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=192G --gres=gpu:1 --time=12:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition
set -euo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/libribrain_full
M0=$SCRATCH/project/results/m0
OUT=$SCRATCH/project/results/lb
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
# The 80-epoch LibriBrain BRAIN arm had NOT converged: val top10 was still climbing at the last
# epoch (x1.0@ep40 -> x1.8@ep60 -> x2.4@ep70 -> x3.4@ep79) while the timing arm had already peaked
# by ep20 and was overfitting. Comparing a still-improving brain against a converged timing model
# is exactly the "your decoder is undertrained" objection this paper cannot afford to earn.
# Retrain the brain to convergence (400 epochs) before any LibriBrain claim.
COMMON="--data_dir $DATA --ctx $M0/libribrain_gpt2_contextual.npy --ctx_meta $M0/libribrain_gpt2_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 --batch 256 --eval_col session"
for H in Sherlock3 Sherlock2; do
  OUTF="$OUT/result_lb_brain400_${H}_L6.json"
  if [ ! -f "$OUTF" ]; then
    echo "[$(date -Is)] === LibriBrain BRAIN 400ep, hold $H ==="
    $PY "$CODE/train_story_holdout.py" $COMMON --holdout_task "$H" --epochs 400 --decim 2 \
        --adj_multi 10,30,60,120,240 --tag "lb_brain400_${H}_L6"
  fi
  # position-free timing arm: is the huge LibriBrain timing number just onset_pos lookup?
  OUTF="$OUT/result_lb_nuis_nopos_${H}_L6.json"
  if [ ! -f "$OUTF" ]; then
    echo "[$(date -Is)] === LibriBrain TIMING no-pos, hold $H ==="
    $PY "$CODE/train_nuisance_story.py" $COMMON --holdout_task "$H" --epochs 80 --drop_pos \
        --adj_multi 10,30,60,120,240 --tag "lb_nuis_nopos_${H}_L6"
  fi
done
echo "[$(date -Is)] DONE"
