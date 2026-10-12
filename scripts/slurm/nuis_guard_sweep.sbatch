#!/bin/bash
#SBATCH --job-name=bci-nuis-sweep
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/meg_masc_full
M0=$SCRATCH/project/results/m0
OUT=$SCRATCH/project/results/m1
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

# Guard-strength sweep for the temporal-autocorrelation confound. train_nuisance_story.py
# eval only depends on adj_s; retraining the tiny MLP each time is cheap. Held-out story 0,
# eval all subjects. Find the guard where the timing-only baseline collapses to ~chance.
COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --epochs 80 --batch 256 --holdout_task 0"
for A in 10 30 60 120 240; do
  OUTF="$OUT/result_nuis_swp_a${A}_task0.json"
  if [ -f "$OUTF" ]; then echo "[skip] adj=$A"; continue; fi
  echo "[$(date -Is)] === nuisance story task0 adj_s=$A ==="
  $PY "$CODE/train_nuisance_story.py" $COMMON --adj_s "$A" --tag "nuis_swp_a${A}_task0"
done
echo "[$(date -Is)] DONE"
