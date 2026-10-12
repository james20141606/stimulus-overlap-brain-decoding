#!/bin/bash
#SBATCH --job-name=bci-nuis-fix
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=04:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition
set -euo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/meg_masc_full
M0=$SCRATCH/project/results/m0
OUT=$SCRATCH/project/results/m1
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --epochs 80 --batch 256 --adj_s 10.0 --adj_multi 10,30,60,120,240"
# (a) all 4 stories, guard sweep from ONE fitted model (mirrors brain --adj_multi), both variants.
#     Fixes: missing task0 file; timing curve previously RETRAINED per guard (guard effect was
#     confounded with training-seed noise); drop_pos only ever reported for story 0.
for T in 0 1 2 3; do
  for V in withpos nopos; do
    EXTRA=""; [ "$V" = "nopos" ] && EXTRA="--drop_pos"
    OUTF="$OUT/result_nuis_swpfix_${V}_task${T}.json"
    if [ -f "$OUTF" ]; then echo "[skip] $V task$T"; continue; fi
    echo "[$(date -Is)] === nuisance sweep-fix $V task=$T ==="
    $PY "$CODE/train_nuisance_story.py" $COMMON --holdout_task "$T" $EXTRA \
        --tag "nuis_swpfix_${V}_task${T}"
  done
done
echo "[$(date -Is)] DONE"
