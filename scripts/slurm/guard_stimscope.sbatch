#!/bin/bash
#SBATCH --job-name=bci-guardstim
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=04:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# Re-run the guard-strength sweep under a STIMULUS-scoped guard (docs/adr/0001), for both arms.
# Criteria for reading the result were PRE-REGISTERED in experiments_detail.md (2026-07-29) BEFORE this
# job was submitted, because the recording-scoped sweep could not establish the "global, not merely
# local" claim: 9/10 subjects heard story 0 twice, so every masked neighbour kept an unmasked twin.
#
# Both arms get the identical treatment (same seg_s, min_words, layer, epochs, guard widths) so the
# comparison stays like-for-like; only the guard's scope changes.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/m1
M0=$R/bci_foundation/results/m0
DATA=$R/meg_papers/data/meg_masc_full
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json \
        --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 \
        --adj_multi 10,30,60,120,240 --epochs 80 --guard_scope stimulus"

echo "[$(date -Is)] === TIMING arm, stimulus-scoped guard, held-out story 0 ==="
$PY "$CODE/train_nuisance_story.py" $COMMON --holdout_task 0 --tag nuis_story_stimguard_task0

echo "[$(date -Is)] === BRAIN arm, stimulus-scoped guard, held-out story 0 ==="
$PY "$CODE/train_story_holdout.py" --out_dir "$OUT" --data_dir "$DATA" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --holdout_task 0 --layer 6 --seg_s 3.0 --min_words 3 --decim 2 --epochs 80 \
    --adj_s 10.0 --adj_multi 10,30,60,120,240 --guard_scope stimulus \
    --tag story_holdout_stimguard_task0

echo "[$(date -Is)] DONE"
