#!/bin/bash
#SBATCH --job-name=bci-fusion-incr
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=10:00:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# INCREMENTAL test (reviewer's decisive control on the "your decoder is just bad" objection).
# Head-to-head brain-vs-timing cannot rule out a weak decoder. This asks instead whether the brain
# adds anything ON TOP OF timing: if x(brain+timing) ~ x(timing), MEG carries no information
# orthogonal to timing in this regime -- a conclusion that does not require the brain to win.
# All three arms share split/seed/epochs/metric; only the encoder input differs.
TASKS=${TASKS:-"0"}
SEEDS=${SEEDS:-"0 1 2"}
MODES=${MODES:-"timing brain fusion"}
EPOCHS=${EPOCHS:-80}
ADJ=${ADJ:-10.0}
COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --adj_s $ADJ --epochs $EPOCHS --batch 256"
echo "[INFO] TASKS=$TASKS SEEDS=$SEEDS MODES=$MODES EPOCHS=$EPOCHS ADJ=$ADJ"

for T in $TASKS; do
  for SD in $SEEDS; do
    for M in $MODES; do
      OUTF="$OUT/result_fusion_${M}_task${T}_seed${SD}.json"
      if [ -f "$OUTF" ]; then echo "[skip] $M task$T seed$SD"; continue; fi
      echo "[$(date -Is)] === incremental mode=$M task=$T seed=$SD ==="
      $PY "$CODE/train_fusion_story.py" $COMMON --mode "$M" --holdout_task "$T" --seed "$SD" \
          --tag "fusion_${M}_task${T}_seed${SD}"
    done
  done
done
echo "[$(date -Is)] DONE"
