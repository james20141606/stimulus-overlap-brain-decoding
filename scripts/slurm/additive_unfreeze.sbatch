#!/bin/bash
#SBATCH --job-name=bci-unfreeze
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=16:00:00
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

# R10: the additive test freezes a target head fitted alongside timing, so the brain must express
# itself in its competitor's space -- a null there is partly a statement about that handicap.
# main.tex conceded "a fusion that re-fits the head jointly is the version we would run next".
# Run it, rather than promise it. --unfreeze_head lets phase 2 re-shape the target space.
#
# CONTROL (--null_brain): identical phase 2, head unfrozen, brain excluded from the optimizer (its
# zero-init weights never move, so brain(x) stays exactly 0). Without this control the extra 80
# epochs of head training would be credited to the brain.
COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 --mode additive --epochs ${EPOCHS:-80} --batch 256 --unfreeze_head"

TASKS=${TASKS:-"0 1 2 3"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
for T in $TASKS; do
  for SD in $SEEDS; do
    F="$OUT/result_addUH_task${T}_seed${SD}.json"
    if [ ! -f "$F" ]; then
      echo "[$(date -Is)] === +brain+head  task=$T seed=$SD ==="
      $PY "$CODE/train_fusion_story.py" $COMMON --holdout_task "$T" --seed "$SD" \
          --tag "addUH_task${T}_seed${SD}"
    fi
    F="$OUT/result_addNULL_task${T}_seed${SD}.json"
    if [ ! -f "$F" ]; then
      echo "[$(date -Is)] === CONTROL nullbrain+head  task=$T seed=$SD ==="
      $PY "$CODE/train_fusion_story.py" $COMMON --null_brain --holdout_task "$T" --seed "$SD" \
          --tag "addNULL_task${T}_seed${SD}"
    fi
  done
done
echo "[$(date -Is)] DONE"
