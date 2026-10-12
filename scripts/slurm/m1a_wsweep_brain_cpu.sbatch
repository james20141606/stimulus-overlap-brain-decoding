#!/bin/bash
#SBATCH --job-name=bci-wsweep-br
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=96G --time=2-00:00:00
#SBATCH --account=CLUSTER_ACCOUNT_B --partition=cpuA_partition

# CPU brain arm of the w-sweep, to complete the two-arm table while the GPU pool is QOS-capped
# (14186086 blocked by QOSMaxGRESPerUser). The MEG epochs array (31G) is memmapped, not loaded, so
# CPU RAM is fine; only speed is the question. This job and the GPU job 14186086 write the SAME
# result_story_holdout_task{K}_w{WT}.json files and both skip existing ones, so whichever reaches a
# file first wins and there is no double-compute -- if the GPU frees, it races ahead on the rest.
set -uo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/m1
M0=$R/bci_foundation/results/m0
DATA=$R/meg_papers/data/meg_masc_full
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} threads=$OMP_NUM_THREADS (CPU brain arm)"

EPOCHS=${EPOCHS:-80}
TASKS=${TASKS:-"0 1 2 3"}
WSET=${WSET:-"1.0 1.5 2.0 4.0 5.0"}    # 3.0 already on disk
echo "[INFO] EPOCHS=$EPOCHS TASKS=$TASKS WSET=$WSET"

for W in $WSET; do
  WT=$(echo "$W" | sed 's/\.//')
  for K in $TASKS; do
    BTAG="story_holdout_task${K}_w${WT}"
    if [ -f "$OUT/result_${BTAG}.json" ]; then echo "[skip] brain w=$W task=$K"; continue; fi
    echo "[$(date -Is)] === BRAIN w=$W task=$K ==="
    $PY "$CODE/train_story_holdout.py" --out_dir "$OUT" --data_dir "$DATA" \
        --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
        --holdout_task "$K" --layer 6 --seg_s "$W" --min_words 3 --decim 2 --epochs "$EPOCHS" \
        --tag "$BTAG" || echo "[warn] BRAIN failed w=$W task=$K"
  done
done
echo "[$(date -Is)] DONE"
