#!/bin/bash
#SBATCH --job-name=bci-wsweep-wc
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=cpuB_partition

# CPU-only word-count-alone arm of the w-sweep. The GPU pool is saturated and my 2-GPU cap is held
# by other papers, but the word-count arm needs no GPU: build_nuisance_segments reads only
# trial_metadata.tsv + memmapped GPT-2 targets (no MEG array), and the encoder is a tiny per-segment
# scalar MLP. So this runs immediately on CPU and delivers the core R3-caveat evidence -- how the
# single-feature (word-count) confound scales with the segment window w -- without waiting for a GPU.
# It writes result_nuis_keepR_n_words_task{K}_w{WT}.json; the GPU sweep job (14186086) sees those and
# skips the word-count arm, running only the brain arm. w=3.0 already on disk.
set -uo pipefail                       # NOT -e: one failed w must not kill the rest
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/m1
M0=$R/bci_foundation/results/m0
DATA=$R/meg_papers/data/meg_masc_full
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA} (CPU, word-count arm)"

EPOCHS=${EPOCHS:-80}
TASKS=${TASKS:-"0 1 2 3"}
WSET=${WSET:-"1.0 1.5 2.0 4.0 5.0"}    # 3.0 already on disk
echo "[INFO] EPOCHS=$EPOCHS TASKS=$TASKS WSET=$WSET"

for W in $WSET; do
  WT=$(echo "$W" | sed 's/\.//')
  for K in $TASKS; do
    NTAG="nuis_keepR_n_words_task${K}_w${WT}"
    if [ -f "$OUT/result_${NTAG}.json" ]; then echo "[skip] wordcount w=$W task=$K"; continue; fi
    echo "[$(date -Is)] === WORDCOUNT w=$W task=$K ==="
    $PY "$CODE/train_nuisance_story.py" --out_dir "$OUT" --data_dir "$DATA" \
        --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
        --holdout_task "$K" --layer 6 --seg_s "$W" --min_words 3 --epochs "$EPOCHS" \
        --keep_feats n_words --tag "$NTAG" || echo "[warn] WORDCOUNT failed w=$W task=$K"
  done
done
echo "[$(date -Is)] DONE"
