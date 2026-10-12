#!/bin/bash
#SBATCH --job-name=bci-narr-kf
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 --time=04:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition
set -euo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
META=$SCRATCH/project/fmri/narratives_meta
OUT=$SCRATCH/project/results/narr
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
# Does the crudest confound (word count alone) cross modality? On Narratives fMRI stimuli, does a
# word-count-only timing model clear chance? Reference baseline -- no brain comparison here.
COMMON="--data_dir $META --ctx $META/narr_contextual.npy --ctx_meta $META/narr_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 --eval_col task --holdout_col story_group --keep_feats n_words --epochs 80 --batch 256"
for G in pieman sherlock tunnel 21styear slumlord merlin; do
  OUTF="$OUT/result_narrkf_nwords_${G}.json"
  if [ ! -f "$OUTF" ]; then
    echo "[$(date -Is)] narr keep n_words, hold group $G"
    $PY "$CODE/train_nuisance_story.py" $COMMON --holdout_task "$G" --tag "narrkf_nwords_${G}"
  fi
done
echo "[$(date -Is)] DONE"
