#!/bin/bash
#SBATCH --job-name=bci-mb1m
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-9%1
#SBATCH --requeue

# MB-1m: extend the HEADLINE comparison from n=5 to n=10 held-out subjects.
#
# The headline is the one-bit contrast: identical pretraining set {0,1,2}, matched single-story eval
# pools, differing only in whether pretraining saw the evaluation story.
#     SEEN   = pretrain{0,1,2}, eval story 0  (mb1k condition)
#     UNSEEN = pretrain{0,1,2}, eval story 3  (mb1h condition)
# Both currently have n=5. Doubling the subject count is the single highest-value remaining experiment:
# it is the only thing that materially strengthens the paper, since the three kill-gates already passed
# and everything else is writing.
#
# Array structure copied from MB-1i, which is the configuration that actually survives this cluster:
# `sacct` showed monolithic jobs dying as `CANCELLED by 0` (root) with Reason=QOSMaxGRESPerUser at
# ~2.2h, i.e. a periodic administrative pass culling whatever exceeds the per-user GPU allowance. One
# subject per task at 1 GPU finishes inside that window; %1 throttles so this never pushes the account
# over; skip-if-exists means a culled task costs one subject, not the run.
#
# Tasks 0-4 add the five missing subjects to the SEEN arm, tasks 5-9 to the UNSEEN arm.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

NEW=(03 09 10 11 13)
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
  S=${NEW[$SLURM_ARRAY_TASK_ID]}; EVAL=0; TAG="mb1k_volctl_s${S}"   # SEEN arm
else
  S=${NEW[$((SLURM_ARRAY_TASK_ID-5))]}; EVAL=3; TAG="mb1h_stimho_s${S}"  # UNSEEN arm
fi
echo "[INFO] host=$(hostname) task=$SLURM_ARRAY_TASK_ID subject=$S eval_story=$EVAL tag=$TAG date=$(date -Is)"

if [ -f "$OUT/result_${TAG}.json" ]; then
  echo "[skip] $TAG already exists"; exit 0
fi

# Both arms hold story 3 out of PRETRAINING; they differ only in which story is scored.
$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story 3 --eval_story "$EVAL" --guard_scope stimulus \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
