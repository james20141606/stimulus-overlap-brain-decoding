#!/bin/bash
#SBATCH --job-name=bci-mb1p
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-9%1
#SBATCH --requeue

# MB-1p: SEED REPLICATES for the architecture-reversal claim. Now non-optional.
#
# The paper's strongest claim is that a pre-registered architecture comparison REVERSES sign once the
# shared stimulus is removed. That claim is currently single-seed. Agarwal et al. (NeurIPS 2021
# Outstanding Paper, arXiv:2108.13264) argues precisely that few-run point estimates make deep-learning
# benchmark conclusions unreliable -- a reviewer will cite it AGAINST a single-seed reversal, and they
# would be right to. An effect that "becomes null or significantly reversed" needs to be shown stable
# across initialisations or it reads as noise.
#
# Two extra seeds (1, 2) on the clean condition -- story 3 out of pretraining, evaluated on story 3 --
# for the five subjects that already have seed 0. Gives 3 seeds x 5 subjects for both arms.
#
# Array structure is the one that survives this cluster: `CANCELLED by 0` with Reason=QOSMaxGRESPerUser
# at ~2.2h is a periodic administrative pass, not preemption, so one subject per task at 1 GPU finishes
# inside the window and %1 keeps the account under its allowance.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

SUBS=(01 02 05 07 08)
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
  S=${SUBS[$SLURM_ARRAY_TASK_ID]}; SEED=1
else
  S=${SUBS[$((SLURM_ARRAY_TASK_ID-5))]}; SEED=2
fi
TAG="mb1p_seed${SEED}_s${S}"
echo "[INFO] host=$(hostname) subject=$S seed=$SEED tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story 3 --guard_scope stimulus --seed "$SEED" \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
