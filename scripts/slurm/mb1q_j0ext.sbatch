#!/bin/bash
#SBATCH --job-name=bci-mb1q
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-4%1
#SBATCH --requeue

# MB-1q: extend the UNSEEN-on-story-0 arm to n=10, so the WITHIN-STORY headline reaches n=10.
#
# Planning error being corrected. MB-1m was written to extend mb1k (SEEN, eval story 0) and mb1h
# (UNSEEN, eval story 3) -- i.e. the CROSS-story contrast. It was launched before the Kajimura et al.
# 2023 citation prompted the realisation that the cross-story contrast confounds stimulus identity
# with the bit under test, and that the headline should instead be:
#     SEEN   = mb1k  pretrain{0,1,2}, eval story 0   (story 0 IS in pretraining)
#     UNSEEN = mb1j0 pretrain{1,2,3}, eval story 0   (story 0 is NOT in pretraining)
# Both score the SAME narrative, so stimulus identity cannot contribute. mb1k is now at n=10 and
# mb1j0 is at n=5, so the headline is pinned at n=5 by this arm alone. This job adds the missing five.
#
# Array structure as elsewhere: one subject per task, 1 GPU, 2h, %1 -- the configuration that survives
# the periodic administrative cull (`CANCELLED by 0`, Reason=QOSMaxGRESPerUser, at ~2.2h).
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

SUBS=(03 09 10 11 13)
S=${SUBS[$SLURM_ARRAY_TASK_ID]}
TAG="mb1j_story0_s${S}"
echo "[INFO] host=$(hostname) subject=$S tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

# holdout_story 0 with the default eval_story => pretrain excludes story 0, evaluate on story 0.
$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big,dynamic_big_noflow --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story 0 --guard_scope stimulus \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
