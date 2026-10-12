#!/bin/bash
#SBATCH --job-name=bci-mb1u
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-19%1
#SBATCH --requeue

# MB-1u: seed replicates for the two arms carrying the MAGNITUDE claim.
#
# Agarwal et al. (NeurIPS 2021 Outstanding Paper) is the standard authority that few-run point
# estimates make benchmark conclusions unreliable, and a reviewer will cite it against a headline
# resting on one seed. The reversal condition already has three seeds; the magnitude claim -- which is
# now the paper's lead -- does not. These are the two arms it is computed from:
#     SEEN   pretrain{0,1,2} (excludes story 3), evaluate story 0   [mb1k]
#     UNSEEN pretrain{1,2,3} (excludes story 0), evaluate story 0   [mb1j0]
# Seeds 1 and 2 on five subjects each, giving 3 seeds x 5 subjects for the paired ratio.
#
# tasks 0-4   SEEN   seed 1      tasks 5-9   SEEN   seed 2
# tasks 10-14 UNSEEN seed 1      tasks 15-19 UNSEEN seed 2
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

SUBS=(01 02 05 07 08)
I=$SLURM_ARRAY_TASK_ID
if   [ "$I" -lt 5  ]; then S=${SUBS[$I]};        HOLD=3; SD=1; ARM=seen
elif [ "$I" -lt 10 ]; then S=${SUBS[$((I-5))]};  HOLD=3; SD=2; ARM=seen
elif [ "$I" -lt 15 ]; then S=${SUBS[$((I-10))]}; HOLD=0; SD=1; ARM=unseen
else                       S=${SUBS[$((I-15))]}; HOLD=0; SD=2; ARM=unseen
fi
TAG="mb1u_${ARM}_seed${SD}_s${S}"
echo "[INFO] host=$(hostname) subject=$S holdout=$HOLD seed=$SD arm=$ARM tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

# eval_story 0 in BOTH arms: SEEN has story 0 in pretraining, UNSEEN does not.
$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story "$HOLD" --eval_story 0 --guard_scope stimulus --seed "$SD" \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
