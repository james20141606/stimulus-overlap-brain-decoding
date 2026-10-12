#!/bin/bash
#SBATCH --job-name=bci-mb1r
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-9%1
#SBATCH --requeue

# MB-1r: COMPOSITION-MATCHED arm. This is now the control the causal claim depends on.
#
# Two objections converge on it.
#   (i) The headline contrast is not one bit. SEEN pretrains on {0,1,2} and UNSEEN on {1,2,3}, both
#       evaluating story 0, so the third pretraining story differs (0 vs 3). If story 3 is a poorer
#       pretraining stimulus, part of the gap comes from that rather than from story 0's absence.
#  (ii) The from-scratch arm does NOT rescue the attribution. It sits at percentile 0.499 and 0.508 at
#       k=100 and k=500 -- at chance, with no signal to lose -- so its flatness is a floor effect, not
#       evidence. Where it does have headroom (k=2000, 0.078 above chance) it drops 22% relative,
#       against the pretrained arm's 46%. That is not "a few points of material difficulty".
#
# This arm removes the substitution instead of controlling for it:
#     SEEN   = pretrain{0,1,2}  eval story 0   (already have, mb1k, n=10)
#     UNSEEN = pretrain{1,2}    eval story 0   (this job)
# The pretraining sets are now nested: {1,2} is exactly {0,1,2} minus the evaluation story. Nothing is
# substituted, so composition is matched and only VOLUME differs -- the complementary design to mb1j0,
# which matched volume and let composition differ. An effect present in both is attributable to the
# evaluation stimulus; an effect in only one is not.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

SUBS=(01 02 03 05 07 08 09 10 11 13)
S=${SUBS[$SLURM_ARRAY_TASK_ID]}
TAG="mb1r_compmatch_s${S}"
echo "[INFO] host=$(hostname) subject=$S tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

# holdout 0 AND 3 -> pretrain{1,2}; eval_story 0 -> score the withheld evaluation stimulus
$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story "0,3" --eval_story 0 --guard_scope stimulus \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
