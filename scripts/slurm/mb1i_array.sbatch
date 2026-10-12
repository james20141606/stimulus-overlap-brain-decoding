#!/bin/bash
#SBATCH --job-name=bci-mb1i
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-4%1
#SBATCH --requeue

# TUNED FROM-SCRATCH BASELINE, restructured as a JOB ARRAY after diagnosing why every long job died.
#
# ROOT CAUSE (2026-08-01): `sacct -o Reason` shows CANCELLED **by 0** -- uid 0, i.e. root -- with
# Reason=QOSMaxGRESPerUser, at 02:13 / 02:27 / 02:30 across four separate jobs. A periodic
# administrative enforcement pass kills jobs that put the user over the per-user GPU allowance, and the
# user's own concurrent work (ct-lanpair x3, sig-dpo-dose, ecog_highgamm) consumes most of it. This is
# NOT preemption, so `--requeue` never fires -- that flag only applies to PREEMPTED jobs.
#
# The fix is therefore structural, not a flag: make every job SHORT enough to finish inside the
# enforcement window, and hold only ONE GPU.
#   * one subject per array task  (~45 min with the extended 15-config grid)
#   * --time=02:00:00, comfortably under the ~2.2h observed kill time
#   * %1 throttle: at most one task at a time, so this never pushes the user over the cap
#   * skip-if-exists: a killed task loses at most one subject, and re-running the array resumes
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
SUBJECTS=(01 02 05 07 08)
S=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) task=$SLURM_ARRAY_TASK_ID subject=$S date=$(date -Is)"

if [ -f "$OUT/result_mb1i_tuned_s${S}.json" ]; then
  echo "[skip] subject $S already done"; exit 0
fi
$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story 3 --guard_scope stimulus \
    --scratch_grid "3e-5,1e-4,3e-4,1e-3,3e-3:1e-2,1e-1,1.0" \
    --tag "mb1i_tuned_s${S}"
rc=$?
echo "[$(date -Is)] subject $S rc=${rc}"
exit $rc
