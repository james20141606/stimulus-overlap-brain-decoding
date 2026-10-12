#!/bin/bash
#SBATCH --job-name=bci-top1eval
#SBATCH --output=$SCRATCH/project/logs/m1/top1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/top1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=200G --gres=gpu:2 --time=01:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# TOP-1 EVAL OF THE REPRODUCED PIPELINE, AGAINST DEFOSSEZ ET AL. 2023's ">41% top-1".
#
# WHY. Our reproduced-pipeline headline (xp 1e87bf54, seed=3102, block split) is top-10:
# Test Summary at best gave wer=42.794% => top-10 57.21%. The paper's published number is
# top-1 (>41%). This job re-evaluates the SAME trained weights with test.wer_topx=1 so the
# reproduction can be compared against the published metric directly. Eval only, no training
# intended.
#
# MECHANISM (why this doesn't retrain). dora keys the xp on the config signature, so
# `seed=3102 test.wer_topx=1` is a NEW xp: b78e9a79 (verified via `dora info`). We pre-seeded
# outputs/xps/b78e9a79/checkpoint.th with a copy of 1e87bf54's checkpoint, patched so:
#   last_test_epoch = -1     -> best_epoch(4) > last_test_epoch  => test stage WILL run
#   best_loss      = -1e30   -> the one replayed-resume train epoch (see below) can never
#                               displace the epoch-4 best_state; the test is guaranteed to
#                               evaluate exactly the weights that produced 42.794% top-10.
# On resume: history len=14 => solver.epoch=15; bm/solver.py's loop body runs ONE train+valid
# epoch (there is no check-before-train path), then early stopping fires
# (15 >= best_epoch 4 + patience 10), the test stage runs on best_state with wer_topx=1,
# commit, break. Budget: train ~22min + valid ~7min + test ~4min + startup < 1h.
#
# POOL. BM_WER_MAX_SEGMENTS=7140 pins the candidate pool identically to every other number
# (the block split retains exactly 7140 queries, so the cap is inert, but the code path is
# identical). Story/mixed/train-frac gates are explicitly unset: this is the ORIGINAL split.
#
# top-1 = 100 - wer from the line "Test Summary | Epoch 15 | wer=XX% ...".
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT
unset BM_STORY_SPLIT_SEED
unset BM_MIXED_SPLIT
unset BM_TRAIN_FRAC
unset BM_TRAIN_FRAC_SEED
export BM_WER_MAX_SEGMENTS=7140

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
LOGD=$SCRATCH/project/logs/m1/top1
XPDIR=$BM/outputs/xps/b78e9a79
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} date=$(date -Is)"

# Refuse to run if the seeded checkpoint is missing: without it dora would TRAIN b78e9a79
# from scratch, which is exactly what this job must never do.
[ -f "$XPDIR/checkpoint.th" ] || { echo "[FAIL] $XPDIR/checkpoint.th missing - aborting to avoid a from-scratch train"; exit 1; }
"$ENV/bin/python" - <<'EOF' || { echo "[FAIL] seeded checkpoint not patched as expected"; exit 1; }
import torch
s = torch.load("$SCRATCH/repro/brainmagick/outputs/xps/b78e9a79/checkpoint.th", map_location="cpu")
assert s["last_test_epoch"] == -1 and s["best_epoch"] == 4 and len(s["history"]) == 14, \
    (s["last_test_epoch"], s["best_epoch"], len(s["history"]))
print("[check] seeded checkpoint ok: last_test_epoch=-1 best_epoch=4 history=14")
EOF

"$ENV/bin/dora" run -d seed=3102 test.wer_topx=1 >>"$LOGD/top1_3102.log" 2>&1
rc=$?
echo "[$(date -Is)] dora_rc=$rc"
grep -E "Test Summary|negatives selected|New best" "$LOGD/top1_3102.log" | tail -5
[ "$rc" -eq 0 ] && grep -q "Test Summary | Epoch 15" "$LOGD/top1_3102.log" && touch "$LOGD/top1_3102.DONE"
echo "[$(date -Is)] eval chunk complete"
