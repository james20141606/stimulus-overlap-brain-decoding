#!/bin/bash
#SBATCH --job-name=bci-calmatch
#SBATCH --output=$SCRATCH/project/logs/calmatch_20260924/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/calmatch_20260924/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=200G --gres=gpu:1 --time=01:30:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition,gpuB_partition
#SBATCH --array=0-7%2
#SBATCH --requeue

# CALIBRATION-DIVERSITY CONTROL for Table 1 (paper_iclr.tex, tab:headline).
# Pre-registration: notes/PREREG_calibmatch_20260924.md (committed before any job exists).
#
# Published arms, story 0 (REUSED, not rerun):
#   SEEN    result_mb1k_volctl_s*    pretrain {0,1,2}  calibrate {1,2}    eval 0
#   UNSEEN  result_mb1j_story0_s*    pretrain {1,2,3}  calibrate {1,2,3}  eval 0
# New arm (this job):
#   MATCHED result_mb1z_calmatch_story0_s*  pretrain {1,2,3}  calibrate {1,2}  eval 0
# MATCHED has SEEN's calibration stories -- and, with seed 0, SEEN's exact validation segments and
# first-k calibration subsets (checked per subject by the preflight) -- and UNSEEN's pretraining set.
# Story 1 (secondary, descriptive): SEEN mb1s_story1_seen (calib {0,2}), UNSEEN mb1s_story1_unseen
# (calib {0,2,3}), MATCHED mb1z_calmatch_story1_s* (pretrain {0,2,3}, calib {0,2}, eval 1).
#
# Tasks 0-4: story 0, two subjects per task (primary; all ten Table-1 subjects).
# Tasks 5-7: story 1, subjects 01 02 05 07 08 (secondary). Array order puts the primary first.
#
# Engineering decisions (see PREREG section "Compute"):
# * CODE = the pinned bundle scripts/calmatch_20260924/src_pinned: train_mb.py SHA 1a4953ec (the
#   remote file the headline runs used, confirmed 2026-09-22) and eval_metrics.py SHA c5b94117
#   (commit 4566e61, 07-28 -- the scorer in place for every Table 1 run). The live src/eval_metrics.py
#   (08-18 hardening) references `_tie_hits` before assignment in guarded_retrieval_metrics and would
#   raise UnboundLocalError at the first evaluation.
# * BACKEND = historical: no cudnn.deterministic, no CUBLAS_WORKSPACE_CONFIG (explicitly unset). The
#   deterministic bundle is what made smoke 18254449 ~195x slower (profile 18321669). The preflight
#   times five conv steps and refuses to train above 0.2 s/step (historical 0.0094 s).
# * TWO SUBJECTS SHARE ONE GPU (second starts 300 s later, after the first's data build peaks). One
#   historical subject run needs ~3-4 GiB of GPU memory and 45-79 GiB host RSS; one H200 per subject
#   leaves the GPU mostly idle, and clusterA cancels jobs below 50% utilisation. Packing also halves the
#   number of allocations under the 2-GPU per-user cap. Utilisation is sampled every 15 s into the
#   log directory and summarised in each receipt.
# * CHECKPOINT = per subject. train_mb.py keeps weights in memory only, so the resumable unit is one
#   subject's result JSON (~16 min historical). A task skips any subject whose result + receipt exist;
#   resubmitting the same script after a cancel reruns only what is missing.
set -uo pipefail
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset CUBLAS_WORKSPACE_CONFIG            # historical recipe; the deterministic bundle is the 195x regression
PY=$SCRATCH/envs/analysis-env/bin/python
R=$SCRATCH/project
B=$R/bci_foundation
CODE=$B/scripts/calmatch_20260924/src_pinned
PRE=$B/scripts/calmatch_preflight_20260924.py
OUT=$B/results/mb
REC=$B/results/calmatch_20260924
LOG=$B/logs/calmatch_20260924
M0=$B/results/m0
DATA=$R/meg_papers/data/meg_masc_full
mkdir -p "$OUT" "$REC" "$LOG"

I=${SLURM_ARRAY_TASK_ID:?array task id required}
case "$I" in
  0) EV=0; SH=3; CAL=1,2; SUBS="01 02" ;;
  1) EV=0; SH=3; CAL=1,2; SUBS="03 05" ;;
  2) EV=0; SH=3; CAL=1,2; SUBS="07 08" ;;
  3) EV=0; SH=3; CAL=1,2; SUBS="09 10" ;;
  4) EV=0; SH=3; CAL=1,2; SUBS="11 13" ;;
  5) EV=1; SH=3; CAL=0,2; SUBS="01 02" ;;
  6) EV=1; SH=3; CAL=0,2; SUBS="05 07" ;;
  7) EV=1; SH=3; CAL=0,2; SUBS="08" ;;
  *) echo "[FATAL] unknown task $I"; exit 2 ;;
esac
JOBTAG="${SLURM_ARRAY_JOB_ID:-local}_${I}"
echo "[INFO] host=$(hostname) task=$I eval_story=$EV calib=$CAL subjects=[$SUBS] date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1

# SHA pin of the code bundle -- refuse to run on a drifted copy.
( cd "$CODE" && sha256sum -c --quiet SHA256SUMS ) || { echo "[FATAL] pinned bundle SHA mismatch"; exit 4; }
SBATCH_SHA=$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1)

# GPU utilisation sampler for the whole task.
UTIL="$LOG/util_${JOBTAG}.csv"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 15 > "$UTIL" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run_one () {   # $1 subject
  local S=$1
  local TAG="mb1z_calmatch_story${EV}_s${S}"
  local RES="$OUT/result_${TAG}.json" RCPT="$REC/receipt_${TAG}.json"
  if [ -f "$RES" ] && [ -f "$RCPT" ]; then echo "[$(date -Is)] [skip] $TAG done"; return 0; fi
  local T0=$(date +%s)
  if [ ! -f "$RES" ]; then
    "$PY" "$PRE" --data_dir "$DATA" --ctx "$M0/masc_gpt2_contextual.npy" \
        --ctx_meta "$M0/masc_gpt2_contextual_meta.json" --test_subject "$S" --eval_story "$EV" \
        --seen_holdout "$SH" --calib_stories "$CAL" --require_cuda --live_src "$B/src" \
        --out "$REC" --tag "$TAG" > "$LOG/preflight_${TAG}.log" 2>&1
    local prc=$?
    cat "$LOG/preflight_${TAG}.log"
    if [ "$prc" -ne 0 ]; then echo "[FAIL] $TAG preflight rc=$prc -- not training"; return 3; fi
    echo "[$(date -Is)] === $TAG: pretrain excludes story $EV, calibrate on {$CAL}, eval story $EV ==="
    "$PY" "$CODE/train_mb.py" --out_dir "$OUT" \
        --data_dir "$DATA" \
        --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
        --test_subject "$S" --variants conv,dynamic_big --layer 6 \
        --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
        --pre_epochs 80 --adapt_epochs 120 \
        --holdout_story "$EV" --calib_stories "$CAL" --guard_scope stimulus --seed 0 \
        --tag "$TAG" > "$LOG/train_${TAG}.log" 2>&1
    local rc=$?
    echo "[$(date -Is)] $TAG rc=$rc"; tail -5 "$LOG/train_${TAG}.log"
    [ "$rc" -ne 0 ] && { echo "[FAIL] $TAG train rc=$rc"; return "$rc"; }
  fi
  local T1=$(date +%s)
  # Receipt: hashes of result, code, preflight; the log lines that state the realised design.
  "$PY" - "$RES" "$RCPT" "$REC/preflight_${TAG}.json" "$LOG/train_${TAG}.log" "$UTIL" \
        "$SBATCH_SHA" "$((T1-T0))" "$JOBTAG" "$CODE" <<'PYEOF'
import hashlib, json, sys, time, os, glob
res, rcpt, pre, log, util, sb_sha, el, jobtag, code = sys.argv[1:10]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
lines = []
if os.path.exists(log):
    lines = [l.rstrip() for l in open(log) if l.startswith("[mb1]") or l.startswith("[save]")]
u = []
if os.path.exists(util):
    for l in open(util):
        p = [x.strip() for x in l.split(",")]
        if len(p) >= 2 and p[1].isdigit():
            u.append(int(p[1]))
d = json.load(open(res))
json.dump({
    "tag": d.get("tag"), "result_file": os.path.basename(res), "result_sha256": sha(res),
    "preflight_sha256": sha(pre), "train_log_sha256": sha(log),
    "pinned_src_sha256": {os.path.basename(p): sha(p) for p in sorted(glob.glob(code + "/*.py"))},
    "sbatch_sha256": sb_sha, "seed": 0,
    "args_echo": {k: d.get(k) for k in ("test_subject", "holdout_story", "eval_story",
                                         "calib_stories", "guard_scope", "klist", "seg_s", "layer")},
    "design_log_lines": lines, "elapsed_s_this_invocation": int(el), "job": jobtag,
    "gpu_util_samples_so_far": len(u),
    "gpu_util_mean_so_far": (round(sum(u) / len(u), 1) if u else None),
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}, open(rcpt, "w"), indent=2)
print(f"[receipt] {rcpt}")
PYEOF
}

set -- $SUBS
FIRST=$1; SECOND=${2:-}
run_one "$FIRST" & P1=$!
P2=""
if [ -n "$SECOND" ]; then
  sleep 300
  run_one "$SECOND" & P2=$!
fi
RC=0
wait "$P1" || RC=$?
if [ -n "$P2" ]; then wait "$P2" || RC=$?; fi
kill $SAMPLER 2>/dev/null
"$PY" - "$UTIL" <<'PYEOF'
import sys
u = []
for l in open(sys.argv[1]):
    p = [x.strip() for x in l.split(",")]
    if len(p) >= 2 and p[1].isdigit():
        u.append(int(p[1]))
if u:
    print(f"[util] samples={len(u)} mean={sum(u)/len(u):.1f}% min={min(u)} max={max(u)}")
PYEOF
echo "[$(date -Is)] task $I done rc=$RC"
exit $RC
