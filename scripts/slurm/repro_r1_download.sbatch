#!/bin/bash
#SBATCH --job-name=bci-repro-dl
#SBATCH --output=$SCRATCH/repro/%x-%j.out
#SBATCH --error=$SCRATCH/repro/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# Defossez / brainmagick STEP 3 (R1 prep): download + preprocess gwilliams2022 (MEG-MASC) with their
# unmodified pipeline. This is the user-gated shared-compute action (approved 2026-07-18). R0 install
# is done and verified (PLAN_REPRO §11). After this lands we reproduce their published top-10 number
# (R1 kill-gate) BEFORE inserting any timing arm.
#
# EVERY cache is redirected to /scratch: the earlier errno 28 was the tiny $HOME inode cap (30k), and
# wav2vec2/HF/torch-hub model downloads default to $HOME/.cache -- which would hit the same wall.
set -uo pipefail
export PYTHONUNBUFFERED=1
BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
export TMPDIR=$CACHE/tmp XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data \
       PIP_CACHE_DIR=$CACHE/pip_cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
myquota 2>/dev/null | grep -E "/home|/scratch"

cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
# download_only fetches raw + runs their preprocessing; NO timing arm, NO modification.
# OSF throws transient HTTP 500s on big multi-file pulls; download_osf skips files that already
# exist (bm/studies/download.py:43 `if file_.exists(): continue`), so retrying RESUMES rather than
# restarting. Retry the whole command a few times to ride out the 500s unattended.
RC=1
for attempt in 1 2 3 4 5 6 7 8; do
  echo "[$(date -Is)] === dora download attempt $attempt (resumes; skips existing files) ==="
  "$ENV/bin/dora" run download_only=true 'dset.selections=[gwilliams2022]' 2>&1
  RC=$?
  [ "$RC" -eq 0 ] && { echo "[$(date -Is)] download+preprocess OK on attempt $attempt"; break; }
  echo "[$(date -Is)] attempt $attempt failed rc=$RC (likely transient OSF 500); retry in 30s"
  sleep 30
done
echo "[$(date -Is)] dora final exit=$RC after retries"
echo "=== data landed? ==="
du -sh "$BM/data/gwilliams2022" 2>/dev/null || echo "no data dir"
find "$BM/data/gwilliams2022" -maxdepth 2 -type f 2>/dev/null | head -10
myquota 2>/dev/null | grep -E "/home|/scratch"
echo "[$(date -Is)] DONE rc=$RC"
