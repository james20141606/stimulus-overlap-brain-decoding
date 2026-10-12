#!/bin/bash
#SBATCH --job-name=bci-f1-fetch
#SBATCH --output=$SCRATCH/repro/%x-%j.out
#SBATCH --error=$SCRATCH/repro/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT_B --partition=cpuA_partition

# F1 data fetch for PLAN_FMRI.md: pull the preprocessed BOLD for the `pieman` story only.
#
# Why pieman: of the stories that have BOTH many subjects and word-level alignments already in
# narratives_meta/trial_metadata.tsv, pieman has the most subjects (82 subjects / 142 runs, 954 words).
# That is the cross-subject arm analogous to MEG-MASC's LOSO design.
#
# Why afni-nosmooth volume: F0 (PLAN_FMRI §6) established that Narratives ships preprocessed
# derivatives, so we do NOT run fMRIPrep on 345 subjects, and afni-nosmooth is the right arm
# (no spatial smoothing). We take space-MNI152NLin2009cAsym res-native desc-clean (59.5 MB/run,
# ~8.5 GB total) rather than the fsaverage6 surface (131 MB/run for both hemis, ~18.6 GB).
#
# Blocker cleared 2026-07-26: git-annex was missing, so all 1298 BOLD files were empty 142-byte
# symlinks. Installed git-annex+datalad into $SCRATCH/envs/annex; a single-file fetch verified
# real content (37,451,725 bytes from the fcp-indi mirror) before committing to this bulk pull.
set -uo pipefail
export PATH=$SCRATCH/envs/annex/bin:$PATH
export TMPDIR=$SCRATCH/repro/tmp
D=$SCRATCH/project/fmri/narratives/derivatives/afni-nosmooth
cd "$D" || { echo "[FAIL] no afni-nosmooth dir"; exit 1; }
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

echo "[$(date -Is)] === fetching pieman volume BOLD (afni-nosmooth, MNI res-native desc-clean) ==="
# -J4: four parallel transfers. git-annex skips anything already present, so this is resumable.
git annex get -J4 sub-*/func/*task-pieman*space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz 2>&1 \
  | grep -vE "^\s*[0-9]+%" | tail -40
echo "[$(date -Is)] annex get exit=$?"

echo "=== how many landed ==="
n=0; tot=0
for f in sub-*/func/*task-pieman*space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz; do
  if [ -f "$f" ] && [ "$(stat -Lc %s "$f" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    n=$((n+1)); tot=$((tot + $(stat -Lc %s "$f")))
  fi
done
echo "fetched $n runs, $((tot/1024/1024)) MB total"
echo "[$(date -Is)] DONE"
