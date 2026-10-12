#!/bin/bash
#SBATCH --job-name=bci-f3-fetch
#SBATCH --output=$SCRATCH/project/logs/fmri/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/fmri/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT_B --partition=cpuA_partition

# Fetch `bronx` ("Running from the Bronx") BOLD, to enable the CROSS-STORY holdout that paper 2's
# limits section names as its next step.
#
# Why this story. F2 established that a within-story segment holdout is enough to kill the
# memorisation, but the stronger design -- and the one an earlier analysis uses -- trains on one
# narrative and tests on a DIFFERENT one, so no test target has been seen in any form. That needs
# subjects who heard two different stories. From the derivative file listing:
#     bronx & piemanpni -> 46 shared subjects   (bronx: 0 fetched)
#     black & piemanpni -> 45,  forgot & piemanpni -> 45
# The PNI session ran four distinct stories over the same ~46 subjects, so this is a clean design.
# bronx is a different narrative from Pie Man (not another telling of it, unlike piemanpni), and it has
# 1356 word-level alignments already in narratives_meta/trial_metadata.tsv.
#
# Size: 47 runs at ~60 MB = ~2.8 GB. Scratch was at 83.9% space / 78.0% inodes when this was written,
# so the fetch is ~0.06% of the space quota and 47 inodes. Negligible here, but the headroom is
# shrinking and is worth watching.
set -uo pipefail
export PATH=$SCRATCH/envs/annex/bin:$PATH
export TMPDIR=$SCRATCH/repro/tmp
D=$SCRATCH/project/fmri/narratives/derivatives/afni-nosmooth
cd "$D" || { echo "[FAIL] no afni-nosmooth dir"; exit 1; }
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

PAT='sub-*/func/*task-bronx*space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz'
echo "[$(date -Is)] === fetching bronx volume BOLD ==="
# -J4: four parallel transfers; git-annex skips what is already present, so this is resumable.
# NOTE: capture rc on its own line. The sibling script f1_fetch_pieman.sbatch writes
# `echo "[$(date -Is)] annex get exit=$?"`, which reports the exit status of `date` rather than of the
# command, because the command substitution runs first and overwrites $?. That same bug hid a total
# failure in seeds 15007388 -- do not reintroduce it.
git annex get -J4 $PAT > /tmp/annex_bronx.$$ 2>&1
rc=$?
grep -vE "^[[:space:]]*[0-9]+%" /tmp/annex_bronx.$$ | tail -25
echo "[$(date -Is)] annex get rc=${rc}"

echo "=== how many actually landed with real content ==="
n=0; tot=0
for f in $PAT; do
  if [ -f "$f" ] && [ "$(stat -Lc %s "$f" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    n=$((n+1)); tot=$((tot + $(stat -Lc %s "$f")))
  fi
done
echo "fetched $n runs, $((tot/1024/1024)) MB total"
[ "$n" -lt 20 ] && echo "[FAIL] expected ~47 runs, got $n -- cross-story arm not viable yet"
rm -f /tmp/annex_bronx.$$
echo "[$(date -Is)] DONE"
