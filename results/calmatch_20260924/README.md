# Calibration-diversity control (paper appendix "Calibration-diversity control")

Pre-registration: `notes/PREREG_calibmatch_20260924.md` (commit and deviations as written).
Rerun the pre-registered analysis from this package's root (numpy only):

    python3 scripts/calmatch_analysis_20260924.py --results results/calmatch_20260924/primary/inputs \
        --receipts results/calmatch_20260924/primary/receipts --out /tmp/analysis.json

`primary/inputs/result_*.json` are byte-identical to the files the analysis read (their SHA-256 are
the ones in `analysis.json` and `primary/selection.json`). Receipts, preflights, `analysis.json`
and `selection.json` are identity-scrubbed copies (cluster names -> clusterA/clusterB, paths ->
$SCRATCH, host names -> gpu-node), so their own hashes differ from the originals; the result-file
hashes recorded inside them are unchanged. The shipped scripts are scrubbed the same way (comments,
help strings and cluster labels only), so they do not hash to `script_sha256` in `analysis.json`.
