# Stimulus Overlap Changes Model Comparisons in Cross-Subject Brain Decoding

Xupeng Chen (NYU), Chenqian Le (NYU), Michael Yue (Columbia University), Runhao Li (University of
Southern California), Jiayi Cheng (NYU)

Code, audit tools, the corrected MEG-MASC split and the receipts behind the paper.

## Summary

Cross-subject brain decoding usually holds out the listener, not the story. On MEG-MASC every
listener hears the same four stories, so the test story reaches the model twice: through the other
listeners in pretraining and, under the standard split, through calibration. A subject-only split does
not check this overlap, and the model comparison we test is sensitive to it. Under the standard protocol
a dynamic encoder beats a matched static convolutional one at low budgets; with story-disjoint
calibration the advantage is undetectable at one pretraining seed and resolved at k=100 over three,
and with the story withheld from pretraining too no narrative-budget combination resolves in its favour.
Transfer survives: withholding the story from pretraining at matched story count leaves about 54% of the
above-chance signal. Returning a narrative's training blocks to a reproduced reference pipeline at
matched volume prices the exposure (+3.00 top-10 points at full dose). This shows protocol sensitivity at
this sample size, not equal architectures. For unheard-stimulus claims, hold the story out of pretraining
and calibration and report a stimulus-derived baseline. The two overlap checkers in this repository need
only the Python standard library.

## Citation

```bibtex
@article{chen2026stimulusoverlap,
  title   = {Stimulus Overlap Changes Model Comparisons in Cross-Subject Brain Decoding},
  author  = {Chen, Xupeng and Le, Chenqian and Yue, Michael and Li, Runhao and Cheng, Jiayi},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## Quick start

The two audit tools run on stock Python 3.9+ with no third-party imports:

    python3 scripts/leakage_report.py --demo meg-masc                          # corpus exposure report
    python3 scripts/leakage_report.py --manifest artifacts/manifest_brennan2019.csv
    python3 scripts/stimulus_leakage_check.py --manifest your_split.csv        # exits 1 on overlap (CI)

`stimulus_leakage_check.py` audits a split: the manifest needs `subject`, `stimulus` and `split`
columns. The shipped corpus manifests have no `split` column, so the tool refuses them by design.

## Verify the release

    bash verify_release.sh            # PY=/path/to/python to choose the interpreter; needs numpy

It runs every check that works without the training cluster: both audit tools, the one-bit readouts
(Table 4), the post-hoc sensitivity intervals (Appendix C), Table 1 from the per-subject receipts, a
rerun of the pre-registered calibration-diversity analysis, the seed-replicate appendix table, and the
receipt SHA-256 prefixes printed in the paper. These are summary recomputations, not training reruns.
The other per-table generators read training logs that are not shipped and fail with a message without
them (`RELEASE.md`, "What does NOT regenerate off-cluster").

## Directory map

| path | contents |
|---|---|
| `RELEASE.md` | inventory: what each piece is and what it needs (start here) |
| `LEAKAGE.md` | the protocol note we recommend |
| `scripts/stimulus_leakage_check.py`, `scripts/leakage_report.py` | the two stdlib-only audit tools |
| `artifacts/` | the stimulus-disjoint MEG-MASC split and subject x stimulus manifests |
| `patches/`, `scripts/brainmagick_patches/` | env-gated patches to the brainmagick reference pipeline (apply order in `RELEASE.md` section 3) |
| `src/` | model code in the versions that produced the reported numbers (needs MEG-MASC, PyTorch, a GPU) |
| `scripts/*.py` | per-table and per-figure generators, analysis scripts |
| `scripts/slurm/` | run configurations of every internal and external arm (cluster paths replaced by placeholders) |
| `results/`, `figures/*.values.json`, `figures/fig1_v4/` | receipts behind the tables and figures |
| `notes/` | pre-registrations cited in the appendix |
| `prespecification/` | excerpts of the lab notebook and program document that fix each criterion in advance |
| `verify_release.sh` | off-cluster checks |

Cluster paths, account and host names in the shipped files are replaced by placeholders (`USER`,
`$SCRATCH`, `CLUSTER_ACCOUNT`, `cluster-login`). The full lab notebook and training logs are not
released.

## Data

No recordings are redistributed. The manifests list subject and stimulus identifiers only; the result
JSONs hold per-subject decoding scores computed by us.

- MEG-MASC: Gwilliams et al. (2023), https://doi.org/10.17605/OSF.IO/AG3KJ, CC0 1.0.
- Brennan & Hale (2019) Alice EEG: only the 33-subject x 1-stimulus shape is reproduced in an example
  manifest; no data from that release were downloaded or are included.
- Narratives fMRI and GPT-2 weights are not included.

## License

Code and documentation: MIT (see `LICENSE`). Exception: the scripts under `patches/` and
`scripts/brainmagick_patches/` quote and modify code from facebookresearch/brainmagick and are
distributed under that project's CC BY-NC 4.0 licence (non-commercial research use); see `RELEASE.md`
section 3.

## Contact

Xupeng Chen, xc1490@nyu.edu
