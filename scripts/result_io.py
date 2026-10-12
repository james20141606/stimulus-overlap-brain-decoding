"""Safe loading of per-subject result JSONs.

Exists because of a real contamination on 2026-07-31. `results/mb` held 11 `mb1vq` files for 10
subjects; the extra was `result_mb1vq_smoke_s01.json`, a smoke test. Two independent mistakes combined:

  * the glob `result_mb1vq_s*.json` MATCHES `result_mb1vq_smoke_s01.json`, because after `_s` the `*`
    happily eats `moke_s01`; and
  * the subject key was parsed from the filename (`split("_s")[-1]`), which maps `smoke_s01` to `01`.

Since `s01` sorts before `smoke_s01`, the smoke run silently overwrote the real subject 01 and the
aggregate mean was wrong by 0.04 with no error raised. The output looked clean and plausible.

The same glob-and-parse shape appears in mb_decompose.py, mb_overlap_check.py and f2_analyze.py, i.e.
in the code producing headline numbers for all three papers. Hence this helper.
"""
import glob
import json
import os

SKIP = ("smoke", "test", "tmp", "debug", "scratch")


def load_subject_results(results_dir, prefix, expect=None, quiet=False):
    """Load {subject: parsed_json} for files named `<prefix>*.json` under `results_dir`.

    Subject comes from the `test_subject` FIELD, never from the filename. Files whose basename
    contains any SKIP marker are excluded and reported. Raises if two files claim the same subject,
    which is the failure this module exists to prevent.
    """
    hits = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json")))
    kept, skipped = {}, []
    for f in hits:
        b = os.path.basename(f)
        if any(k in b.lower() for k in SKIP):
            skipped.append(b)
            continue
        d = json.load(open(f))
        s = d.get("test_subject")
        if s is None:
            skipped.append(f"{b} (no test_subject field)")
            continue
        # Type-normalization guard: 1 (int) and "01" (str) must not silently become two subjects.
        # Digit-like keys normalize to their integer form; anything else stays verbatim.
        s = str(int(s)) if str(s).strip().isdigit() else str(s)
        if s in kept:
            raise RuntimeError(f"duplicate subject {s}: {b} collides with an earlier file")
        kept[s] = d
    # Skip reports print even under quiet=True: every headline generator passes quiet, which had
    # silently disabled the one report this module exists to give (audit 2026-08-19). quiet now
    # suppresses only the routine inventory line.
    if skipped:
        print(f"[inventory] {prefix}*: SKIPPED {len(skipped)} file(s) -> {skipped}")
    if not quiet:
        print(f"[inventory] {prefix}*: {len(hits)} matched, {len(kept)} kept, {len(skipped)} skipped")
    if expect is not None and len(kept) != expect:
        print(f"[inventory] WARNING {prefix}*: expected {expect} subjects, kept {len(kept)}")
    return kept
