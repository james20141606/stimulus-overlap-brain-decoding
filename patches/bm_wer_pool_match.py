#!/usr/bin/env python
"""Cap the WER evaluation set so the story-split run is scored against the SAME candidate pool as the
block-split run it is meant to correct.

WHY THIS EXISTS -- two problems, one cause, one fix.

Job 15508620 (story-level split, 500G) died OUT_OF_MEMORY at 01:34:45. So did three earlier chunks at
192G. Both arms died at the identical point: "WER | 450/453", the last progress line before
`torch.cat(estimates_list)` in get_wer. It is not a leak -- raising 192G -> 500G bought eleven minutes
and hit the same wall, because the peak is a single allocation, not a drift.

  The story split makes the test partition LARGER THAN THE TRAIN partition (train | valid | test =
  118310 | 53933 | 141946), so the WER loop grows from 56 batches to 453 at batch_size=256. The
  block-split run retained 7,140 queries from its 56 batches, so at the same ~50% reject rate 453
  batches retain roughly 58,000 -- eight times the arrays. get_wer holds estimates_list, outputs_list
  and their concatenated copies live at the same moment, which is what runs into 500G to the byte
  (MaxRSS 524251404K).

THE SECOND PROBLEM, which is the one that actually matters.

Fixing this by asking for more memory would have produced a corrected external number that is not
comparable to the number it corrects. The reference run's log says:

    bm.wer INFO - wer 7140 negatives selected

7,140 < wer_negatives (10,000), so the block-split run used ALL of its rank-local queries as the
candidate pool: 7,140. The story-split run accumulates roughly 58,000, so `perm[:10000]` would have BOUND,
and the corrected number would have been scored against a 10,000-candidate pool. Chance moves
0.140% -> 0.100% and top-10 accuracy moves with it at fixed skill. The paper discloses the pool
("7,140 candidates") precisely because this metric is not pool-invariant; producing its own external
correction against a different pool would repeat the error the paper is about.

THE FIX. Stop accumulating at BM_WER_MAX_SEGMENTS segments and truncate to exactly that many. The
loader is already built with shuffle=True (get_wer shuffles so that sharding does not bias the
negatives), so the retained set is a uniform random subset of the same 40 recordings rather than a
prefix. Setting it to 7,140 makes the story-split run's candidate pool identical to the block-split
run's by construction, and drops the peak to ~32 GB as a side effect.

The cap is per rank, which is the right scope: 7,140 is itself a rank-local count, and both runs use
--gres=gpu:2.

Unset (the default) leaves get_wer exactly as it was.

Apply on the cluster:  python patches/bm_wer_pool_match.py
"""
import ast
import shutil

import os as _os
W = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/wer.py")
src = open(W).read()

# 1. os is not imported in this module.
OLD_IMP = "import logging\nimport typing as tp\n"
NEW_IMP = "import logging\nimport os\nimport typing as tp\n"
assert OLD_IMP in src, "import anchor not found"
src = src.replace(OLD_IMP, NEW_IMP, 1)

# 2. Read the cap next to the accumulators it governs.
OLD_INIT = """    estimates_list = []
    outputs_list = []
    word_hashes_list = []
    tmin = solver.args.dset.test.tmin"""
NEW_INIT = """    estimates_list = []
    outputs_list = []
    word_hashes_list = []
    # BM_WER_MAX_SEGMENTS pins the number of retained queries, and therefore the candidate pool,
    # so that runs under different splits are scored against the same pool. See
    # patches/bm_wer_pool_match.py. Unset -> original behaviour.
    _wer_cap = int(os.environ.get("BM_WER_MAX_SEGMENTS", "0")) or None
    _n_kept = 0
    tmin = solver.args.dset.test.tmin"""
assert OLD_INIT in src, "accumulator anchor not found"
src = src.replace(OLD_INIT, NEW_INIT, 1)

# 3. Stop early. estimate/wh are already reject-masked, so shape[0] is the kept count.
OLD_TAIL = """            assert (wh != 0).all()
            word_hashes_list.append(wh)
    estimates = torch.cat(estimates_list, dim=0)
    outputs = torch.cat(outputs_list, dim=0)
    word_hashes = torch.cat(word_hashes_list, dim=0).int()
"""
NEW_TAIL = """            assert (wh != 0).all()
            word_hashes_list.append(wh)
            _n_kept += int(wh.shape[0])
            # STRICTLY greater, not >=. When the natural retained count equals the cap exactly -- which
            # is the block-split case, 7,140 on the nose -- `>=` breaks on the final batch and takes a
            # different code path from the unpatched reference for no benefit. `>` lets that case fall
            # out of the loop naturally, so the arm we are comparing against runs byte-identical code.
            if _wer_cap is not None and _n_kept > _wer_cap:
                logger.info("wer segment cap reached: %d collected, truncating to %d",
                            _n_kept, _wer_cap)
                break
    estimates = torch.cat(estimates_list, dim=0)
    outputs = torch.cat(outputs_list, dim=0)
    word_hashes = torch.cat(word_hashes_list, dim=0).int()
    if _wer_cap is not None:
        # Exact, not approximate: the final batch overshoots the cap, and an off-by-a-batch pool is
        # still a different pool. Truncation is unbiased because the loader is shuffled.
        estimates = estimates[:_wer_cap]
        outputs = outputs[:_wer_cap]
        word_hashes = word_hashes[:_wer_cap]
"""
assert OLD_TAIL in src, "cat anchor not found"
src = src.replace(OLD_TAIL, NEW_TAIL, 1)

ast.parse(src)
shutil.copy(W, W + ".orig_werpool")
open(W, "w").write(src)
print("[patched] bm/wer.py: BM_WER_MAX_SEGMENTS caps the WER query set and hence the candidate pool")
