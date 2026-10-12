#!/usr/bin/env python
"""The one-bit contrast: were OTHER blocks of the evaluation narrative in training, at matched
volume, matched evaluation stimulus, matched candidate pool?

WHY THIS EXISTS

The volume-matched arm (B, patches/bm_train_subsample.py) bounds how much of the 58.41% -> 47.69%
drop is training volume, but B and C still differ in TWO ways at once: B trains on blocks of the
evaluation narrative and C does not, AND B evaluates on all four stories' test blocks while C
evaluates on The_Black_Willow alone. Neither existing pair isolates the single question the paper
turns on: does having other blocks of the SAME narrative in training inflate the score on that
narrative's held-out blocks? This patch builds the two arms that differ in that bit and nothing
else:

    arm D       train = lw1 + easy_money (whole) + The_Black_Willow's block-hash TRAIN blocks,
                subsampled via BM_TRAIN_FRAC to arm Cprime's natural train count.
    arm Cprime  train = lw1 + easy_money (whole) only.

Both arms: valid = cable_spool_fort (whole story, story-pure), and test = EXACTLY The_Black_Willow's
block-hash TEST blocks under the untouched original assignment (block uid hashing, split_assign_seed
12) -- byte-identical to the blocks the original block-split run scores on for that story. Same eval
stimulus, same candidate pool (BM_WER_MAX_SEGMENTS caps both arms identically), same volume after
the BM_TRAIN_FRAC match. The only difference left is whether The_Black_Willow's train blocks are in
the training set.

WHAT IT DOES MECHANICALLY

Gated on BM_MIXED_SPLIT in {'D', 'Cprime'}; unset leaves assign_blocks byte-identical. assign_blocks
runs once per recording and each recording is one story, so the recording's story name (passed in as
story_id, see patches/bm_story_split4.py history) decides the branch:

    lw1, easy_money      -> every block to train (index len(cdf)-1)
    cable_spool_fort     -> every block to valid (index 1)
    The_Black_Willow     -> the ORIGINAL block-hash path, seed untouched; then a drop stage removes
                            the valid-assigned blocks (mode D) or the valid- AND train-assigned
                            blocks (mode Cprime). Dropped rows belong to NO split: dataset.py
                            selects `blocks.split == j`, so a removed row is simply never used.

The drop runs AFTER the min_n_blocks_per_split check (which therefore sees the same assignment the
block-split run saw) and BEFORE the BM_TRAIN_FRAC stage, so volume matching thins the post-drop
train set. Dropping The_Black_Willow's valid blocks in BOTH modes keeps valid story-pure
(cable_spool_fort only) and keeps the eval set identical across modes.

COMPOSITION WITH BM_TRAIN_FRAC, AND THE ONE EDIT TO IT

BM_TRAIN_FRAC (patches/bm_train_subsample.py) locates the train split as split.max() -- correct
under the block split, where every recording contains all three splits. Under BM_MIXED_SPLIT that
heuristic breaks: a cable_spool_fort recording is entirely valid (max()==1), so BM_TRAIN_FRAC would
silently thin the VALIDATION set of arm D. This patch therefore also edits the train-frac stage to
use the true train index (len(cdf)-1) whenever BM_MIXED_SPLIT is set; mixed mode asserts
remove_ratio == 0, so no renumbering can shift that index. Recordings with no train blocks become a
no-op ("kept 0/0").

MUTUAL EXCLUSION

BM_MIXED_SPLIT and BM_STORY_SPLIT answer different questions and share machinery; setting both is a
config error and raises immediately rather than silently composing.

Apply on the cluster:  python patches/bm_mixed_split.py   (idempotent: skips if already applied)
"""
import ast
import shutil

import os as _os
E = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
src = open(E).read()

if "BM_MIXED_SPLIT" in src:
    print("[skip] bm/events.py already contains BM_MIXED_SPLIT; not applying twice")
    raise SystemExit(0)

# 1. Mode setup, directly after the story-split setup so the guards read as one block. The anchor is
#    the CURRENT text of the file (story-split + train-frac patches applied), not upstream's.
OLD_SETUP = """    _story_col = [story_id] * len(blocks) if story_id is not None else None
    if _story_split and _story_col is None:
        raise RuntimeError('BM_STORY_SPLIT=1 but assign_blocks got no story_id from the caller')
"""
NEW_SETUP = """    _story_col = [story_id] * len(blocks) if story_id is not None else None
    if _story_split and _story_col is None:
        raise RuntimeError('BM_STORY_SPLIT=1 but assign_blocks got no story_id from the caller')

    # BM_MIXED_SPLIT ('D' | 'Cprime') builds the one-bit contrast: both modes evaluate ONLY on
    # The_Black_Willow's block-hash TEST blocks (original path, seed untouched); lw1 and easy_money
    # go wholly to train and cable_spool_fort wholly to valid. Mode D additionally trains on
    # The_Black_Willow's block-hash TRAIN blocks; mode Cprime drops them. Both modes drop
    # The_Black_Willow's valid blocks so valid stays story-pure and the eval set is identical.
    # See patches/bm_mixed_split.py for the design rationale.
    _mixed = os.environ.get('BM_MIXED_SPLIT', '')
    if _mixed:
        if _mixed not in ('D', 'Cprime'):
            raise ValueError(f"BM_MIXED_SPLIT must be 'D' or 'Cprime', got {_mixed!r}")
        if _story_split:
            raise RuntimeError('BM_MIXED_SPLIT and BM_STORY_SPLIT are mutually exclusive')
        if _story_col is None:
            raise RuntimeError(f'BM_MIXED_SPLIT={_mixed} but assign_blocks got no story_id')
        if remove_ratio > 0.:
            raise RuntimeError('BM_MIXED_SPLIT assumes remove_ratio == 0: split indices are fixed')
        _MIXED_EVAL_STORY = 'The_Black_Willow'
        # Split indices: ratios = [test_ratio, valid_ratio] -> 0 = test, 1 = valid, last = train.
        _mixed_whole = {'lw1': len(cdf) - 1, 'easy_money': len(cdf) - 1, 'cable_spool_fort': 1}
        if story_id != _MIXED_EVAL_STORY and story_id not in _mixed_whole:
            raise RuntimeError(f'BM_MIXED_SPLIT: unrecognized story {story_id!r}')
"""
assert OLD_SETUP in src, "setup anchor not found"
src = src.replace(OLD_SETUP, NEW_SETUP, 1)

# 2. The assignment loop. Non-evaluation stories are assigned wholesale; The_Black_Willow falls
#    through to the untouched original block-hash path (uid = block.uid, seed = split_assign_seed),
#    so its test blocks are byte-identical to the block-split run's.
OLD_LOOP = """    for _i, block in enumerate(blocks.event.iter()):
        uid = _story_col[_i] if _story_split else block.uid
"""
NEW_LOOP = """    for _i, block in enumerate(blocks.event.iter()):
        if _mixed and _story_col[_i] != _MIXED_EVAL_STORY:
            # Whole-story assignment for the non-evaluation stories under BM_MIXED_SPLIT.
            split.append(_mixed_whole[_story_col[_i]])
            continue
        uid = _story_col[_i] if _story_split else block.uid
"""
assert OLD_LOOP in src, "loop anchor not found"
src = src.replace(OLD_LOOP, NEW_LOOP, 1)

# 3. The drop stage: after the min_n_blocks_per_split check (which sees the original assignment for
#    The_Black_Willow, exactly as in the block-split run) and before BM_TRAIN_FRAC (so the volume
#    match thins the post-drop train set). One print per recording, like the other patches.
OLD_TF_HEAD = """    # BM_TRAIN_FRAC subsamples the TRAIN split only, leaving valid/test untouched so the candidate"""
NEW_TF_HEAD = """    # BM_MIXED_SPLIT drop stage. Dropped rows belong to no split: dataset.py selects blocks with
    # `blocks.split == j`, so a removed row is simply unused by every split.
    if _mixed:
        _n0 = len(assigned_blocks)
        if story_id == _MIXED_EVAL_STORY:
            _drop = [1] if _mixed == 'D' else [1, len(cdf) - 1]
            assigned_blocks = assigned_blocks[~assigned_blocks.split.isin(_drop)]
        _vc = assigned_blocks.split.value_counts().to_dict()
        print(f"[BM_MIXED_SPLIT] mode={_mixed} story={story_id}: kept {len(assigned_blocks)}/{_n0} "
              f"blocks (test={_vc.get(0, 0)} valid={_vc.get(1, 0)} train={_vc.get(len(cdf) - 1, 0)} "
              f"dropped={_n0 - len(assigned_blocks)})", flush=True)

    # BM_TRAIN_FRAC subsamples the TRAIN split only, leaving valid/test untouched so the candidate"""
assert OLD_TF_HEAD in src, "train-frac head anchor not found"
src = src.replace(OLD_TF_HEAD, NEW_TF_HEAD, 1)

# 4. Fix BM_TRAIN_FRAC's train-split detection under mixed mode. split.max() is correct when every
#    recording contains all three splits (the block split) but under BM_MIXED_SPLIT a
#    cable_spool_fort recording is entirely valid (max()==1) and would be thinned as if it were
#    train. The true train index is the last ratio index; mixed mode asserts remove_ratio == 0, so
#    no renumbering can shift it. Behaviour with BM_MIXED_SPLIT unset is unchanged.
OLD_TIND = """        _train_ind = int(assigned_blocks.split.max())
"""
NEW_TIND = """        _train_ind = (len(cdf) - 1) if _mixed else int(assigned_blocks.split.max())
"""
assert OLD_TIND in src, "train_ind anchor not found"
src = src.replace(OLD_TIND, NEW_TIND, 1)

ast.parse(src)
shutil.copy(E, E + ".orig_mixedsplit")
open(E, "w").write(src)
print("[patched] bm/events.py: BM_MIXED_SPLIT builds arms D and Cprime "
      "(backup at events.py.orig_mixedsplit)")
