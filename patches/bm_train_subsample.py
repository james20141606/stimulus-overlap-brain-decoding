#!/usr/bin/env python
"""Subsample the TRAIN split only, so training volume can be varied without touching evaluation.

WHY THIS EXISTS

The corrected external number (58.41% -> 47.69% guarded top-10 under story-level splits) is confounded
and the paper says so: with four narratives the corrected partition is 1 test / 1 valid / 2 train, so
the training set also falls from 203,152 to 118,310 examples (58%). Story-level splitting and reduced
training volume move together and cannot be separated by choosing a different split.

They CAN be separated by refitting. Run the ORIGINAL block-level split with the training set cut to the
story-split's volume and everything else untouched:

    block split, full train      203,152   -> 58.41%   (have)
    block split, train at 58%    118,310   -> ?        (this patch)
    story split, full train      118,310   -> 47.69%   (have)

The middle arm shares its volume with the bottom arm and its split with the top one, so the two
contrasts decompose the 10.72-point drop. If the middle arm stays near 58.41%, volume is not carrying
the effect and the drop is the shared story. If it falls most of the way to 47.69%, volume is.

WHY NOT `remove_ratio`, WHICH ALREADY EXISTS

`assign_blocks(..., remove_ratio=r)` drops a fraction of ALL blocks before splitting, so it shrinks
test along with train. That would be fatal here for a reason specific to this project: bm/wer.py sets
the candidate pool to min(wer_negatives, N_queries), and the block-split run currently retains exactly
7,140 queries. Shrinking test by 42% would leave ~4,100, the BM_WER_MAX_SEGMENTS=7140 cap would no
longer bind, and the pool would silently change from 7,140 to ~4,100 -- making the new arm
incomparable to both existing arms in precisely the way this paper is about. Train-only subsampling
leaves test untouched, so all three arms keep one pool.

WHAT IT DOES

BM_TRAIN_FRAC=f keeps a fraction f of TRAIN blocks, chosen with a fixed seed, and leaves valid and test
exactly as they were. Unset -> no change. The kept set is a random subset of blocks rather than a
prefix, so it is not biased toward the start of recordings.

Note the composition caveat, which is real and must be disclosed: under the block split the retained
training blocks still come from all four stories, whereas the story-split arm trains on two whole
stories. The middle arm therefore matches the bottom arm in VOLUME but not in stimulus composition --
the same volume-matched / composition-matched distinction the paper already draws for its own
experiments. It bounds the volume contribution; it does not make the comparison composition-free.

Apply on the cluster:  python patches/bm_train_subsample.py
"""
import ast
import shutil

import os as _os
E = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
src = open(E).read()

# Inject immediately before assign_blocks returns, i.e. AFTER the min_n_blocks_per_split check (so
# subsampling cannot trip a check about the ORIGINAL assignment) and after the remove_ratio
# renumbering (so `split` indices are final). Train is the remainder appended by
#   ratios.append(1. - sum(ratios))
# and is therefore always the LAST index, whether or not remove_ratio was used.
ANCHOR = """    return assigned_blocks


def split_wav_as_block("""
assert ANCHOR in src, "assign_blocks return anchor not found"

INJECT = """    # BM_TRAIN_FRAC subsamples the TRAIN split only, leaving valid/test untouched so the candidate
    # pool cannot move. See patches/bm_train_subsample.py for why remove_ratio is unusable here.
    # assign_blocks runs once per recording, so this thins each recording's train blocks by the same
    # fraction rather than dropping whole recordings.
    _tf = os.environ.get('BM_TRAIN_FRAC')
    if _tf and float(_tf) < 1.0:
        _frac = float(_tf)
        if not 0.0 < _frac <= 1.0:
            raise ValueError(f'BM_TRAIN_FRAC must be in (0,1], got {_frac}')
        _train_ind = int(assigned_blocks.split.max())
        _tr = list(assigned_blocks.index[assigned_blocks.split == _train_ind])
        _rng = random.Random(int(os.environ.get('BM_TRAIN_FRAC_SEED', '0')))
        _keep = set(_rng.sample(_tr, int(round(len(_tr) * _frac))))
        assigned_blocks = assigned_blocks.drop(index=[k for k in _tr if k not in _keep])
        # bm/events.py has no module logger; print lands in the dora run log all the same.
        print(f'[BM_TRAIN_FRAC] {_frac:.4f}: kept {len(_keep)}/{len(_tr)} train blocks '
              f'(valid/test untouched)', flush=True)

    return assigned_blocks


def split_wav_as_block("""
src = src.replace(ANCHOR, INJECT, 1)

ast.parse(src)
shutil.copy(E, E + ".orig_trainfrac")
open(E, "w").write(src)
print("[patched] BM_TRAIN_FRAC subsamples the train split only")
