# ADR 0001 — The adjacency guard is scoped to the stimulus, not the recording

Date: 2026-07-29
Status: Accepted

## Context

The guarded-retrieval metric masks, for each query segment, other candidates that are temporally
adjacent to it, so that top-$k$ cannot be won by the trivial similarity of neighbouring moments. Our
implementation (`src/eval_metrics.py`) masked candidates from the same **recording** within
$\pm$adj seconds:

```python
adj = (rec == rec[i]) & (np.abs(on - on[i]) <= adj_s)
```

where a recording is keyed by `["subject", "session", "task", "run"]`.

On 2026-07-28 a review pass established that this is unsound for MEG-MASC, and we verified it:

* 9 of our 10 subjects heard story 0 in **two sessions**; the two presentations are word-for-word
  identical (position-by-position `word_text` agreement `1.0000`), 654 words each.
* Because `session` is part of the recording key, the two presentations are *different recordings*.
* Therefore every neighbour the guard removes keeps an **unremoved twin** at the same stimulus time in
  the other session, with identical nuisance features and an identical target.

The consequence is that the guard's reach is structurally capped at roughly the query's own
presentation. Widening it from $\pm10$ s to $\pm240$ s removes 46% of the pool — close to that ceiling —
so "widening the guard never closes the gap" was partly guaranteed by construction, and the sweep could
not establish that the confound is global rather than merely local.

This also retro-explains a long-standing oddity in the notebook: subject 3, the only single-session
subject, has half the pool of the others (~94 vs ~189). That was recorded months earlier and never
connected to the guard.

## Decision

Guard on **stimulus identity and stimulus time**, not on recording:

```python
scope = (rec == rec[i]) if stim is None else (stim == stim[i])
adj = scope & (np.abs(on - on[i]) <= adj_s)
```

`guarded_retrieval_metrics` takes an optional `stim_ids`. When supplied, repeated presentations of the
same stimulus are inside the guard's scope, so both the exact twin and near neighbours in the other
presentation are masked. The recording-scoped behaviour remains the default so that no previously
reported number silently changes.

We additionally report a **collapsed-pool** variant (deduplicate so each stimulus segment appears once,
then guard normally) as a robustness check. If the stimulus-scoped and collapsed-pool readings agree,
the result is robust to how this question is framed at all.

## Alternatives considered

1. **Keep recording scope, delete only the exact twin.** Rejected: it fixes the symptom (the $\Delta=0$
   duplicate) while leaving the mechanism intact — a segment 5 s away in the other presentation still
   survives a $\pm240$ s guard, so the ceiling returns in weaker form. This is the kind of half-fix that
   fails the next review round.
2. **Collapse the pool only.** Clean, and we report it, but it changes pool size (≈180→≈90) and
   therefore every $\times$chance figure, while the stimulus-scoped guard leaves the pool definition
   intact and fixes the actual defect.
3. **Do nothing and disclose.** Rejected: this is the one defect a reader can find in our released code.
   Disclosing it is survivable; disclosing it *and having fixed it* is a contribution.

## Consequences

* **Not reversible cheaply.** Changing guard semantics makes $\times$chance figures non-comparable with
  every number we and others have already published under recording scope, which is why the old
  behaviour stays the default and the new one is opt-in.
* We must recommend this to the field, because any group using a recording-scoped guard on a dataset
  with repeated stimuli inherits the same ceiling. `§protocol` now leads its guard advice with it, and
  names it as the item we got wrong ourselves.
* The claim "the confound is global rather than merely short-range proximity" is **withdrawn pending
  re-measurement** under the new scope. The criteria for restoring or reversing it were pre-registered
  in `experiments_detail.md` (2026-07-29) *before* the re-run was submitted.
* Head-to-head results are unaffected: both arms are always scored in the same pool, and none of them
  relied on the guard removing anything.
* The ±2 s per-word jitter between presentations means stimulus-time alignment is approximate at the
  edges; immaterial at $\pm10$ s and above, and disclosed.
