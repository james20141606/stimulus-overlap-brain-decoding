"""Honest retrieval metrics for M1: temporal-adjacency-guarded pool + bootstrap CIs.

M0 used a naive full-pool diagonal rank (Paper-4 metric). Two rigor upgrades:
  * adjacency guard: when ranking test item i, mask candidates from the SAME
    recording within +/-adj_s seconds, so top-k cannot be won by temporal/semantic
    proximity of neighbouring segments.
  * bootstrap CI: resample queries to get a 95% CI on top-1/top-10 (test pools are
    small, N~1-2k segments, so point estimates need error bars).
"""
from __future__ import annotations
import numpy as np


def _block_boot_ci(hits, order, block_len=10, n_boot=1000, seed=0):
    """Moving-block bootstrap on per-query top-k hit indicators, ordered in time,
    to account for within-recording temporal/linguistic autocorrelation that makes
    i.i.d. per-segment bootstrap CIs anti-conservative. Returns [lo, hi] on the rate."""
    h = np.asarray(hits, dtype=np.float64)[order]
    N = len(h)
    bl = int(min(max(2, block_len), N))
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(N / bl))
    smax = N - bl
    out = np.empty(n_boot)
    blk = np.arange(bl)
    for b in range(n_boot):
        starts = rng.integers(0, smax + 1, nblk)
        idx = (starts[:, None] + blk[None, :]).ravel()[:N]
        out[b] = h[idx].mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def multi_shuffle_null(zm, zl, n_draws=100, seed=0):
    """Multi-draw shuffled-label null (naive, unguarded): permute brain<->text n_draws
    times and report the distribution of top-10 x-chance. A single draw is noisy at
    N~1-4k; this tightens the null for small effects. Returns mean/sd/95th-percentile."""
    zm = np.asarray(zm); zl = np.asarray(zl); N = zm.shape[0]
    sim = zm @ zl.T
    rng = np.random.default_rng(seed)
    xs = np.empty(n_draws)
    ar = np.arange(N)
    for d in range(n_draws):
        perm = rng.permutation(N)
        tgt = sim[ar, perm]
        ranks = 1 + np.sum(sim > tgt[:, None], axis=1)
        xs[d] = float(np.mean(ranks <= 10)) * N / 10.0
    return {"n_draws": int(n_draws), "null_mean_x": float(xs.mean()),
            "null_sd_x": float(xs.std()), "null_p95_x": float(np.percentile(xs, 95))}


def matched_pool_metric(zm, zl, feats, rec_ids, onsets, pool_m=50, adj_s=10.0,
                        n_boot=1000, seed=0):
    """Timing-matched retrieval — the R6-CRITICAL-2 control.

    For each query i the candidate pool is the `pool_m` segments whose (standardized) LOCAL
    nuisance features are NEAREST to the query's, EXCLUDING same-recording neighbours within
    +-adj_s (proximity) and always including the true target. Because the foils are matched on
    word-count / duration / speech-rate / word-length, a timing-only model is ~chance among
    them by construction; if the brain still clears chance, it is using content beyond
    lexical-surface timing. chance top-k = k / pool_size.

    feats: (N, k) LOCAL nuisance features (pass WITHOUT onset_pos, else foils collapse to the
    same story position = same content). Standardized internally."""
    zm = np.asarray(zm); zl = np.asarray(zl)
    F = np.asarray(feats, dtype=np.float64)
    N = zm.shape[0]
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    sim = zm @ zl.T
    rec = np.asarray(rec_ids); on = np.asarray(onsets, dtype=np.float64)
    fsq = (F * F).sum(1)
    D2 = fsq[:, None] + fsq[None, :] - 2.0 * (F @ F.T)     # pairwise feature dist^2
    ranks = np.empty(N, dtype=np.int64); pools = np.empty(N, dtype=np.int64)
    for i in range(N):
        excl = (rec == rec[i]) & (np.abs(on - on[i]) <= adj_s)
        excl[i] = False
        d = D2[i].copy()
        d[excl] = np.inf
        d[i] = -np.inf                                      # force true target into the pool
        m = int(min(pool_m, int(np.sum(np.isfinite(d)))))
        cand = np.argpartition(d, m - 1)[:m]
        srow = sim[i, cand]
        ranks[i] = 1 + int(np.sum(srow > sim[i, i]))
        pools[i] = m
    mean_pool = float(np.mean(pools))

    def topk(k):
        return float(np.mean(ranks <= k))

    rng = np.random.default_rng(seed)
    b10 = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, N, N)
        b10[b] = np.mean(ranks[s] <= 10)
    return {
        "n_test": int(N), "pool_m": mean_pool,
        "top1": topk(1), "top5": topk(5), "top10": topk(10),
        "top10_ci95": [float(np.percentile(b10, 2.5)), float(np.percentile(b10, 97.5))],
        "median_rank": float(np.median(ranks)),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / max(mean_pool - 1, 1))),
        "chance_top1": 1.0 / mean_pool, "chance_top10": 10.0 / mean_pool,
        "top1_over_chance": float(topk(1) * mean_pool),
        "top10_over_chance": float(topk(10) * mean_pool / 10.0),
    }


def pool_size_sweep(zm, zl, rec_ids, onsets, adj_s=10.0, pool_sizes=(50, 100, 200, 500, 1000),
                    n_rep=20, seed=0):
    """Is x-chance comparable across datasets with different pool sizes?  No -- and this measures it.

    x-chance = accuracy / (10/N) is meant to normalize away the pool, but it over-corrects: chance
    falls as 1/N while accuracy falls more slowly, so the SAME model scores a higher x-chance simply
    for being evaluated against more candidates. Across Narratives' 26 stories x-chance tracks
    ln(pool) at r=0.97 -- but there story length and pool size are confounded, so that is only
    suggestive.

    This isolates it: fix the model, fix the story, and randomly subsample the candidate pool to
    size M (the true target always retained, the +-adj_s guard still applied). Any change in
    x-chance is then attributable to M alone.
    """
    zm = np.asarray(zm); zl = np.asarray(zl)
    N = zm.shape[0]
    sim = zm @ zl.T
    rec = np.asarray(rec_ids); on = np.asarray(onsets, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = {}
    for M in pool_sizes:
        if M > N:
            continue
        accs = []
        n_elig = 0
        for rep in range(n_rep):
            hits = 0; elig = 0
            for i in range(N):
                adj = (rec == rec[i]) & (np.abs(on - on[i]) <= adj_s)
                adj[i] = False
                cand = np.flatnonzero(~adj)
                cand = cand[cand != i]
                if len(cand) < M - 1:
                    continue      # too few admissible distractors to build a size-M pool for i
                elig += 1
                sub = rng.choice(cand, size=M - 1, replace=False)
                sub = np.append(sub, i)                       # true target always in the pool
                r = 1 + int(np.sum(sim[i, sub] > sim[i, i]))
                hits += int(r <= 10)
            if elig == 0:
                break
            # Divide by ELIGIBLE queries, not N. Dividing by N would silently score every
            # ineligible query as a miss, which depresses accuracy exactly at the large-M end --
            # i.e. it would manufacture part of the very pool effect this function measures.
            accs.append(hits / elig)
            n_elig = elig
        if not accs:
            continue
        a = float(np.mean(accs))
        out[str(M)] = {"pool": int(M), "top10": a, "top10_over_chance": a * M / 10.0,
                       "n_eligible": int(n_elig), "n_total": int(N),
                       "top10_sd_over_reps": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}
    return out


def guarded_retrieval_metrics(zm, zl, rec_ids, onsets, adj_s=10.0,
                              n_boot=1000, seed=0, block_boot=False, block_len=10,
                              stim_ids=None):
    """zm, zl: (N,D) L2-normalized (torch or numpy). rec_ids: (N,) hashable per
    recording. onsets: (N,) float seconds (segment mean onset).

    stim_ids: (N,) hashable per STIMULUS (e.g. the story/task id). Strongly recommended.

    Why it matters: the default guard masks same-`rec_ids` neighbours. If a subject hears the same
    stimulus more than once, the two presentations are different recordings, so every masked
    neighbour keeps an UNMASKED twin at the same stimulus time -- identical nuisance features,
    identical target -- and the guard cannot exceed a structural ceiling however wide it is set.
    MEG-MASC is exactly this case (9/10 of our subjects heard story 0 in two sessions). Passing
    `stim_ids` masks by stimulus identity + stimulus time instead, which is what the guard is
    supposed to do."""
    zm = np.asarray(zm); zl = np.asarray(zl)
    N = zm.shape[0]
    sim = zm @ zl.T                                  # (N, N)
    rec = np.asarray(rec_ids)
    on = np.asarray(onsets, dtype=np.float64)
    stim = None if stim_ids is None else np.asarray(stim_ids)
    ranks = np.empty(N, dtype=np.int64)
    pools = np.empty(N, dtype=np.int64)
    for i in range(N):
        scope = (rec == rec[i]) if stim is None else (stim == stim[i])
        adj = scope & (np.abs(on - on[i]) <= adj_s)
        adj[i] = False                               # keep the true target
        row = sim[i].copy()
        row[adj] = -np.inf
        valid = ~np.isinf(row)
        ranks[i] = 1 + int(np.sum(row[valid] > row[i]))
        pools[i] = int(np.sum(valid))
    mean_pool = float(np.mean(pools))

    def topk(k):
        return float(np.mean(ranks <= k))

    rng = np.random.default_rng(seed)
    b1 = np.empty(n_boot); b10 = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, N, N)
        b1[b] = np.mean(ranks[s] <= 1)
        b10[b] = np.mean(ranks[s] <= 10)
    out = {
        "n_test": int(N), "mean_pool_size": mean_pool,
        "top1": topk(1), "top5": topk(5), "top10": topk(10),
        "top1_ci95": [float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))],
        "top10_ci95": [float(np.percentile(b10, 2.5)), float(np.percentile(b10, 97.5))],
        "median_rank": float(np.median(ranks)),
        "mean_percentile": float(np.mean(1.0 - (ranks - 1) / np.maximum(pools - 1, 1))),
        "chance_top1": 1.0 / mean_pool, "chance_top10": 10.0 / mean_pool,
        "top1_over_chance": float(topk(1) * mean_pool),
        "top10_over_chance": float(topk(10) * mean_pool / 10.0),
    }
    if block_boot:
        order = np.lexsort((on, rec))
        bci = _block_boot_ci(ranks <= 10, order, block_len, n_boot, seed)
        out["top10_blockci95"] = bci
        out["top10_blockci95_x"] = [float(bci[0] * mean_pool / 10.0),
                                    float(bci[1] * mean_pool / 10.0)]
    return out
