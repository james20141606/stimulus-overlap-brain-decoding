#!/usr/bin/env python
"""MB-1: dynamic-token vs static-patch, LOW-SHOT CROSS-SUBJECT transfer (within MASC).

Pretrain an encoder variant on N-1 subjects, then few-shot calibrate on k segments of
a held-out subject and evaluate that subject's held-out recordings with the honest
adjacency-guarded metric. Compare, at each k:
  * FM      : pretrained -> fine-tune all params (low LR) on k calib segments;
  * SCRATCH : fresh encoder trained on the same k (no pretraining).
Run for variant in {static, dynamic, ...}. Headline = FM curve (x-chance vs k),
static vs dynamic. Held-out subject's calib and eval come from DISJOINT recordings.

Reuses the M1a segment builder and the InfoNCE head; swaps only the encoder.
"""
from __future__ import annotations
import argparse, json, time, copy
from pathlib import Path
import numpy as np
import numpy as _np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_infonce import LLMHead, retrieval_metrics
from train_m1a_seg import build_segments_with_meta
from eval_metrics import guarded_retrieval_metrics
from tokenizers import build_encoder


@torch.no_grad()
def embed_mb(enc, llm, X, Y, device, bs=1024):
    enc.eval(); llm.eval()
    zm, zl = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yb = torch.from_numpy(Y[i:i + bs]).to(device)
        zm.append(F.normalize(enc(xb)[0], dim=-1).cpu())
        zl.append(F.normalize(llm(yb), dim=-1).cpu())
    return torch.cat(zm), torch.cat(zl)


def train_one(enc, llm, X, Y, device, epochs, lr, wd, batch, Xva, Yva, seed, tag):
    ls = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(llm.parameters()) + [ls],
                            lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)
    n = len(X); best = -1.0; best_state = None
    for ep in range(epochs):
        enc.train(); llm.train()
        perm = rng.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 8:
                continue
            xb = torch.from_numpy(X[b]).to(device)
            yb = torch.from_numpy(Y[b]).to(device)
            zm, aux = enc(xb); zm = F.normalize(zm, dim=-1)
            zl = F.normalize(llm(yb), dim=-1)
            sc = ls.clamp(max=np.log(100)).exp()
            logits = sc * zm @ zl.T; lab = torch.arange(len(b), device=device)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) + aux
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        zm, zl = embed_mb(enc, llm, Xva, Yva, device)
        vt = retrieval_metrics(zm, zl)["top10"]
        if vt > best:
            best = vt
            best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(llm.state_dict()))
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"    [{tag} ep{ep:03d}] loss={tot/max(nb,1):.4f} val={vt:.4f}", flush=True)
    if best_state:
        enc.load_state_dict(best_state[0]); llm.load_state_dict(best_state[1])
    return best


def eval_guarded(enc, llm, X, Y, rec, on, device, adj_s, seed, guard_scope="recording"):
    zm, zl = embed_mb(enc, llm, X, Y, device)
    if guard_scope == "stimulus":
        sid, son = stimulus_scope(rec, on)
        g = guarded_retrieval_metrics(zm, zl, rec, son, adj_s=adj_s, seed=seed, stim_ids=sid)
    else:
        g = guarded_retrieval_metrics(zm, zl, rec, on, adj_s=adj_s, seed=seed)
    ctrl = retrieval_metrics(zm, zl[torch.randperm(zl.shape[0])])
    return {"x": g["top10_over_chance"], "ci": [round(v * 100, 2) for v in g["top10_ci95"]],
            "top1x": round(g["top1_over_chance"], 2), "medrank": round(g["median_rank"], 1),
            "meanpct": round(g["mean_percentile"], 3), "pool": round(g["mean_pool_size"], 0),
            "shuf": round(ctrl["top10_over_chance"], 2), "n": g["n_test"]}


def _story_of(rec):
    """Story id from a recording key 'subject|session|task|run' (REC_KEYS order); task == story."""
    return np.array([str(r).split("|")[2] if "|" in str(r) else str(r) for r in rec])


def stimulus_scope(rec, onsets):
    """(stim_ids, stimulus_time) for a STIMULUS-scoped guard.

    The default guard masks neighbours from the same RECORDING. MEG-MASC presents each story in two
    sessions, so for every masked neighbour there is an unmasked TWIN at the same stimulus time in the
    other session, with an identical target -- the guard leaves the shortcut partly open. Keying on
    story identity plus time-within-story masks the twin too. Same fix as ADR 0001 for paper 1.
    """
    rec = np.asarray(rec)
    on = np.asarray(onsets, dtype=np.float64).copy()
    for r in np.unique(rec):
        m = rec == r
        on[m] -= on[m].min()                 # onset relative to that presentation's start
    return _story_of(rec), on


def split_by_story(rec, frac_eval=0.5, seed=0):
    """Assign whole STORIES to calib vs eval, so the two never share a stimulus.

    split_by_recording only guarantees disjoint recordings, and since each story is recorded twice,
    9/10 held-out subjects ended up with a story on BOTH sides -- part of what the calibration budget
    buys is then stimulus familiarity rather than subject adaptation. Splitting by story removes it.
    """
    st = _story_of(rec)
    stories = sorted(set(st.tolist()))
    rng = np.random.default_rng(seed); rng.shuffle(stories)
    n_eval = max(1, int(round(len(stories) * frac_eval)))
    ev_s = set(stories[:n_eval]); ca_s = set(stories[n_eval:])
    if not ca_s:
        ca_s = ev_s
    return np.array([s in ca_s for s in st]), np.array([s in ev_s for s in st])


def split_by_recording(rec, frac_eval=0.5, seed=0):
    """Assign whole recordings to calib vs eval so the two never share a recording."""
    recs = sorted(set(rec.tolist()))
    rng = np.random.default_rng(seed); rng.shuffle(recs)
    n_eval = max(1, int(round(len(recs) * frac_eval)))
    eval_recs = set(recs[:n_eval]); calib_recs = set(recs[n_eval:])
    if not calib_recs:                                   # subject has 1 recording -> fall back
        calib_recs = eval_recs
    ev = np.array([r in eval_recs for r in rec])
    ca = np.array([r in calib_recs for r in rec])
    return ca, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--ctx_meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_subject", default="01")
    ap.add_argument("--variants", default="static,dynamic")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--t0", type=int, default=50)
    ap.add_argument("--t1", type=int, default=250)
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--seg_s", type=float, default=3.0)
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--adj_s", type=float, default=10.0)
    ap.add_argument("--klist", default="100,200,500,1000")
    ap.add_argument("--val_n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pre_epochs", type=int, default=80)
    ap.add_argument("--adapt_epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adapt_lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dose_target", type=int, default=0, help="segment count from the dose=1.0 run; backfill up to it")
    ap.add_argument("--dose", type=float, default=None, help="fraction of other subjects keeping the eval story in pretraining (0..1); volume held fixed by backfill")
    ap.add_argument("--calib_stories", default="", help="comma list; pin calibration stories for a diversity-matched comparison")
    ap.add_argument("--eval_story", default="", help="story to evaluate on; default = holdout_story")
    ap.add_argument("--scratch_grid", default="", help="lr1,lr2:wd1,wd2 -> tuned scratch baseline")
    ap.add_argument("--holdout_story", default="", help="story held out of PRETRAINING and calib; used as eval")
    ap.add_argument("--guard_scope", default="recording", choices=["recording", "stimulus"])
    ap.add_argument("--split_by", default="recording", choices=["recording", "story"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="mb1")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.klist.split(",")]
    variants = args.variants.split(",")

    meta = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t",
                       dtype={"subject": str, "session": str, "task": str, "run": str},
                       usecols=lambda c: c in
                       {"trial_index", "subject", "session", "task", "run", "onset_s"})
    cmeta = json.loads(Path(args.ctx_meta).read_text())
    li = cmeta["layers_kept"].index(args.layer)
    ctx = np.load(args.ctx, mmap_mode="r")
    epochs = np.load(data_dir / "epochs_data.npy", mmap_mode="r")
    n_ch = epochs.shape[1]

    subj = meta["subject"].values.astype(str)
    tr_mask = subj != str(args.test_subject)
    te_mask = subj == str(args.test_subject)
    if args.holdout_story:
        # STIMULUS-HELD-OUT PRETRAINING. Without this, pretraining sees all 4 stories from the other
        # 26 subjects, so the evaluation stimuli have been seen (from other brains) before the held-out
        # subject is ever touched. A model can then succeed by matching a stimulus-locked template
        # learned across subjects -- time-point identification, not language decoding. That is exactly
        # the confound an earlier analysis shows drives retrieval to ceiling.
        # With this flag the named story is removed from pretraining AND from calibration, and is used
        # as the evaluation set: no brain in the training set has ever heard it.
        # --holdout_story accepts a COMMA LIST. Why that matters: with a single withheld story you
        # cannot hold both pretraining VOLUME and pretraining COMPOSITION fixed while removing the
        # evaluation stimulus. Excluding story 0 and substituting story 3 keeps the volume but swaps
        # which third story is present, so a difference could come from story 3 being a poorer
        # pretraining stimulus rather than from story 0's absence. Excluding both 0 and 3 keeps the
        # composition of the shared part identical and shrinks the volume instead. Reporting both
        # triangulates: only an effect present in each is attributable to the evaluation stimulus.
        hold = {s.strip() for s in str(args.holdout_story).split(",") if s.strip()}
        story_col = meta["task"].values.astype(str)
        if args.dose is not None:
            # DOSE-RESPONSE. Instead of removing the evaluation story from pretraining entirely, keep
            # a fraction `dose` of the OTHER SUBJECTS who heard it, and backfill the excluded subjects
            # with the same subjects' recordings of a different story. Pretraining volume, subject
            # composition and optimisation are therefore held fixed while only the AMOUNT of exposure
            # to the evaluation stimulus varies. Estimating the leakage curve directly beats inferring
            # it from two endpoints whose confounds run in opposite directions.
            ev_story = str(args.eval_story) if args.eval_story else sorted(hold)[0]
            subj_all = sorted(set(subj[tr_mask & (story_col == ev_story)]))
            rng_d = _np.random.default_rng(args.seed)
            n_keep = int(round(args.dose * len(subj_all)))
            keep_subj = set(rng_d.permutation(subj_all)[:n_keep].tolist())
            drop = tr_mask & (story_col == ev_story) & (~_np.isin(subj, list(keep_subj)))
            tr_mask = tr_mask & (~drop)
            args._dose_short = int(drop.sum())     # segments are backfilled after the build
            print(f"[mb1] DOSE {args.dose:.2f}: {len(keep_subj)}/{len(subj_all)} subjects keep story "
                  f"{ev_story} in pretraining ({int(drop.sum())} word rows removed; "
                  f"segment-count backfill happens after the build ONLY if --dose_target is set)",
                  flush=True)
        else:
            tr_mask = tr_mask & (~np.isin(story_col, list(hold)))
        print(f"[mb1] PRETRAIN EXCLUDES stories {sorted(hold)}: "
              f"train_words {int(tr_mask.sum())}", flush=True)
    Xtr, Ytr, _, _ = build_segments_with_meta(meta, tr_mask, epochs, ctx, li,
                                              args.t0, args.t1, args.decim, args.seg_s, args.min_words)
    if args.dose is not None:
        # VOLUME BACKFILL at the segment level. Removing the evaluation story from some subjects
        # removes segments; left alone, a lower dose would also mean less training data and the two
        # would be confounded -- precisely what this experiment exists to rule out. Resample the
        # surviving segments up to the count obtained at dose=1.0, so every dose trains on the same
        # number of segments from the same subjects and differs only in how much of the evaluation
        # stimulus they contain. `--dose_target` carries the dose=1.0 count; the run at dose=1.0
        # prints it.
        print(f"[mb1] dose={args.dose:.2f} built {len(Xtr)} pretraining segments", flush=True)
        if args.dose_target and len(Xtr) < args.dose_target:
            need = args.dose_target - len(Xtr)
            extra = _np.random.default_rng(args.seed).integers(0, len(Xtr), size=need)
            Xtr = _np.concatenate([Xtr, Xtr[extra]], axis=0)
            Ytr = _np.concatenate([Ytr, Ytr[extra]], axis=0)
            print(f"[mb1] backfilled {need} segments -> {len(Xtr)} (target {args.dose_target})",
                  flush=True)
    Xhs, Yhs, rec_hs, on_hs = build_segments_with_meta(meta, te_mask, epochs, ctx, li,
                                                       args.t0, args.t1, args.decim, args.seg_s, args.min_words)
    print(f"[mb1] n_ch={n_ch} train_seg={len(Xtr)} heldsubj_seg={len(Xhs)} "
          f"(subject={args.test_subject})", flush=True)

    # held-out subject: disjoint calib vs eval recordings
    if args.holdout_story:
        st_hs = _story_of(rec_hs)
        # --eval_story decouples WHICH story is withheld from pretraining from WHICH story we score on.
        # Default: evaluate on the withheld story (so no brain in training heard it).
        # Set it to a DIFFERENT story to hold pretraining VOLUME fixed while making the eval stimulus
        # one that pretraining DID see -- the control for "the collapse is just 25% less pretraining
        # data", which is otherwise confounded with the stimulus holdout.
        hold = {s.strip() for s in str(args.holdout_story).split(",") if s.strip()}
        # default eval target: the first withheld story, so single-story usage is unchanged
        ev_story = str(args.eval_story) if args.eval_story else sorted(hold)[0]
        ev = st_hs == ev_story
        ca = (~ev) & (~np.isin(st_hs, list(hold)))        # never calibrate on a withheld story
        if args.calib_stories:
            # CALIBRATION-DIVERSITY CONTROL. With --holdout_story 3, the SEEN arm (eval story 0)
            # calibrates on {1,2} while the UNSEEN arm (eval story 3) calibrates on {0,1,2}: the eval
            # story is excluded from calibration in both, but the NUMBER of distinct calibration
            # stories differs (2 vs 3). k is fixed so the segment count matches, yet stimulus diversity
            # does not. This flag pins the calibration stories explicitly so both arms can be made
            # identical on that axis.
            keep = set(str(args.calib_stories).split(","))
            ca = ca & np.isin(st_hs, list(keep))
            print(f"[mb1] calibration restricted to stories {sorted(keep)}", flush=True)
        print(f"[mb1] pretrain excludes story {args.holdout_story}; EVAL ON story {ev_story} "
              f"({'UNSEEN in pretraining' if ev_story == str(args.holdout_story) else 'SEEN in pretraining'})",
              flush=True)
    else:
        _splitter = split_by_story if args.split_by == "story" else split_by_recording
        ca, ev = _splitter(rec_hs, frac_eval=0.5, seed=args.seed)
    print(f"[mb1] split_by={args.split_by} guard_scope={args.guard_scope} "
          f"calib_seg={int(ca.sum())} eval_seg={int(ev.sum())}", flush=True)
    Xev, Yev, rec_ev, on_ev = Xhs[ev], Yhs[ev], rec_hs[ev], on_hs[ev]
    Xca, Yca = Xhs[ca], Yhs[ca]
    rng = np.random.default_rng(args.seed); p = rng.permutation(len(Xca))
    Xval, Yval = Xca[p[:args.val_n]], Yca[p[:args.val_n]]
    Xcal, Ycal = Xca[p[args.val_n:]], Yca[p[args.val_n:]]
    ks = [k for k in ks if k <= len(Xcal)]
    print(f"[mb1] eval_seg={len(Xev)} calib_pool={len(Xcal)} val={len(Xval)} k-sweep={ks}", flush=True)

    # pretrain val from train pool
    q = rng.permutation(len(Xtr)); pvn = min(1500, len(Xtr) // 10)
    PXv, PYv = Xtr[q[:pvn]], Ytr[q[:pvn]]
    PXt, PYt = Xtr[q[pvn:]], Ytr[q[pvn:]]

    all_results = {}
    for variant in variants:
        print(f"\n==== VARIANT {variant} ====", flush=True)
        enc = build_encoder(variant, n_ch).to(device)
        llm = LLMHead(768, 768).to(device)
        n_par = sum(p.numel() for p in enc.parameters())
        print(f"[mb1] {variant} encoder params={n_par/1e6:.2f}M", flush=True)
        train_one(enc, llm, PXt, PYt, device, args.pre_epochs, args.lr, args.wd,
                  args.batch, PXv, PYv, args.seed, f"pre-{variant}")
        pre = (copy.deepcopy(enc.state_dict()), copy.deepcopy(llm.state_dict()))
        base = eval_guarded(enc, llm, Xev, Yev, rec_ev, on_ev, device, args.adj_s, args.seed,
                            guard_scope=args.guard_scope)
        print(f"[mb1] {variant} pretrained zero-calib on held subj: top10 x{base['x']:.2f} "
              f"CI{base['ci']} | top1 x{base['top1x']:.2f} | medrank {base['medrank']}/"
              f"{base['pool']:.0f} | meanpct {base['meanpct']} | shuf x{base['shuf']}", flush=True)

        curve = []
        for k in ks:
            Xk, Yk = Xcal[:k], Ycal[:k]
            # FM: reload pretrained, low-LR fine-tune on k
            enc.load_state_dict(pre[0]); llm.load_state_dict(pre[1])
            train_one(enc, llm, Xk, Yk, device, args.adapt_epochs, args.adapt_lr,
                      args.wd, args.batch, Xval, Yval, args.seed, f"fm-{variant}-k{k}")
            fm = eval_guarded(enc, llm, Xev, Yev, rec_ev, on_ev, device, args.adj_s, args.seed,
                            guard_scope=args.guard_scope)
            # SCRATCH: fresh, train on k.
            # With --scratch_grid this becomes a TUNED baseline: sweep (lr, wd) and select the config
            # by VALIDATION top-10, then evaluate that config once on the held-out set. Four
            # independent 2025-26 audits (arXiv:2507.01196, 2605.26910, 2601.17883, 2605.14698) find
            # tuned supervised baselines matching or beating brain foundation models, so an untuned
            # from-scratch arm -- one lr, one wd, inherited from the fine-tuning config -- is a
            # strawman and the most likely reason a reviewer rejects a transfer claim.
            if args.scratch_grid:
                lr_s, wd_s = args.scratch_grid.split(":")
                lrs = [float(x) for x in lr_s.split(",")]
                wds = [float(x) for x in wd_s.split(",")]
                best_v, sc, best_cfg = -1.0, None, None
                for _lr in lrs:
                    for _wd in wds:
                        senc = build_encoder(variant, n_ch).to(device)
                        sllm = LLMHead(768, 768).to(device)
                        v = train_one(senc, sllm, Xk, Yk, device, args.adapt_epochs, _lr, _wd,
                                      args.batch, Xval, Yval, args.seed, f"sc-{variant}-k{k}-{_lr}-{_wd}")
                        if v > best_v:                     # selection on VALIDATION only
                            best_v, best_cfg = v, (_lr, _wd)
                            sc = eval_guarded(senc, sllm, Xev, Yev, rec_ev, on_ev, device,
                                              args.adj_s, args.seed, guard_scope=args.guard_scope)
                sc["tuned_lr"], sc["tuned_wd"], sc["tuned_val"] = best_cfg[0], best_cfg[1], round(best_v, 4)
                print(f"[mb1] {variant} k={k} TUNED scratch: lr={best_cfg[0]} wd={best_cfg[1]} "
                      f"val={best_v:.4f} -> x{sc['x']:.2f}", flush=True)
            else:
                senc = build_encoder(variant, n_ch).to(device); sllm = LLMHead(768, 768).to(device)
                train_one(senc, sllm, Xk, Yk, device, args.adapt_epochs, args.lr, args.wd,
                          args.batch, Xval, Yval, args.seed, f"sc-{variant}-k{k}")
                sc = eval_guarded(senc, sllm, Xev, Yev, rec_ev, on_ev, device, args.adj_s, args.seed,
                                guard_scope=args.guard_scope)
            curve.append({"k": k, "fm": fm, "scratch": sc})
            print(f"[mb1] {variant} k={k}: FM x{fm['x']:.2f} CI{fm['ci']} shuf x{fm['shuf']} "
                  f"| SCRATCH x{sc['x']:.2f} CI{sc['ci']}", flush=True)
        all_results[variant] = {"params_M": round(n_par / 1e6, 3),
                                "pretrained_zerocalib": base, "curve": curve}

    result = {"tag": args.tag, "test_subject": args.test_subject, "layer": args.layer,
              "seg_s": args.seg_s, "n_ch": int(n_ch), "klist": ks,
           "guard_scope": args.guard_scope, "split_by": args.split_by,
           "holdout_story": args.holdout_story, "eval_story": args.eval_story, "dose": args.dose,
           "calib_stories": args.calib_stories,
              "results": all_results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"result_{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(f"[save] {out_dir}/result_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
