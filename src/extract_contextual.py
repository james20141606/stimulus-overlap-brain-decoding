#!/usr/bin/env python
"""Extract CONTEXTUAL per-occurrence LLM hidden states for every word-epoch.

Generalised from Paper 4's `p3_extract_contextual_gpt2.py` to support both cohort
metadata schemas (auto-detected from trial_metadata.tsv columns):

  * LibriBrain : sentence group = (subject,session,task,run,sentenceidx),
                 order by `wordidx`
  * MEG-MASC   : sentence group = (subject,session,task,run,story,sequence_id),
                 order by `word_index`

WHY THIS EXISTS: the cached `llm/gpt2/layer_12.npy` targets are *type-level* —
every occurrence of "the" gets a byte-identical vector (cosine 1.0000), which
makes contrastive retrieval ill-posed. We reconstruct each sentence, forward it
through the LLM, and read the layer-k hidden at the LAST sub-token of each word,
aligned to trial_index (== epoch row order).

Output: {out_dir}/{tag}_contextual.npy  (N, n_layers_kept, d) fp16
        {out_dir}/{tag}_contextual_meta.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd


def detect_schema(cols):
    if "sentenceidx" in cols and "wordidx" in cols:
        return dict(group=["subject", "session", "task", "run", "sentenceidx"],
                    order="wordidx", schema="libribrain")
    if "sequence_id" in cols and "word_index" in cols:
        grp = ["subject", "session", "task", "run", "story", "sequence_id"]
        grp = [c for c in grp if c in cols]
        return dict(group=grp, order="word_index", schema="meg_masc")
    raise SystemExit(f"unrecognised metadata schema; cols={sorted(cols)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--tag", default="gpt2")
    ap.add_argument("--max_sentences", type=int, default=0, help="0=all")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hdr = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t", nrows=0).columns
    sch = detect_schema(set(hdr))
    usecols = ["trial_index", "word_text"] + sch["group"] + [sch["order"]]
    usecols = list(dict.fromkeys(usecols))
    meta = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t", usecols=usecols)
    N = len(meta)
    assert (meta["trial_index"].values == np.arange(N)).all(), \
        "trial_index must equal epoch row order"
    meta["word_text"] = meta["word_text"].fillna("").astype("object").map(str)
    print(f"[data] {N} word-epochs, schema={sch['schema']} group={sch['group']}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model,
                                      output_hidden_states=True).to(device).eval()
    d_model = model.config.hidden_size
    n_layers_total = (getattr(model.config, "n_layer", None)
                      or model.config.num_hidden_layers) + 1
    layers = [l for l in args.layers if l < n_layers_total]
    print(f"[model] {args.model} d={d_model} keep layers {layers}", flush=True)

    out = np.lib.format.open_memmap(
        out_dir / f"{args.tag}_contextual.npy", mode="w+",
        dtype=np.float16, shape=(N, len(layers), d_model))
    filled = np.zeros(N, dtype=bool)

    groups = meta.groupby(sch["group"], sort=False)
    n_groups = len(groups)
    t0 = time.time(); n_done = 0
    with torch.no_grad():
        for gi, (_, sub) in enumerate(groups):
            if args.max_sentences and gi >= args.max_sentences:
                break
            sub = sub.sort_values(sch["order"])
            gidx = sub["trial_index"].values
            words = sub["word_text"].tolist()
            text = ""; spans = []
            for i, w in enumerate(words):
                w = "" if w is None else str(w)
                if i > 0:
                    text += " "
                s = len(text); text += w; spans.append((s, len(text)))
            enc = tok(text, return_offsets_mapping=True)
            ids = enc["input_ids"]; offs = enc["offset_mapping"]
            if len(ids) == 0:
                continue
            if len(ids) > 1024:
                ids = ids[:1024]; offs = offs[:1024]
            last_tok = np.full(len(words), -1, dtype=np.int64)
            for wj, (ws, we) in enumerate(spans):
                for ti, (cs, ce) in enumerate(offs):
                    if ce > cs and cs < we and ce > ws:
                        last_tok[wj] = ti
            ids_t = torch.tensor([ids], device=device)
            hs = model(input_ids=ids_t).hidden_states
            stack = torch.stack([hs[l][0] for l in layers], dim=0)  # (Lk,T,d)
            stack = stack.to(torch.float16).cpu().numpy()
            for wj in range(len(words)):
                t_idx = last_tok[wj]
                if t_idx < 0:
                    t_idx = 0 if wj == 0 else max(0, int(last_tok[max(0, wj - 1)]))
                out[gidx[wj]] = stack[:, t_idx, :]
                filled[gidx[wj]] = True
            n_done += len(words)
            if (gi + 1) % 1000 == 0:
                print(f"  {gi+1}/{n_groups} sentences, {n_done} words, "
                      f"{time.time()-t0:.0f}s", flush=True)

    out.flush()
    nfill = int(filled.sum())
    (out_dir / f"{args.tag}_contextual_meta.json").write_text(json.dumps({
        "model": args.model, "d_model": int(d_model), "layers_kept": layers,
        "n_epochs": N, "n_filled": nfill, "shape": list(out.shape),
        "dtype": "float16", "schema": sch["schema"],
        "elapsed_sec": time.time() - t0,
        "note": "contextual per-occurrence hidden at last sub-token of each word",
    }, indent=2))
    print(f"[save] {out_dir}/{args.tag}_contextual.npy shape={out.shape} "
          f"filled={nfill}/{N} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
