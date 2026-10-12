#!/usr/bin/env python3
"""How much of the GPT-2 layer-6 target does lexical-surface timing explain? (paper's §intro claim)

Cross-validated held-out-story R^2 of the 8 local timing features predicting the frozen target.
One subject (s01) suffices: targets are STIMULUS-tied (mean of per-word GPT-2 embeddings), so they
are identical across subjects hearing the same story -- the target<->timing relationship is a
property of the stimulus, not the brain. Run on the cluster. Result: mean held-out R^2 = 0.215.
"""
import sys; sys.path.insert(0,"src")
import numpy as np, pandas as pd, json
from train_nuisance_seg import build_nuisance_segments
DATA="$SCRATCH/project/meg_papers/data/meg_masc_full"
meta = pd.read_csv(DATA+"/trial_metadata.tsv", sep="\t", dtype={"subject":str,"session":str,"task":str,"run":str},
                   usecols=lambda c: c in {"trial_index","subject","session","task","run","onset_s","duration_s","speech_rate","word_text"})
ctx=np.load("results/m0/masc_gpt2_contextual.npy", mmap_mode="r")
cm=json.load(open("results/m0/masc_gpt2_contextual_meta.json")); li=cm["layers_kept"].index(6)
# pool 4 stories, subject 01 -> more segments, honest held-out R^2
res={}
for held in ["0","1","2","3"]:
    tr=(meta["task"].values.astype(str)!=held)&(meta["subject"].values.astype(str)=="01")
    te=(meta["task"].values.astype(str)==held)&(meta["subject"].values.astype(str)=="01")
    Ftr,Ytr,_,_=build_nuisance_segments(meta,tr,ctx,li,3.0,3)
    Fte,Yte,_,_=build_nuisance_segments(meta,te,ctx,li,3.0,3)
    mu=Ftr[:,:8].mean(0); sd=Ftr[:,:8].std(0)+1e-9
    Fz=(Ftr[:,:8]-mu)/sd; Fzt=(Fte[:,:8]-mu)/sd
    ym=Ytr.mean(0); Yc=Ytr-ym; Yct=Yte-ym
    lam=10.0; Fa=np.concatenate([Fz,np.ones((len(Fz),1))],1)
    W=np.linalg.solve(Fa.T@Fa+lam*np.eye(Fa.shape[1]), Fa.T@Yc)
    Fat=np.concatenate([Fzt,np.ones((len(Fzt),1))],1); Yhat=Fat@W
    r2=1-((Yct-Yhat)**2).sum()/((Yct)**2).sum()
    res[held]=r2
    print("held-out story %s: cross-validated R^2 = %.3f  (n_tr=%d n_te=%d)" % (held,r2,len(Ftr),len(Fte)))
print("MEAN held-out R^2 = %.3f" % np.mean(list(res.values())))
