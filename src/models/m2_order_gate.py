# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A9-B M2 ORDER GATE (v2 with both shuffle controls).
Task (order-dependent by construction): CONTIGUOUS run of K dropped blocks (y=0) vs SAME K
blocks SCATTERED (y=1). Global QC identical -> chance. Controls:
 C1a fixed block permutation (leaky: memorizable), C1b PER-SAMPLE random permutation (clean).
CPU only. seed=20260529."""
import os, json, time
import numpy as np
SEED=20260529; np.random.seed(SEED); RNG=np.random.default_rng(SEED)
OUT="genome_behavior"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
X=np.load(f"{OUT}/block_features.npy"); feat=json.load(open(f"{OUT}/feature_names.json"))
n,B,F=X.shape; K=15; log(f"X{X.shape} K={K}")
drop=np.zeros(F,dtype=np.float32); drop[feat.index("missing_rate")]=1.0; drop[feat.index("n_snps_norm")]=1.0
def make(xi,contig):
    x=xi.copy(); idx=np.arange(s:=RNG.integers(0,B-K),s+K) if contig else RNG.choice(B,K,replace=False)
    x[idx]=drop; return x
Xs=[];y=[];base=[]
for i in range(n):
    Xs.append(make(X[i],True)); y.append(0); base.append(i)
    Xs.append(make(X[i],False)); y.append(1); base.append(i)
Xs=np.array(Xs,dtype=np.float32); y=np.array(y); base=np.array(base)
perm=RNG.permutation(n); cut=int(0.8*n); trb=set(perm[:cut].tolist())
tr=np.array([k for k in range(len(y)) if base[k] in trb]); te=np.array([k for k in range(len(y)) if base[k] not in trb])
mu=Xs[tr].reshape(-1,F).mean(0); sd=Xs[tr].reshape(-1,F).std(0)+1e-8; Xn=(Xs-mu)/sd
log(f"train={len(tr)} test={len(te)}")
def auroc(yt,sc):
    yt=np.asarray(yt); sc=np.asarray(sc); n1=int((yt==1).sum()); n0=int((yt==0).sum())
    if n1==0 or n0==0: return float('nan')
    o=np.argsort(sc); rk=np.empty(len(sc)); rk[o]=np.arange(1,len(sc)+1)
    return float((rk[yt==1].sum()-n1*(n1+1)/2)/(n1*n0))
def psh(A):  # per-sample independent block shuffle
    o=A.copy()
    for i in range(len(o)): o[i]=o[i][RNG.permutation(A.shape[1])]
    return o
import torch, torch.nn as nn
torch.manual_seed(SEED); torch.set_num_threads(4)
def train_eval(model,Xtr,Xte):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()
    xt=torch.tensor(Xtr).permute(0,2,1); yt=torch.tensor(y[tr],dtype=torch.float32); xe=torch.tensor(Xte).permute(0,2,1)
    for ep in range(40):
        model.train(); o=torch.randperm(len(xt))
        for s in range(0,len(xt),128):
            b=o[s:s+128]; opt.zero_grad(); l=lf(model(xt[b]).ravel(),yt[b]); l.backward(); opt.step()
    model.eval()
    with torch.no_grad(): return auroc(y[te],model(xe).ravel().numpy())
class TCN(nn.Module):
    def __init__(s,F):
        super().__init__(); s.c=nn.Sequential(nn.Conv1d(F,32,7,padding=3),nn.ReLU(),nn.Conv1d(32,32,7,padding=3),nn.ReLU()); s.h=nn.Linear(32,1)
    def forward(s,x): return s.h(s.c(x).max(dim=2).values)
class QCsum(nn.Module):
    def __init__(s,F):
        super().__init__(); s.net=nn.Sequential(nn.Linear(2*F,32),nn.ReLU(),nn.Linear(32,1))
    def forward(s,x): return s.net(torch.cat([x.mean(2),x.std(2)],dim=1))
sh=np.load(f"{OUT}/block_order_shuffle.npy")
res={}
res["M0_QC_summary"]=train_eval(QCsum(F),Xn[tr],Xn[te])
res["M2_Genome_TCN"]=train_eval(TCN(F),Xn[tr],Xn[te])
res["M2_TCN_fixed_shuffle_C1a"]=train_eval(TCN(F),Xn[tr][:,sh,:],Xn[te][:,sh,:])
res["M2_TCN_persample_shuffle_C1b"]=train_eval(TCN(F),psh(Xn[tr]),psh(Xn[te]))
log("=== AUROC (contiguous vs scattered; same K dropped blocks) ===")
for k,v in res.items(): log(f"  {k}: AUROC={v:.4f}")
json.dump(res,open(f"{OUT}/order_gate_auroc.json","w")); log("M2 ORDER GATE v2 DONE")
