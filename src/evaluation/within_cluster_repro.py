# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Reproduce tab:rankcal within-cluster (relatedness-controlled, sample-weighted) Pearson.
Method (a): within each (test fold, base-cluster) group, residualise the group mean from
y_true and y_pred; pool residuals; Pearson. base cluster = group_5fold/cluster_membership (40 fine).
Grid over {std,raw} x MINW to match locked values: GBLUP 0.356/0.255/0.260/0.265,
FullSNP-MLP 0.391/0.287/0.290/0.271 (random/group10/group5/lco)."""
import csv, numpy as np, sys
B="benchmarks/holstein"
CM="splits/holstein/relationship_group/group_5fold/cluster_membership.csv"
FAMS=["random","group10","group5","leave_cluster_out"]
MODELS=["GBLUP_full","FullSNP_MLP_top50k","Ridge_top50k","PCARidge_PCA80","SparseGate_MLP_top20k"]
TARGET={"GBLUP_full":[0.356,0.255,0.260,0.265],"FullSNP_MLP_top50k":[0.391,0.287,0.290,0.271]}

cl={}
for d in csv.DictReader(open(CM)): cl[int(d["sample_id"])]=int(d["cluster_id"])

# rows[(model,family)] -> list of (fold, sid, yt_std, yp_std, yt_raw, yp_raw)
rows={}
for d in csv.DictReader(open(f"{B}/holstein_per_sample.csv")):
    k=(d["model"],d["family"])
    rows.setdefault(k,[]).append((d["fold"],int(d["sample_id"]),
        float(d["y_true_std"]),float(d["y_pred_std"]),float(d["y_true_raw"]),float(d["y_pred_raw"])))

def pear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or a.std()<1e-9 or b.std()<1e-9: return float("nan")
    return float(np.corrcoef(a,b)[0,1])

# lco own 10-cluster map
CMLCO="splits/holstein/leave_cluster_out/cluster_membership.csv"
cl_lco={}
for d in csv.DictReader(open(CMLCO)): cl_lco[int(d["sample_id"])]=int(d["cluster_id"])

def cmap(family,which):
    if family=="leave_cluster_out" and which=="lco10": return cl_lco
    return cl

def within_pooledresid(model,family,use_std,minw,which="base40"):
    cm=cmap(family,which); rs=rows.get((model,family),[]); groups={}
    for fold,sid,yts,yps,ytr,ypr in rs:
        c=cm.get(sid);
        if c is None: continue
        yt,yp=(yts,yps) if use_std else (ytr,ypr); groups.setdefault((fold,c),[]).append((yt,yp))
    RT=[];RP=[]
    for g,v in groups.items():
        if len(v)<minw: continue
        yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]); RT+=list(yt-yt.mean());RP+=list(yp-yp.mean())
    return pear(RT,RP)

def within_percluster(model,family,use_std,minw,pool_folds=True,which="base40"):
    """method b: per-cluster Pearson, sample-weighted avg. pool_folds: pool a cluster across folds."""
    cm=cmap(family,which); rs=rows.get((model,family),[]); groups={}
    for fold,sid,yts,yps,ytr,ypr in rs:
        c=cm.get(sid)
        if c is None: continue
        yt,yp=(yts,yps) if use_std else (ytr,ypr)
        key=c if pool_folds else (fold,c); groups.setdefault(key,[]).append((yt,yp))
    num=0.0; den=0
    for g,v in groups.items():
        if len(v)<minw: continue
        yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]); p=pear(yt,yp)
        if not np.isnan(p): num+=len(v)*p; den+=len(v)
    return num/den if den else float("nan")

print("### method b: per-cluster Pearson, sample-weighted (pool_folds=True) ###")
for use_std in [True,False]:
    for minw in [2,3,5,10]:
        line=f"std={int(use_std)} minw={minw:2d} | "; ok=True
        for model in ["GBLUP_full","FullSNP_MLP_top50k"]:
            vals=[within_percluster(model,f,use_std,minw) for f in FAMS]; tg=TARGET[model]
            match=all(abs(v-t)<0.004 for v,t in zip(vals,tg)); ok=ok and match
            line+=f"{model[:6]}=["+",".join(f"{v:.3f}" for v in vals)+f"]{'OK' if match else 'x'} "
        print(line+("  <<< MATCH" if ok else ""))

print("### method b: per (fold,cluster) Pearson, sample-weighted (pool_folds=False) ###")
for use_std in [True,False]:
    for minw in [3,5,10]:
        line=f"std={int(use_std)} minw={minw:2d} | "; ok=True
        for model in ["GBLUP_full","FullSNP_MLP_top50k"]:
            vals=[within_percluster(model,f,use_std,minw,pool_folds=False) for f in FAMS]; tg=TARGET[model]
            match=all(abs(v-t)<0.004 for v,t in zip(vals,tg)); ok=ok and match
            line+=f"{model[:6]}=["+",".join(f"{v:.3f}" for v in vals)+f"]{'OK' if match else 'x'} "
        print(line+("  <<< MATCH" if ok else ""))

print("### lco with own 10-cluster map (method b pooled) ###")
for use_std in [True,False]:
    for minw in [2,5,10]:
        vals=[within_percluster(m,"leave_cluster_out",use_std,minw,which="lco10") for m in ["GBLUP_full","FullSNP_MLP_top50k"]]
        print(f"std={int(use_std)} minw={minw:2d} | lco GBLUP={vals[0]:.3f} FullSNP={vals[1]:.3f} (target 0.265/0.271)")

# ---- H1: 10-cluster lco map for ALL regimes, pooled-residual (method a) ----
def within_resid_map(model,family,use_std,minw,cm):
    rs=rows.get((model,family),[]); groups={}
    for fold,sid,yts,yps,ytr,ypr in rs:
        c=cm.get(sid)
        if c is None: continue
        yt,yp=(yts,yps) if use_std else (ytr,ypr); groups.setdefault((fold,c),[]).append((yt,yp))
    RT=[];RP=[]
    for g,v in groups.items():
        if len(v)<minw: continue
        yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]); RT+=list(yt-yt.mean());RP+=list(yp-yp.mean())
    return pear(RT,RP)
print("### H1: 10-cluster lco map ALL regimes, pooled-residual (method a) ###")
for use_std in [True,False]:
    for minw in [1,2,3]:
        line=f"std={int(use_std)} minw={minw} | "; ok=True
        for model in ["GBLUP_full","FullSNP_MLP_top50k"]:
            vals=[within_resid_map(model,f,use_std,minw,cl_lco) for f in FAMS]; tg=TARGET[model]
            match=all(abs(v-t)<0.004 for v,t in zip(vals,tg)); ok=ok and match
            line+=f"{model[:6]}=["+",".join(f"{v:.3f}" for v in vals)+f"]{'OK' if match else 'x'} "
        print(line+("  <<< MATCH" if ok else ""))

# ---- H2: 40-map, GLOBAL (regime-pooled) per-cluster mean residual then pool ----
def within_globalresid(model,family,use_std,cm,minc=1):
    rs=rows.get((model,family),[]); byc={}
    for fold,sid,yts,yps,ytr,ypr in rs:
        c=cm.get(sid)
        if c is None: continue
        yt,yp=(yts,yps) if use_std else (ytr,ypr); byc.setdefault(c,[]).append((yt,yp))
    RT=[];RP=[]
    for c,v in byc.items():
        if len(v)<minc: continue
        yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]); RT+=list(yt-yt.mean());RP+=list(yp-yp.mean())
    return pear(RT,RP)
print("### H2: 40-map GLOBAL per-cluster mean residual (pool across folds) ###")
for use_std in [True,False]:
    line=f"std={int(use_std)} | "; ok=True
    for model in ["GBLUP_full","FullSNP_MLP_top50k"]:
        vals=[within_globalresid(model,f,use_std,cl) for f in FAMS]; tg=TARGET[model]
        match=all(abs(v-t)<0.004 for v,t in zip(vals,tg)); ok=ok and match
        line+=f"{model[:6]}=["+",".join(f"{v:.3f}" for v in vals)+f"]{'OK' if match else 'x'} "
    print(line+("  <<< MATCH" if ok else ""))
print("### H2b: 10-map GLOBAL per-cluster mean residual ###")
for use_std in [True,False]:
    line=f"std={int(use_std)} | "
    for model in ["GBLUP_full","FullSNP_MLP_top50k"]:
        vals=[within_globalresid(model,f,use_std,cl_lco) for f in FAMS]; tg=TARGET[model]
        match=all(abs(v-t)<0.004 for v,t in zip(vals,tg))
        line+=f"{model[:6]}=["+",".join(f"{v:.3f}" for v in vals)+f"]{'OK' if match else 'x'} "
    print(line)
