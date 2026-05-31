# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Aggregate BayesB ladder -> (1) within-cluster row [verified 10-lco map method],
(2) random/group fold-mean + lco pooled [holstein_lock method], (3) Delta vs GBLUP,
(4) convergence diagnostics from varE traces. Reproduces GBLUP/MLP within-cluster as anchor."""
import csv, numpy as np, glob, os
B="benchmarks/holstein"
LAD="/tmp/bglr_ladder"
CMLCO="splits/holstein/leave_cluster_out/cluster_membership.csv"
SP="splits/holstein"
FAMS=["random","group10","group5","leave_cluster_out"]
cl_lco={int(d["sample_id"]):int(d["cluster_id"]) for d in csv.DictReader(open(CMLCO))}
def pear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-9 and b.std()>1e-9 else float("nan")

def load_persample(path):
    rows={}
    for d in csv.DictReader(open(path)):
        rows.setdefault((d["model"],d["family"]),[]).append(
            (d["fold"],int(float(d["sample_id"])),float(d["y_true_std"]),float(d["y_pred_std"]),
             float(d["y_true_raw"]),float(d["y_pred_raw"])))
    return rows

def within_cluster(rows,model,family,minw=1):   # VERIFIED method
    g={}
    for fold,sid,yts,yps,_,_ in rows.get((model,family),[]):
        c=cl_lco.get(sid)
        if c is None: continue
        g.setdefault((fold,c),[]).append((yts,yps))
    RT=[];RP=[]
    for k,v in g.items():
        if len(v)<minw: continue
        yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]);RT+=list(yt-yt.mean());RP+=list(yp-yp.mean())
    return pear(RT,RP)

def headline(rows,model,family):  # holstein_lock family_pearson: fold-mean(std) for random/group, pooled(raw) for lco
    byfold={}
    for fold,sid,yts,yps,ytr,ypr in rows.get((model,family),[]):
        byfold.setdefault(fold,[]).append((yts,yps,ytr,ypr))
    if family=="leave_cluster_out":
        pooled=[(x[2],x[3]) for f,v in byfold.items() if len(v)>=10 for x in v]
        return pear([a for a,b in pooled],[b for a,b in pooled]), len([f for f,v in byfold.items() if len(v)>=10])
    pf=[pear([x[0] for x in v],[x[1] for x in v]) for v in byfold.values()]
    pf=[p for p in pf if not np.isnan(p)]
    return (float(np.mean(pf)),float(np.std(pf)),len(pf))

# ---- load both ----
locked=load_persample(f"{B}/holstein_per_sample.csv")
bayes=load_persample(f"{LAD}/bayesb_per_sample.csv")
# auto-detect the new model name (BRR_full164k or BayesB_full164k) from the ladder file
M=sorted(set(m for (m,f) in bayes.keys()))[0]
print(f"new model in ladder = {M}\n")

print("="*70)
print("WITHIN-CLUSTER (verified 10-lco map; anchor: GBLUP/MLP must match locked)")
print("%-22s %-7s %-7s %-7s %-7s" % ("model","random","group10","group5","lco"))
for mdl,src in [("GBLUP_full",locked),("FullSNP_MLP_top50k",locked),(M,bayes)]:
    v=[within_cluster(src,mdl,f) for f in FAMS]
    print("%-22s %s" % (mdl,"  ".join("%.3f"%x for x in v)))
gb=[within_cluster(locked,"GBLUP_full",f) for f in FAMS]
bb=[within_cluster(bayes,M,f) for f in FAMS]
print("Delta(BayesB-GBLUP) within: "+"  ".join("%+.3f"%(b-g) for b,g in zip(bb,gb)))

print("="*70)
print("HEADLINE fold-mean (random/group) / pooled (lco) [holstein_lock method]")
for mdl,src in [("GBLUP_full",locked),("FullSNP_MLP_top50k",locked),(M,bayes)]:
    out=[]
    for f in FAMS:
        r=headline(src,mdl,f)
        out.append("%.4f"%r[0] + (f"±{r[1]:.3f}" if len(r)==3 and f!="leave_cluster_out" else ""))
    print("%-22s %s" % (mdl," | ".join(out)))

print("="*70)
print("CONVERGENCE (varE trace per fold; thin=5, burnIn=2000 -> burn cutoff sample 400)")
print("%-26s %7s %8s %8s %10s" % ("fold","nsamp","postMean","postSD","split1-2"))
for wd in sorted(glob.glob(f"{LAD}/*/")):
    ve=os.path.join(wd,"bglr_varE.dat")
    if not os.path.exists(ve): continue
    try: t=np.loadtxt(ve)
    except Exception: continue
    t=np.atleast_1d(t); burn=400
    post=t[burn:] if len(t)>burn else t
    h=len(post)//2
    s12=abs(post[:h].mean()-post[h:].mean())/(post.std()+1e-9) if h>0 else float("nan")
    print("%-26s %7d %8.3f %8.3f %10.3f" % (os.path.basename(wd.rstrip("/")),len(t),post.mean(),post.std(),s12))
