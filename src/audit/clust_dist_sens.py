# CattleOmni reproducibility repo -- sanitised analysis script (paths repo-relative; no server paths/credentials/raw data).
#!/usr/bin/env python3
"""Clustering-distance SENSITIVITY. The relationship-aware folds use average-linkage clustering
on d = max(G) - G. Check that the within-cluster relatedness-controlled MLP-GBLUP contrast (the
load-bearing claim) is insensitive to the distance definition, by re-clustering the SAME GRM with
three distances at K=10 and recomputing the within-cluster sample-weighted contrast on the LOCKED
per-sample predictions (no model re-run). Distances:
  (1) current     : d = max(G) - G_ij
  (2) Euclidean   : d = sqrt(max(0, G_ii + G_jj - 2 G_ij))
  (3) normalized  : d = 1 - G_ij / sqrt(G_ii G_jj)   (correlation-like)
"""
import os, sys, csv, time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
sys.path.insert(0,"src")
import importlib.util
spec=importlib.util.spec_from_file_location("H","src/run_holstein_benchmark.py")
H=importlib.util.module_from_spec(spec); _a=sys.argv; sys.argv=["H"]; spec.loader.exec_module(H); sys.argv=_a
B="results"
OUT=f"{B}/clust_dist"; os.makedirs(OUT,exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)

M=np.load(H.GENO).astype(np.float64)
row_for_id,ids=H.load_alignment(); n=len(ids)
p=M.mean(0)/2.0; sd=np.sqrt(2*p*(1-p)); Z=M-2*p; denom=2.0*np.sum(p*(1-p)); G=(Z@Z.T)/denom
log(f"GRM {G.shape}")
d=np.sqrt(np.clip(np.diag(G),1e-9,None))

def dist_current(G): return G.max()-G
def dist_eucl(G):
    gg=np.diag(G); D=gg[:,None]+gg[None,:]-2*G; return np.sqrt(np.clip(D,0,None))
def dist_norm(G):
    r=G/np.outer(d,d); return 1.0-r

def cluster_K(D,K):
    Dsym=(D+D.T)/2; np.fill_diagonal(Dsym,0)
    Z=linkage(squareform(Dsym,checks=False),method="average")
    return fcluster(Z,t=K,criterion="maxclust")

# locked per-sample predictions (std space) keyed by sample_id
rows={}
for r in csv.DictReader(open(f"{B}/holstein_per_sample.csv")):
    rows.setdefault((r["model"],r["family"]),[]).append((r["fold"],int(float(r["sample_id"])),float(r["y_true_std"]),float(r["y_pred_std"])))
def pear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-9 and b.std()>1e-9 else float("nan")
def within_delta(cl_map, fam, minw=1):
    """sample-weighted within-(fold,cluster) residualised Pearson for MLP and GBLUP, return MLP-GBLUP."""
    res={}
    for mdl in ("FullSNP_MLP_top50k","GBLUP_full"):
        g={}
        for fold,sid,yts,yps in rows.get((mdl,fam),[]):
            c=cl_map.get(sid)
            if c is None: continue
            g.setdefault((fold,c),[]).append((yts,yps))
        RT=[];RP=[]
        for k,v in g.items():
            if len(v)<minw: continue
            yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]);RT+=list(yt-yt.mean());RP+=list(yp-yp.mean())
        res[mdl]=pear(RT,RP)
    return res["FullSNP_MLP_top50k"]-res["GBLUP_full"], res["FullSNP_MLP_top50k"], res["GBLUP_full"]

idx_of_sid={ids[i]:i for i in range(n)}  # sample_id -> row
results=[]
for dname,dfun in [("max(G)-G",dist_current),("euclidean",dist_eucl),("1-normG",dist_norm)]:
    lab=cluster_K(dfun(G),10)
    cl_map={ids[i]:int(lab[i]) for i in range(n)}
    row=[dname]
    for fam in ["random","group10","group5","leave_cluster_out"]:
        dlt,mlp,gb=within_delta(cl_map,fam)
        row.append(dlt)
    results.append(row)
    log(f"{dname}: within-Delta r,g10,g5,lco = "+", ".join(f"{x:+.3f}" for x in row[1:]))

with open(f"{OUT}/clust_dist_sens.csv","w") as f:
    f.write("distance,within_delta_random,within_delta_group10,within_delta_group5,within_delta_lco\n")
    for r in results: f.write(",".join([r[0]]+[f"{x:.4f}" for x in r[1:]])+"\n")
print("\n=== clustering-distance sensitivity: within-cluster Delta(MLP-GBLUP) at K=10 ===")
print(f"{'distance':12s} {'random':>8s} {'group10':>8s} {'group5':>8s} {'lco':>8s}")
for r in results: print(f"{r[0]:12s} "+" ".join(f"{x:+8.3f}" for x in r[1:]))
print(f"saved {OUT}/clust_dist_sens.csv")
