# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Verify all 5 locked models reproduce tab:rankcal via the confirmed method:
10-cluster lco map (all regimes), pooled-residual within (fold,cluster), standardized, minw=1."""
import csv, numpy as np
B="benchmarks/holstein"
CMLCO="splits/holstein/leave_cluster_out/cluster_membership.csv"
FAMS=["random","group10","group5","leave_cluster_out"]
cl_lco={int(d["sample_id"]):int(d["cluster_id"]) for d in csv.DictReader(open(CMLCO))}
rows={}
for d in csv.DictReader(open(f"{B}/holstein_per_sample.csv")):
    rows.setdefault((d["model"],d["family"]),[]).append(
        (d["fold"],int(d["sample_id"]),float(d["y_true_std"]),float(d["y_pred_std"])))
def pear(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-9 and b.std()>1e-9 else float("nan")
def wc(model,family,minw=1):
    g={}
    for fold,sid,yts,yps in rows.get((model,family),[]):
        c=cl_lco.get(sid)
        if c is None: continue
        g.setdefault((fold,c),[]).append((yts,yps))
    RT=[];RP=[]
    for k,v in g.items():
        if len(v)<minw: continue
        yt=np.array([x[0] for x in v]);yp=np.array([x[1] for x in v]);RT+=list(yt-yt.mean());RP+=list(yp-yp.mean())
    return pear(RT,RP)
LOCK={"GBLUP_full":[0.356,0.255,0.260,0.265],"FullSNP_MLP_top50k":[0.391,0.287,0.290,0.271],
"Ridge_top50k":[0.351,0.279,0.270,0.305],"PCARidge_PCA80":[0.312,0.271,0.279,0.289],
"SparseGate_MLP_top20k":[0.278,0.167,0.201,0.179]}
print("%-24s %-30s %-30s %s" % ("model","computed(r/g10/g5/lco)","locked","maxdiff"))
allok=True
for m,tg in LOCK.items():
    v=[wc(m,f) for f in FAMS]; d=max(abs(a-b) for a,b in zip(v,tg)); ok=d<0.0025; allok=allok and ok
    print("%-24s [%s] [%s] %.4f %s" % (m,",".join("%.3f"%x for x in v),",".join("%.3f"%x for x in tg),d,"OK" if ok else "CHK"))
print("\nALL 5 MODELS REPRODUCE tab:rankcal" if allok else "\nsome rows differ >0.0025")
