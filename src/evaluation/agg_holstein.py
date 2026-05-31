# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
import csv, numpy as np
OUT="benchmarks/holstein"
rows=[d for d in csv.DictReader(open(f"{OUT}/fold_metrics_full.csv"))]
def num(x):
    try:
        v=float(x); return None if np.isnan(v) else v
    except: return None
def agg(mode,model,fam,w=False):
    vs=[];ws=[]
    for d in rows:
        if d["mode"]==mode and d["model"]==model and d["family"]==fam:
            p=num(d["pearson"])
            if p is None: continue
            vs.append(p); ws.append(float(d["n_test"]))
    if not vs: return None
    return float(np.average(vs,weights=ws)) if w else float(np.mean(vs))
def af(mode,model,fam,fold):
    for d in rows:
        if d["mode"]==mode and d["model"]==model and d["family"]==fam and d["fold"]==fold: return num(d["pearson"])
    return None
MS=["GBLUP_full","PCARidge_PCA80","Ridge_top50k","Ridge_top20k","FullSNP_MLP_top50k","SparseGate_MLP_top20k"]
def row(name,vals): print(f"{name:26}"+" ".join(f"{v:8.4f}" if v is not None else f"{'NA':>8}" for v in vals))
print("=== STRICT mean Pearson by relatedness level (rand->grp5->grp10->lco_wt) ===")
print(f"{'model':26}{'random':>8}{'group5':>8}{'group10':>8}{'lco_wt':>8}")
for m in MS: row(m,[agg('strict',m,'random'),agg('strict',m,'group5'),agg('strict',m,'group10'),agg('strict',m,'leave_cluster_out',w=True)])
T=["q99.5","q99","q97.5","q95","q90"]
print("\n=== STRICT relatedness-budget NATURAL (train shrinks 838->101) ===")
print(f"{'model':26}"+"".join(f"{t:>8}" for t in T))
for m in MS: row(m,[af('strict',m,'budget_natural',f'natural_{t}') for t in T])
print("\n=== STRICT relatedness-budget MATCHED (train fixed=101) ===")
for m in MS: row(m,[af('strict',m,'budget_matched_size',f'matched_size_{t}') for t in T])
print("\n=== CLEAN(strict) vs LEAK(historical) on random CV ===")
print(f"{'model':26}{'strict':>8}{'leak':>8}{'gap':>8}")
for m in MS:
    s=agg('strict',m,'random'); h=agg('historical',m,'random')
    row(m,[s,h,(h-s) if (s is not None and h is not None) else None])
print("\n=== Delta(FullSNP_MLP - GBLUP) across relatedness (the thesis) ===")
for fam,lab,w in [("random","random",False),("group5","group5",False),("group10","group10",False),("leave_cluster_out","lco_wt",True)]:
    mlp=agg('strict','FullSNP_MLP_top50k',fam,w); gb=agg('strict','GBLUP_full',fam,w)
    if mlp is not None and gb is not None: print(f"  {lab:10} MLP={mlp:.4f}  GBLUP={gb:.4f}  delta={mlp-gb:+.4f}")
# budget delta
for t in T:
    mlp=af('strict','FullSNP_MLP_top50k','budget_natural',f'natural_{t}'); gb=af('strict','GBLUP_full','budget_natural',f'natural_{t}')
    if mlp is not None and gb is not None: print(f"  budget_{t:7} MLP={mlp:.4f}  GBLUP={gb:.4f}  delta={mlp-gb:+.4f}")
print("\n=== n_test range for lco (why weighted) ===")
nt=[float(d['n_test']) for d in rows if d['family']=='leave_cluster_out' and d['model']=='GBLUP_full' and d['mode']=='strict']
print(f"  lco folds={len(nt)} n_test min={min(nt):.0f} max={max(nt):.0f}")
