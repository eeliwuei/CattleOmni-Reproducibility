# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="benchmarks/holstein"
FIG="figures"; os.makedirs(FIG, exist_ok=True)
rows=[d for d in csv.DictReader(open(f"{OUT}/fold_metrics_full.csv"))]
def num(x):
    try:
        v=float(x); return None if (v!=v) else v
    except: return None
def agg(mode,model,fam,w=False):
    vs=[];ws=[]
    for d in rows:
        if d["mode"]==mode and d["model"]==model and d["family"]==fam:
            p=num(d["pearson"]);
            if p is None: continue
            vs.append(p); ws.append(float(d["n_test"]))
    if not vs: return None
    return float(np.average(vs,weights=ws)) if w else float(np.mean(vs))
MS=[("GBLUP_full","GBLUP"),("PCARidge_PCA80","PCA-Ridge"),("Ridge_top50k","Ridge50k"),
    ("FullSNP_MLP_top50k","FullSNP-MLP"),("SparseGate_MLP_top20k","SparseGate")]
cats=[("random","random"),("group5","group5"),("group10","group10"),("leave_cluster_out","leave-cluster-out")]
# Fig2: relatedness ladder
plt.figure(figsize=(7.5,5))
for mk,lab in MS:
    ys=[agg("strict",mk,f,w=(f=="leave_cluster_out")) for f,_ in cats]
    plt.plot(range(len(cats)),ys,marker="o",label=lab,lw=2)
plt.xticks(range(len(cats)),[c for _,c in cats]); plt.ylabel("Pearson r (strict, leakage-controlled)")
plt.title("Holstein milk-yield: accuracy vs validation stringency")
plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{FIG}/fig2_relatedness_ladder.png",dpi=140); plt.close()
# Fig3: delta(MLP-GBLUP)
plt.figure(figsize=(7.5,5)); dlt=[]
for f,_ in cats:
    w=(f=="leave_cluster_out"); mlp=agg("strict","FullSNP_MLP_top50k",f,w); gb=agg("strict","GBLUP_full",f,w)
    dlt.append((mlp-gb) if (mlp is not None and gb is not None) else 0.0)
plt.bar(range(len(cats)),dlt,color=["#4c72b0" if d>0 else "#c44e52" for d in dlt])
plt.axhline(0,color="k",lw=.8); plt.xticks(range(len(cats)),[c for _,c in cats])
plt.ylabel("Δ Pearson  (FullSNP-MLP − GBLUP)")
plt.title("Deep-model advantage collapses (reverses) under relatedness constraints")
plt.grid(alpha=.3,axis="y"); plt.tight_layout()
plt.savefig(f"{FIG}/fig3_delta_collapse.png",dpi=140); plt.close()
# Fig leak-gap
plt.figure(figsize=(7.5,5)); xi=np.arange(len(MS))
s=[agg("strict",mk,"random") or 0 for mk,_ in MS]; h=[agg("historical",mk,"random") or 0 for mk,_ in MS]
plt.bar(xi-0.2,s,0.4,label="strict (clean)",color="#55a868")
plt.bar(xi+0.2,h,0.4,label="leak (global supervised selection)",color="#c44e52")
plt.xticks(xi,[lab for _,lab in MS],rotation=18); plt.ylabel("Pearson r (random CV)")
plt.title("Supervised feature-selection leak inflates Pearson  (PCA-Ridge: none)")
plt.legend(fontsize=9); plt.grid(alpha=.3,axis="y"); plt.tight_layout()
plt.savefig(f"{FIG}/fig_leak_gap.png",dpi=140); plt.close()
print("FIGS DONE:", sorted(os.listdir(FIG)))
