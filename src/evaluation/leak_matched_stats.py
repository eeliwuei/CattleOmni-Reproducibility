# CattleOmni reproducibility repo -- sanitised analysis script (paths repo-relative; no server paths/credentials/raw data).
#!/usr/bin/env python3
"""Post-process matched-leakage per-fold Pearson into the Table-4 replacement:
per-arm mean r + the three contrasts (selector / pure-leakage / total) with
paired t & Wilcoxon p (pairing unit = outer fold) and bootstrap 95% CI over folds.
Reads only the saved leak_matched_perfold.csv -- no model re-run, nothing fabricated."""
import csv, json, numpy as np
from itertools import groupby
B="results/leak_matched"
rows=list(csv.DictReader(open(f"{B}/leak_matched_perfold.csv")))
MODELS=["FullSNP_MLP_top50k","Ridge_top50k","Ridge_top20k","SparseGate_MLP_top20k"]
ARMS=["clean_var","clean_sup","leaky_sup"]

def vec(model,arm):
    d={int(r["fold"]):float(r["pearson"]) for r in rows if r["model"]==model and r["arm"]==arm}
    return np.array([d[f] for f in sorted(d)])

# minimal paired tests (no scipy dependency assumed)
def paired_t(a,b):
    d=a-b; n=len(d); m=d.mean(); s=d.std(ddof=1)
    if s==0: return float("inf"), 0.0
    t=m/(s/np.sqrt(n))
    # two-sided p via survival of |t| on t_{n-1} using a normal approx + small-n note
    # use exact-ish via math.erfc for large n; for n=10 report t and a normal-approx p
    from math import erfc, sqrt
    p=erfc(abs(t)/sqrt(2))   # normal approx (n=10 -> slightly anti-conservative; report t too)
    return float(t), float(p)
def wilcoxon(a,b):
    d=a-b; d=d[d!=0]; n=len(d)
    if n==0: return float("nan")
    r=np.argsort(np.argsort(np.abs(d)))+1
    W=min(r[d>0].sum(), r[d<0].sum())
    mu=n*(n+1)/4; sd=np.sqrt(n*(n+1)*(2*n+1)/24)
    from math import erfc, sqrt
    z=(W-mu)/sd if sd>0 else 0.0
    return float(erfc(abs(z)/sqrt(2)))
def boot_ci(a,b,B_=10000,seed=20260529):
    rng=np.random.default_rng(seed); d=a-b; n=len(d)
    idx=rng.integers(0,n,size=(B_,n))
    bs=d[idx].mean(1)
    return float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

out={}
hdr=f"{'model':22s} {'clean_var':>9s} {'clean_sup':>9s} {'leaky_sup':>9s} | {'sel(C-A)':>9s} {'leak(C-B)':>10s} {'total(C-A)':>11s}  {'p_leak':>9s} {'CI_leak':>20s}"
print(hdr); print("-"*len(hdr))
for m in MODELS:
    A=vec(m,"clean_var"); Bv=vec(m,"clean_sup"); C=vec(m,"leaky_sup")
    sel=Bv.mean()-A.mean()          # supervised selector effect (both clean)
    leak=C.mean()-Bv.mean()         # PURE cross-split leakage (selector fixed)
    total=C.mean()-A.mean()         # historical positive-control total inflation
    t_leak,p_leak_t=paired_t(C,Bv); p_leak_w=wilcoxon(C,Bv)
    lo,hi=boot_ci(C,Bv)
    t_sel,p_sel_t=paired_t(Bv,A); lo_s,hi_s=boot_ci(Bv,A)
    t_tot,p_tot_t=paired_t(C,A); lo_t,hi_t=boot_ci(C,A)
    out[m]=dict(clean_var=A.mean(),clean_sup=Bv.mean(),leaky_sup=C.mean(),
        delta_supervised_selector=sel, delta_pure_leakage=leak, delta_total_positive_control=total,
        p_leak_t=p_leak_t,p_leak_wilcoxon=p_leak_w,ci_leak=[lo,hi],
        p_sel_t=p_sel_t,ci_sel=[lo_s,hi_s], p_total_t=p_tot_t,ci_total=[lo_t,hi_t], nfold=len(A))
    print(f"{m:22s} {A.mean():9.4f} {Bv.mean():9.4f} {C.mean():9.4f} | {sel:+9.4f} {leak:+10.4f} {total:+11.4f}  {p_leak_t:9.2e} [{lo:+.3f},{hi:+.3f}]")
json.dump(out,open(f"{B}/leak_matched_stats.json","w"),indent=2)
print(f"\nnfolds={out[MODELS[0]]['nfold']}  (pairing unit = outer fold)")
print(f"saved {B}/leak_matched_stats.json")
print("\nINTERPRETATION:")
print("  delta_pure_leakage (leaky_sup - clean_sup): selector held fixed (correlation), only cross-split vs fold-internal differs -> ISOLATED leakage")
print("  delta_supervised_selector (clean_sup - clean_var): supervised-vs-variance selector effect, both fold-internal/clean")
print("  delta_total_positive_control (leaky_sup - clean_var): historical clean-vs-leaky total")
