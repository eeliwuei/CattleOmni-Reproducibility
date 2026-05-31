# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
"""Phase 1a: GREML/REML heritability for Holstein traits.
Single-component REML via GRM eigendecomposition (exact 1D profile over h^2).
Alignment = locked benchmark: phenotype sorted by 'order' column -> matches GRM row order.
All numbers from disk; no fabrication."""
import numpy as np, csv, time
GRM="data/processed/holstein/relationship_splits/holstein_grm.npy"
PHE="data/processed/holstein/parse_qc/holstein_phenotype_table.csv"
TRAITS=["milk_le_sum_305","fat_le_sum_305","scs_le_ave305"]
OUT="benchmarks/holstein/h2_reml.csv"

A=np.load(GRM).astype(np.float64); A=(A+A.T)/2.0
n=A.shape[0]
# read CSV with csv module (pandas binary-incompatible on this box), sort by 'order'
_rows=list(csv.DictReader(open(PHE)))
_rows.sort(key=lambda r: float(r["order"]))
assert len(_rows)==n, f"pheno {len(_rows)} != GRM {n}"
class _P:
    def __init__(s,rows): s.rows=rows
    def col(s,name): return np.array([float(r[name]) for r in s.rows], dtype=np.float64)
phe=_P(_rows)
print(f"GRM {A.shape}, pheno {len(_rows)} (sorted by order)")

# eigendecomposition once
t0=time.time()
d,U=np.linalg.eigh(A)
print(f"eigh done {time.time()-t0:.1f}s; eigval range [{d.min():.4f},{d.max():.4f}]")
ones=np.ones(n)
ones_rot=U.T@ones

def reml_negll(h2, y_rot):
    D=h2*d+(1.0-h2)
    if np.any(D<=1e-9): return 1e18
    # GLS intercept with rotated design (intercept only)
    a=np.sum(ones_rot*ones_rot/D); b=np.sum(ones_rot*y_rot/D)
    beta=b/a
    r=y_rot-beta*ones_rot
    s2p=np.sum(r*r/D)/(n-1)        # REML variance estimate (1 fixed effect)
    if s2p<=0: return 1e18
    # REML log-likelihood (up to const): -0.5[ sum log D + (n-1) log s2p + log(sum ones^2/D) ]
    ll=-0.5*(np.sum(np.log(D))+(n-1)*np.log(s2p)+np.log(a))
    return -ll

def fit_trait(name):
    y=phe.col(name)
    m=np.isfinite(y);
    if not m.all():
        # keep finite only -> would break GRM alignment; instead require all finite
        raise SystemExit(f"{name}: {(~m).sum()} non-finite")
    y=(y-y.mean())/(y.std()+1e-12)
    y_rot=U.T@y
    # 1D grid then refine
    grid=np.linspace(0.001,0.999,400)
    nlls=np.array([reml_negll(h,y_rot) for h in grid])
    i=int(np.argmin(nlls)); h0=grid[i]
    # golden refine around h0
    lo,hi=max(0.001,h0-0.01),min(0.999,h0+0.01)
    for _ in range(60):
        m1=lo+(hi-lo)*0.382; m2=lo+(hi-lo)*0.618
        if reml_negll(m1,y_rot)<reml_negll(m2,y_rot): hi=m2
        else: lo=m1
    h2=(lo+hi)/2
    # SE from curvature of profile (numerical 2nd deriv of negLL = observed info)
    eps=1e-3
    f0=reml_negll(h2,y_rot); fp=reml_negll(min(0.999,h2+eps),y_rot); fm=reml_negll(max(0.001,h2-eps),y_rot)
    info=(fp-2*f0+fm)/(eps*eps)   # d2(negLL)/dh2^2
    se=float(np.sqrt(1.0/info)) if info>0 else float("nan")
    return h2, se

rows=[]
for t in TRAITS:
    h2,se=fit_trait(t)
    print(f"  {t:20s} h2={h2:.3f}  SE={se:.3f}")
    rows.append((t,round(h2,4),round(se,4)))
with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["trait","h2_reml","se"]); w.writerows(rows)
print("saved",OUT)
