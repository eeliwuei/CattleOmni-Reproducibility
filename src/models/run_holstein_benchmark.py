# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""
Holstein leakage-aware genomic prediction benchmark (paper CORE experiment).
- STRICT pipeline = OFFICIAL: every preprocessing step (top-k SNP selection,
  feature standardization, PCA, target standardization, lambda) fit on TRAIN FOLD ONLY.
- HISTORICAL pipeline = DIAGNOSTIC ONLY: global preprocessing (top-k by corr on ALL y,
  PCA on ALL, target std on ALL) to reproduce the apparent ~0.46 and quantify inflation.
Runs on H100. Deep models on GPU as polite tenant (<=5GB). Pure numpy+torch (no pandas/sklearn).
seed=20260529. All official results reproducible from saved splits/seed/configs/predictions.
"""
import os, sys, json, time, glob, argparse, csv, traceback, hashlib
import numpy as np

SEED = 20260529
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

QC    = "data/processed/holstein/parse_qc"
ROOT  = "."
GENO  = f"{QC}/holstein_genotype_matrix_qc.npy"
ALIGN = f"{QC}/holstein_sample_alignment.csv"
PHENO = f"{ROOT}/data/holstein_phenotype_table.csv"
SP    = f"{ROOT}/splits/holstein"
OUT   = f"{ROOT}/benchmarks/holstein"
TRAIT = os.environ.get("HTRAIT", "milk_le_sum_305")  # multi-trait via env (milk default; fat_le_sum_305, scs_le_ave305)
LAM_GRID = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
TOP50K, TOP20K, NPCA = 50000, 20000, 80

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------------- IO ----------------
def read_ids(path):
    out = []
    with open(path) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if row and row[0] != "": out.append(int(float(row[0])))
    return out

def load_alignment():
    rows = []
    with open(ALIGN) as f:
        for d in csv.DictReader(f):
            rows.append((int(d["sample_id"]), int(d["order"])))
    rows.sort(key=lambda x: x[1])                 # order ascending -> genotype row = order-1
    row_for_id = {sid: i for i, (sid, _) in enumerate(rows)}
    ids_by_row = [sid for sid, _ in rows]
    return row_for_id, ids_by_row

def load_pheno(row_for_id, n):
    y = np.full(n, np.nan)
    with open(PHENO) as f:
        for d in csv.DictReader(f):
            sid = int(d["ID"])
            if sid in row_for_id:
                y[row_for_id[sid]] = float(d[TRAIT])
    return y

# ---------------- metrics ----------------
def metrics(yt, yp):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
    pear = float(np.corrcoef(yt, yp)[0, 1]) if (np.std(yp) > 1e-12 and np.std(yt) > 1e-12) else float("nan")
    ss_res = float(np.sum((yt - yp) ** 2)); ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    b = float(np.cov(yt, yp)[0, 1] / np.var(yp)) if np.std(yp) > 1e-12 else float("nan")
    return dict(pearson=pear, r2=float(r2), rmse=rmse, cal_slope=b)

# ---------------- dual ridge / kernel ridge ----------------
def dual_solve(K_tr, y_tr, K_te_tr, lam):
    n = K_tr.shape[0]
    alpha = np.linalg.solve(K_tr + lam * np.eye(n), y_tr)
    return K_te_tr @ alpha

def pick_lam(K_tr, y_tr, grid=LAM_GRID, k=3):
    n = len(y_tr); idx = np.arange(n); RNG.shuffle(idx)
    folds = np.array_split(idx, k); best = (1.0, -2.0)
    for lam in grid:
        ps = []
        for j in range(k):
            te = folds[j]; tr = np.concatenate([folds[m] for m in range(k) if m != j])
            pred = dual_solve(K_tr[np.ix_(tr, tr)], y_tr[tr], K_tr[np.ix_(te, tr)], lam)
            mp = metrics(y_tr[te], pred)["pearson"]
            if not np.isnan(mp): ps.append(mp)
        s = np.mean(ps) if ps else -2.0
        if s > best[1]: best = (lam, s)
    return best[0]

# ---------------- preprocessing helpers ----------------
def topk_corr_idx(Xtr, ytr, k):
    yc = ytr - ytr.mean(); Xc = Xtr - Xtr.mean(0)
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(0)) * (np.sqrt((yc ** 2).sum()) + 1e-12) + 1e-12
    corr = np.abs(num / den)
    return np.argpartition(-corr, k)[:k]

def topk_var_idx(Xtr, k):
    return np.argpartition(-Xtr.var(0), k)[:k]  # unsupervised, clean; matches old v5 selection

def std_fit_apply(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd

def pca_gram(Xtr, Xte, ncomp):
    mu = Xtr.mean(0); Xtr_c = Xtr - mu; Xte_c = Xte - mu
    Kx = Xtr_c @ Xtr_c.T
    w, V = np.linalg.eigh(Kx); w = w[::-1]; V = V[:, ::-1]
    w = np.clip(w[:ncomp], 1e-9, None); V = V[:, :ncomp]
    Ztr = V * np.sqrt(w)
    Wfeat = (Xtr_c.T @ V) / np.sqrt(w)
    Zte = Xte_c @ Wfeat
    return Ztr, Zte

def ridge_primal(Ztr, ytr, Zte, grid=LAM_GRID):
    # tiny feature dim -> primal with inner CV
    n, d = Ztr.shape; idx = np.arange(n); RNG.shuffle(idx); folds = np.array_split(idx, 3)
    best = (1.0, -2.0)
    for lam in grid:
        ps = []
        for j in range(3):
            te = folds[j]; tr = np.concatenate([folds[m] for m in range(3) if m != j])
            A = Ztr[tr].T @ Ztr[tr] + lam * np.eye(d)
            wts = np.linalg.solve(A, Ztr[tr].T @ ytr[tr])
            mp = metrics(ytr[te], Ztr[te] @ wts)["pearson"]
            if not np.isnan(mp): ps.append(mp)
        s = np.mean(ps) if ps else -2.0
        if s > best[1]: best = (lam, s)
    lam = best[0]
    A = Ztr.T @ Ztr + lam * np.eye(d)
    wts = np.linalg.solve(A, Ztr.T @ ytr)
    return Zte @ wts

# ---------------- MLP (torch, GPU polite tenant) ----------------
def mlp_fit_predict(Xtr, ytr, Xte, hidden=256, dropout=0.35, epochs=30, lr=8e-4, wd=1e-3,
                    gate=False, l1=0.0):
    # Architecture & training matched to old v5 FullSnpMLP (LayerNorm, [256,128], fixed epochs,
    # AdamW) -> robust on small budget folds (no BatchNorm), no early stopping (never peeks test).
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if dev == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.06, 0)  # ~5GB cap, polite tenant
    except Exception:
        pass
    torch.manual_seed(SEED)
    d = Xtr.shape[1]
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev).view(-1, 1)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=dev)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Parameter(torch.zeros(d)) if gate else None
            self.net = nn.Sequential(
                nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        def forward(self, x):
            if self.gate is not None: x = x * torch.sigmoid(self.gate)
            return self.net(x)
    m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); lossf = nn.MSELoss()
    n = Xt.shape[0]; bs = 256
    for ep in range(epochs):
        m.train(); order = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            b = order[s:s + bs]
            if b.numel() < 2: continue
            opt.zero_grad(); out = m(Xt[b]); l = lossf(out, yt[b])
            if gate and l1 > 0: l = l + l1 * torch.sigmoid(m.gate).abs().mean()
            l.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xe).cpu().numpy().ravel()
    del Xt, yt, Xe, m
    if dev == "cuda":
        import torch as _t; _t.cuda.empty_cache()
    return pred

# ---------------- model runners (return test predictions) ----------------
def run_model(name, M, G, tr, te, ytr, yte, mode, glob_cache):
    if name == "GBLUP_full":
        K = G[np.ix_(tr, tr)]; lam = pick_lam(K, ytr); return dual_solve(K, ytr, G[np.ix_(te, tr)], lam)
    if name in ("Ridge_top50k", "Ridge_top20k", "FullSNP_MLP_top50k", "SparseGate_MLP_top20k"):
        k = TOP50K if "50k" in name else TOP20K
        if mode == "historical":
            idx = glob_cache["idx50k"] if k == TOP50K else glob_cache["idx20k"]
            mu, sd = glob_cache["mu"][idx], glob_cache["sd"][idx]
            Xtr = (M[np.ix_(tr, idx)] - mu) / sd; Xte = (M[np.ix_(te, idx)] - mu) / sd
        else:
            idx = topk_var_idx(M[tr], k)  # STRICT: train-fold VARIANCE (clean, matches old v5)
            Xtr, Xte = std_fit_apply(M[np.ix_(tr, idx)], M[np.ix_(te, idx)])
        if name.startswith("Ridge"):
            K = Xtr @ Xtr.T; lam = pick_lam(K, ytr); return dual_solve(K, ytr, Xte @ Xtr.T, lam)
        if name == "FullSNP_MLP_top50k":
            return mlp_fit_predict(Xtr, ytr, Xte, gate=False)
        return mlp_fit_predict(Xtr, ytr, Xte, gate=True, l1=1e-3)
    if name == "PCARidge_PCA80":
        if mode == "historical":
            Wfeat = glob_cache["pca_W"]; mu = glob_cache["pca_mu"]
            Ztr = (M[tr] - mu) @ Wfeat; Zte = (M[te] - mu) @ Wfeat
        else:
            Ztr, Zte = pca_gram(M[tr], M[te], NPCA)
        return ridge_primal(Ztr, ytr, Zte)
    raise ValueError(name)

# ---------------- GRM ----------------
def vanraden_grm(M):
    p = M.mean(0, dtype=np.float64) / 2.0
    Z = M - (2.0 * p).astype(np.float32)          # float32 to limit RAM (avoid OOM-kill on shared host)
    denom = float(2.0 * np.sum(p * (1 - p)))
    return ((Z @ Z.T) / denom).astype(np.float32)

# ---------------- split collection ----------------
def collect_splits(families):
    out = []
    def add(fam, label, trc, tec):
        if os.path.exists(trc) and os.path.exists(tec): out.append((fam, label, trc, tec))
    if "random" in families:
        for t in sorted(glob.glob(f"{SP}/random/*_train.csv")):
            lb = os.path.basename(t)[:-10]; add("random", lb, t, t.replace("_train.csv", "_test.csv"))
    if "group" in families:
        for g, fam in (("group_5fold", "group5"), ("group_10fold", "group10")):
            for t in sorted(glob.glob(f"{SP}/relationship_group/{g}/*_train.csv")):
                lb = os.path.basename(t)[:-10]; add(fam, lb, t, t.replace("_train.csv", "_test.csv"))
    if "lco" in families:
        for t in sorted(glob.glob(f"{SP}/leave_cluster_out/*_train.csv")):
            lb = os.path.basename(t)[:-10]; add("leave_cluster_out", lb, t, t.replace("_train.csv", "_test.csv"))
    if "budget" in families:
        for kind in ("natural", "matched_size"):
            for tau in ("q99.5", "q99", "q97.5", "q95", "q90"):
                t = f"{SP}/relatedness_budget/{kind}/{tau}/train.csv"
                add(f"budget_{kind}", f"{kind}_{tau}", t, t.replace("train.csv", "test.csv"))
    return out

# ---------------- driver ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["strict", "historical", "both"])
    ap.add_argument("--families", default="random,group,lco,budget")
    ap.add_argument("--models", default="GBLUP_full,Ridge_top50k,Ridge_top20k,PCARidge_PCA80,FullSNP_MLP_top50k,SparseGate_MLP_top20k")
    ap.add_argument("--max-folds", type=int, default=0, help="limit folds per family (0=all) for --validate")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    families = args.families.split(","); models = args.models.split(",")
    modes = ["strict", "historical"] if args.mode == "both" else [args.mode]
    os.makedirs(f"{OUT}/strict", exist_ok=True); os.makedirs(f"{OUT}/historical", exist_ok=True)

    log(f"load genotype {GENO}"); M = np.load(GENO).astype(np.float32)
    assert M.shape == (1092, 164312), f"bad shape {M.shape}"
    row_for_id, ids_by_row = load_alignment(); n = len(ids_by_row)
    y = load_pheno(row_for_id, n)
    assert np.isfinite(y).all(), f"non-finite phenotype: {np.sum(~np.isfinite(y))}"
    log(f"M{M.shape} y finite={np.isfinite(y).sum()} mean={y.mean():.2f} std={y.std():.2f}")

    log("compute VanRaden GRM ..."); G = vanraden_grm(M)
    log(f"GRM diag mean={np.diag(G).mean():.3f} offdiag q99={np.quantile(G[np.triu_indices(n,1)],0.99):.4f}")

    glob_cache = {}
    if "historical" in modes:
        log("precompute HISTORICAL global leak quantities (corr/PCA/std on ALL data)")
        ystd = (y - y.mean()) / y.std()
        glob_cache["idx50k"] = topk_corr_idx(M, ystd, TOP50K)
        glob_cache["idx20k"] = topk_corr_idx(M, ystd, TOP20K)
        glob_cache["mu"] = M.mean(0); glob_cache["sd"] = M.std(0) + 1e-8
        Ztr, _ = pca_gram(M, M[:1], NPCA)  # need Wfeat; recompute explicitly
        mu = M.mean(0); Xc = M - mu; Kx = Xc @ Xc.T
        w, V = np.linalg.eigh(Kx); w = w[::-1]; V = V[:, ::-1]
        w = np.clip(w[:NPCA], 1e-9, None); V = V[:, :NPCA]
        glob_cache["pca_W"] = (Xc.T @ V) / np.sqrt(w); glob_cache["pca_mu"] = mu

    splits_all = collect_splits(families)
    if args.max_folds > 0:
        seen = {}; lim = []
        for s in splits_all:
            seen[s[0]] = seen.get(s[0], 0)
            if seen[s[0]] < args.max_folds: lim.append(s); seen[s[0]] += 1
        splits_all = lim
    log(f"{len(splits_all)} folds x {len(models)} models x modes={modes}")

    met_path = f"{OUT}/fold_metrics_{args.tag}.csv"; pred_path = f"{OUT}/fold_predictions_{args.tag}.csv"
    prog_path = f"{OUT}/progress_{args.tag}.json"
    MHEAD = "mode,model,family,fold,n_train,n_test,pearson,r2,rmse,cal_slope,lam_or_na,seconds"
    PHEAD = "mode,model,family,fold,sample_id,y_true_std,y_pred"
    # RESUME: sanitize any existing partial files (drop corrupt/partial lines), keep done units
    done_keys = set(); good_m = []
    if os.path.exists(met_path):
        with open(met_path, errors="ignore") as f:
            for ln in f:
                p = ln.replace("\x00", "").rstrip("\n").split(",")
                if len(p) == 12 and p[0] in ("strict", "historical"):
                    good_m.append(",".join(p)); done_keys.add((p[0], p[1], p[2], p[3]))
    with open(met_path, "w") as f:
        f.write(MHEAD + "\n");
        if good_m: f.write("\n".join(good_m) + "\n")
    good_p = []
    if os.path.exists(pred_path):
        with open(pred_path, errors="ignore") as f:
            for ln in f:
                p = ln.replace("\x00", "").rstrip("\n").split(",")
                if len(p) == 7 and (p[0], p[1], p[2], p[3]) in done_keys:
                    good_p.append(",".join(p))
    with open(pred_path, "w") as f:
        f.write(PHEAD + "\n")
        if good_p: f.write("\n".join(good_p) + "\n")
    mf = open(met_path, "a"); pf = open(pred_path, "a")
    total = len(modes) * len(splits_all) * len(models); done = 0; t0 = time.time()
    if done_keys: log(f"RESUME: {len(done_keys)} units already done, skipping them")
    for mode in modes:
        for (fam, fold, trc, tec) in splits_all:
            if mode == "historical" and fam != "random":
                continue
            tr = np.array([row_for_id[i] for i in read_ids(trc)])
            te = np.array([row_for_id[i] for i in read_ids(tec)])
            mu_y, sd_y = (y[tr].mean(), y[tr].std()) if mode == "strict" else (y.mean(), y.std())
            ytr = (y[tr] - mu_y) / sd_y; yte = (y[te] - mu_y) / sd_y
            te_sids = [ids_by_row[i] for i in te]
            for name in models:
                if mode == "historical" and name == "GBLUP_full":
                    done += 1; continue  # GBLUP has no feature-selection leak; skip in historical
                if (mode, name, fam, fold) in done_keys:
                    done += 1; continue  # RESUME: already computed in a prior attempt
                ts = time.time()
                try:
                    pred = run_model(name, M, G, tr, te, ytr, yte, mode, glob_cache)
                    mt = metrics(yte, pred)
                    mf.write(f"{mode},{name},{fam},{fold},{len(tr)},{len(te)},{mt['pearson']:.5f},{mt['r2']:.5f},{mt['rmse']:.5f},{mt['cal_slope']:.5f},na,{time.time()-ts:.1f}\n"); mf.flush()
                    for sid, yp in zip(te_sids, pred):
                        pf.write(f"{mode},{name},{fam},{fold},{sid},{0:.0f},{yp:.6f}\n")
                    pf.flush()
                    log(f"[{mode}/{name}/{fam}/{fold}] r={mt['pearson']:.4f} R2={mt['r2']:.3f} ({time.time()-ts:.1f}s)")
                except Exception as e:
                    mf.write(f"{mode},{name},{fam},{fold},{len(tr)},{len(te)},ERR,ERR,ERR,ERR,na,{time.time()-ts:.1f}\n"); mf.flush()
                    log(f"[{mode}/{name}/{fam}/{fold}] ERROR {e}\n{traceback.format_exc()}")
                done += 1
                json.dump({"done": done, "total": total, "elapsed_s": round(time.time()-t0,1),
                           "last": f"{mode}/{name}/{fam}/{fold}"}, open(prog_path, "w"))
    mf.close(); pf.close()
    open(f"{OUT}/ALL_DONE_{args.tag}", "w").write(f"{done}/{total}\n")
    log(f"ALL_DONE {done}/{total} in {time.time()-t0:.0f}s -> {met_path}")

if __name__ == "__main__":
    main()
