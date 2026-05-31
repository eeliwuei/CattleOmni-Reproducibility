# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A7 Trait Potential = evidence-aware SELECTIVE prediction (risk-coverage), NOT a GEBV producer.
Evidence signal per test animal = max GRM to its training fold (close relative in reference => high evidence).
Selective prediction: abstain on lowest-evidence animals; does accuracy on the RETAINED set improve, and does
evidence-ranked abstention beat RANDOM abstention? Skill = R2-vs-mean (train-mean baseline = 0 in std space).
Pure numpy. Uses existing per-sample predictions (no model re-fit). Random-CV family (full N)."""
import csv, glob, time
import numpy as np
SP = "splits/holstein"
GRM = "data/processed/holstein/relationship_splits/holstein_grm.npy"
ALIGN = "data/processed/holstein/parse_qc/holstein_sample_alignment.csv"
PS = "benchmarks/holstein/holstein_per_sample.csv"
OUT = "benchmarks/holstein"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
G = np.load(GRM); nG = G.shape[0]
sid2idx = {int(d["sample_id"]): int(d["order"]) - 1 for d in csv.DictReader(open(ALIGN))}  # order is 1-based
all_idx = np.array(sorted(sid2idx.values()))
log(f"GRM {G.shape}; aligned ids {len(sid2idx)}")
# load per-sample predictions (random family only)
rows = {}
for d in csv.DictReader(open(PS)):
    if d["family"] != "random": continue
    m = d["model"]
    rows.setdefault(m, []).append((d["fold"], int(d["sample_id"]), float(d["y_true_std"]), float(d["y_pred_std"])))
def r2_vs_mean(yt, yp): return 1.0 - np.sum((yt - yp) ** 2) / np.sum(yt ** 2)   # train-mean baseline = 0 (std)
def pear(yt, yp):
    a, b = yt - yt.mean(), yp - yp.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if d == 0 else float((a * b).sum() / d)
COVS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
out = open(f"{OUT}/A7_RISK_COVERAGE.csv", "w")
out.write("model,coverage,n,evidence_min,pearson,r2_vs_mean,pearson_random,r2_random,evidence_thresh\n")
for m, rr in rows.items():
    if m not in ("GBLUP_full", "FullSNP_MLP_top50k"): continue
    # evidence per (sample,fold): max GRM to train (=all aligned minus this fold's test set)
    byfold = {}
    for fold, sid, yt, yp in rr: byfold.setdefault(fold, []).append((sid, yt, yp))
    ev, YT, YP = [], [], []
    for fold, items in byfold.items():
        test_idx = np.array([sid2idx[s] for s, _, _ in items])
        train_mask = np.ones(nG, bool); train_mask[test_idx] = False
        sub = G[np.ix_(test_idx, np.where(train_mask)[0])]   # test x train GRM
        emax = sub.max(axis=1)
        for k, (s, yt, yp) in enumerate(items):
            ev.append(emax[k]); YT.append(yt); YP.append(yp)
    ev = np.array(ev); YT = np.array(YT); YP = np.array(YP)
    order = np.argsort(-ev)                       # high evidence first
    rng = np.random.RandomState(20260530)
    log(f"{m}: N={len(ev)} units; full Pearson={pear(YT,YP):.4f} R2vm={r2_vs_mean(YT,YP):.4f} ; evidence[min={ev.min():.3f} med={np.median(ev):.3f} max={ev.max():.3f}]")
    for c in COVS:
        k = int(round(len(ev) * c)); k = max(k, 20)
        keep = order[:k]
        # random-abstention baseline (mean of 20 random subsets of same size)
        pr, rr2 = [], []
        for _ in range(20):
            idx = rng.choice(len(ev), k, replace=False); pr.append(pear(YT[idx], YP[idx])); rr2.append(r2_vs_mean(YT[idx], YP[idx]))
        out.write(f"{m},{c:.2f},{k},{ev[keep].min():.4f},{pear(YT[keep],YP[keep]):.4f},{r2_vs_mean(YT[keep],YP[keep]):.4f},{np.mean(pr):.4f},{np.mean(rr2):.4f},{ev[keep].min():.4f}\n")
out.close()
log("A7 DONE -> A7_RISK_COVERAGE.csv")
# headline print
for line in open(f"{OUT}/A7_RISK_COVERAGE.csv"):
    if ",1.00," in line or ",0.50," in line or ",0.30," in line: print("  " + line.strip())
