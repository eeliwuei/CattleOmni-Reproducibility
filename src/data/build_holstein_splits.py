# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""
GRM / Relatedness-budget split constructor for Holstein (1092 x 164,312).

Authoritative, reproducible, auditable splits shared by all downstream models.
- seed = 20260529 (fixed, no unseeded randomness anywhere)
- GRM = VanRaden method 1
- Output root: splits/holstein/

Read-only on production data. Writes only under splits/.
"""
import os, json, hashlib, sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

SEED = 20260529
QC_DIR = "./yanbian_cattle_breeding_ai_results/cattleomni_v4_full_public_supervised/03_holstein_milk/qc"
OUT_ROOT = "splits/holstein"
TRAIT = "milk_le_sum_305"

def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_ids(path, ids):
    """Write a one-column CSV of sample_id (sorted ascending for determinism)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"sample_id": sorted(int(x) for x in ids)}).to_csv(path, index=False)

def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ---------------------------------------------------------------------------
# 1. Load + verify genotype matrix
# ---------------------------------------------------------------------------
print("=== STEP 1: load genotype ===", flush=True)
M = np.load(os.path.join(QC_DIR, "holstein_genotype_matrix_qc.npy"))  # (1092, 164312) float32 dosage
n_ind, n_snp = M.shape
assert M.shape == (1092, 164312), f"DIMENSION MISMATCH: got {M.shape}, expected (1092, 164312)"
print(f"genotype shape verified: {M.shape}, dtype={M.dtype}", flush=True)

align = pd.read_csv(os.path.join(QC_DIR, "holstein_sample_alignment.csv"))
align = align.sort_values("genotype_column_index").reset_index(drop=True)
assert list(align.genotype_column_index) == list(range(n_ind)), "alignment index not 0..n-1 contiguous"
sample_ids = align.sample_id.to_numpy()  # row i of M corresponds to sample_ids[i]
assert len(sample_ids) == n_ind

# Encoding / missingness / MAF audit (treat negative dosage / sentinel as missing if any)
uniq_sample = np.unique(M[:200, :2000])
vals_present = sorted(float(v) for v in uniq_sample if -1 <= v <= 2)
# A common missing sentinel in such matrices is a negative value or NaN; check both.
nan_count = int(np.isnan(M).sum())
# Detect sentinel: any value not in {0,1,2} (after rounding) and not NaN
flat_unique = np.unique(M)
nonstd = [float(v) for v in flat_unique if not np.isnan(v) and v not in (0.0, 1.0, 2.0)]
print(f"distinct genotype values (excl NaN): {[float(v) for v in flat_unique if not np.isnan(v)][:12]}", flush=True)
print(f"NaN count: {nan_count}; non-standard (not 0/1/2) values: {nonstd[:10]}", flush=True)

# Build missing mask: NaN or negative sentinel
missing_mask = np.isnan(M) | (M < 0)
n_missing = int(missing_mask.sum())
overall_missing_rate = n_missing / (n_ind * n_snp)
print(f"overall missing rate: {overall_missing_rate:.6g} ({n_missing} cells)", flush=True)

# Per-SNP allele frequency p = mean(dosage)/2 over non-missing
Mw = M.astype(np.float64).copy()
Mw[missing_mask] = np.nan
col_mean = np.nanmean(Mw, axis=0)          # mean dosage per SNP (non-missing)
p = col_mean / 2.0                          # allele freq of counted allele
maf = np.minimum(p, 1.0 - p)
print(f"MAF: min={np.nanmin(maf):.5f} median={np.nanmedian(maf):.5f} max={np.nanmax(maf):.5f}", flush=True)
print(f"  fraction SNPs MAF<0.01: {(maf<0.01).mean():.4f}; MAF<0.05: {(maf<0.05).mean():.4f}", flush=True)

# Cross-check MAF against snp_metadata.csv precomputed maf (sanity)
snp_meta = pd.read_csv(os.path.join(QC_DIR, "holstein_snp_metadata.csv"))
maf_corr = np.corrcoef(snp_meta.maf.to_numpy()[:len(maf)], maf)[0, 1]
print(f"MAF correlation vs snp_metadata precomputed: {maf_corr:.6f}", flush=True)

# ---------------------------------------------------------------------------
# 2. Load phenotype, align
# ---------------------------------------------------------------------------
print("\n=== STEP 2: phenotype ===", flush=True)
phe = pd.read_csv(os.path.join(QC_DIR, "holstein_phenotype_table.csv"))
assert TRAIT in phe.columns, f"{TRAIT} not in phenotype table"
phe_idcol = "sample_id" if "sample_id" in phe.columns else "ID"
phe = phe[[phe_idcol, TRAIT]].rename(columns={phe_idcol: "sample_id"})
phe["sample_id"] = phe["sample_id"].astype(int)
phe_map = dict(zip(phe.sample_id, phe[TRAIT]))
# strict alignment: every genotype sample_id must have a phenotype
missing_phe = [int(s) for s in sample_ids if int(s) not in phe_map]
assert len(missing_phe) == 0, f"{len(missing_phe)} genotype IDs lack phenotype: {missing_phe[:10]}"
y = np.array([phe_map[int(s)] for s in sample_ids], dtype=np.float64)
n_pheno = int(np.isfinite(y).sum())
print(f"phenotype '{TRAIT}' aligned: n={len(y)}, finite={n_pheno}, mean={np.nanmean(y):.4f}, std={np.nanstd(y):.4f}", flush=True)
assert n_pheno == n_ind, "phenotype has non-finite values; downstream expects complete"

# ---------------------------------------------------------------------------
# 3. GRM (VanRaden method 1)  G = ZZ' / (2 * sum p_i(1-p_i)); Z = M - 2p; mean-impute missing
# ---------------------------------------------------------------------------
print("\n=== STEP 3: GRM VanRaden method 1 ===", flush=True)
# mean-impute missing dosage to 2p (so centered Z=0 at missing -> standard VanRaden handling)
M_imp = Mw.copy()
inds = np.where(missing_mask)
if inds[0].size > 0:
    M_imp[inds] = (2.0 * p)[inds[1]]
Z = M_imp - 2.0 * p[None, :]
denom = 2.0 * np.sum(p * (1.0 - p))
G = (Z @ Z.T) / denom
print(f"GRM shape {G.shape}; scaling denom 2*sum(p(1-p))={denom:.4f}", flush=True)

diag = np.diag(G).copy()
iu = np.triu_indices(n_ind, k=1)
offdiag = G[iu]
print(f"diag: mean={diag.mean():.4f} min={diag.min():.4f} max={diag.max():.4f}", flush=True)
print(f"offdiag: mean={offdiag.mean():.4f} min={offdiag.min():.4f} max={offdiag.max():.4f}", flush=True)

qs = [10, 25, 50, 75, 90, 95, 99]
offq = {f"q{q}": float(np.percentile(offdiag, q)) for q in qs}
diagq = {f"q{q}": float(np.percentile(diag, q)) for q in qs}
print("offdiag quantiles:", {k: round(v, 4) for k, v in offq.items()}, flush=True)

# histogram bins for offdiag relatedness
hist_counts, hist_edges = np.histogram(offdiag, bins=40)
grm_stats = {
    "n_individuals": n_ind,
    "scaling_denominator_2sum_p_1mp": float(denom),
    "diag": {"mean": float(diag.mean()), "std": float(diag.std()), "min": float(diag.min()),
             "max": float(diag.max()), "quantiles": diagq},
    "offdiag": {"mean": float(offdiag.mean()), "std": float(offdiag.std()), "min": float(offdiag.min()),
                "max": float(offdiag.max()), "n_pairs": int(offdiag.size), "quantiles": offq,
                "histogram": {"counts": [int(c) for c in hist_counts],
                              "bin_edges": [float(e) for e in hist_edges]}},
}
dump_json(os.path.join(OUT_ROOT, "grm", "grm_stats.json"), grm_stats)

# Save GRM diagonal per-id and the relatedness quantile table as CSV (audit; not the full matrix)
os.makedirs(os.path.join(OUT_ROOT, "grm"), exist_ok=True)
pd.DataFrame({"sample_id": sample_ids.astype(int), "GRM_diagonal": diag}).to_csv(
    os.path.join(OUT_ROOT, "grm", "grm_diagonal_per_id.csv"), index=False)
pd.DataFrame({"quantile": [f"q{q}" for q in qs],
              "offdiag_relatedness": [offq[f"q{q}"] for q in qs],
              "diag_value": [diagq[f"q{q}"] for q in qs]}).to_csv(
    os.path.join(OUT_ROOT, "grm", "grm_offdiag_quantiles.csv"), index=False)
pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:], "count": hist_counts}).to_csv(
    os.path.join(OUT_ROOT, "grm", "grm_offdiag_histogram.csv"), index=False)

# id -> row index map for relatedness lookups
id2row = {int(s): i for i, s in enumerate(sample_ids)}
all_ids = sample_ids.astype(int)

def manifest_relatedness(train_ids, test_ids):
    """Per-test-individual realized max relatedness to the train set + summary."""
    tr_rows = np.array([id2row[i] for i in train_ids], dtype=int)
    te_rows = np.array([id2row[i] for i in test_ids], dtype=int)
    rows = []
    if len(tr_rows) == 0:
        for tid in test_ids:
            rows.append({"sample_id": int(tid), "max_train_relatedness": float("nan"),
                         "mean_train_relatedness": float("nan")})
        return rows, {"max": None, "mean_of_max": None}
    sub = G[np.ix_(te_rows, tr_rows)]  # (n_test, n_train)
    mx = sub.max(axis=1)
    mn = sub.mean(axis=1)
    for k, tid in enumerate(test_ids):
        rows.append({"sample_id": int(tid), "max_train_relatedness": float(mx[k]),
                     "mean_train_relatedness": float(mn[k])})
    return rows, {"max": float(mx.max()), "mean_of_max": float(mx.mean()),
                  "median_of_max": float(np.median(mx))}

# ---------------------------------------------------------------------------
# 4a. RANDOM: repeated 5-fold x 10 = 50 folds
# ---------------------------------------------------------------------------
print("\n=== STEP 4a: random repeated 5-fold x 10 ===", flush=True)
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
random_dir = os.path.join(OUT_ROOT, "random")
random_manifest = {"scheme": "RepeatedKFold", "n_splits": 5, "n_repeats": 10, "n_folds": 50,
                   "seed": SEED, "library_alias": "random5fold10", "folds": []}
idx_arr = np.arange(n_ind)
for fi, (tr, te) in enumerate(rkf.split(idx_arr)):
    rep = fi // 5
    k = fi % 5
    tag = f"rep{rep:02d}_fold{k}"
    train_ids = all_ids[tr].tolist()
    test_ids = all_ids[te].tolist()
    write_ids(os.path.join(random_dir, f"{tag}_train.csv"), train_ids)
    write_ids(os.path.join(random_dir, f"{tag}_test.csv"), test_ids)
    rel_rows, rel_sum = manifest_relatedness(train_ids, test_ids)
    pd.DataFrame(rel_rows).to_csv(os.path.join(random_dir, f"{tag}_test_relatedness.csv"), index=False)
    random_manifest["folds"].append({"fold_id": fi, "tag": tag, "repeat": rep, "inner_fold": k,
                                      "n_train": len(train_ids), "n_test": len(test_ids),
                                      "realized_max_train_relatedness": rel_sum})
dump_json(os.path.join(random_dir, "manifest.json"), random_manifest)
print(f"random: wrote 50 folds (n_train~{random_manifest['folds'][0]['n_train']}, "
      f"n_test~{random_manifest['folds'][0]['n_test']})", flush=True)

# ---------------------------------------------------------------------------
# Hierarchical clustering on GRM (shared by group / leave-cluster / for reference)
# distance = max_relatedness - relatedness, average linkage
# ---------------------------------------------------------------------------
print("\n=== Hierarchical clustering on GRM ===", flush=True)
# distance matrix: larger relatedness => smaller distance. Use (c - G) with c = max offdiag-ish.
dmax = float(G.max())
D = dmax - G
np.fill_diagonal(D, 0.0)
D = (D + D.T) / 2.0
D[D < 0] = 0.0
condensed = squareform(D, checks=False)
Zlink = linkage(condensed, method="average")

def cluster_at_k(target_k):
    """Cut tree to get exactly target_k flat clusters (maxclust)."""
    labels = fcluster(Zlink, t=target_k, criterion="maxclust")
    return labels

# ---------------------------------------------------------------------------
# 4b. RELATIONSHIP_GROUP: group 5-fold and group 10-fold (same-cluster never split)
#     Strategy: form many fine clusters, then greedily pack clusters into K folds
#     balancing size, so that an entire cluster stays in one test fold.
# ---------------------------------------------------------------------------
print("\n=== STEP 4b: relationship group k-fold ===", flush=True)

def group_kfold(n_groups_target_factor, K, name):
    """Build group K-fold. Cluster individuals, then assign whole clusters to K folds
    (greedy largest-cluster-first into currently-smallest fold) for balance.
    A cluster's members are the TEST set when its fold is the held-out fold."""
    # use a fairly fine clustering so packing into K balanced folds is possible
    n_clusters = max(K, min(n_ind, K * n_groups_target_factor))
    labels = cluster_at_k(n_clusters)
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(int(all_ids[i]))
    # greedy balanced packing of clusters into K bins (deterministic: sort by size desc then id)
    sorted_clusters = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    fold_members = {f: [] for f in range(K)}
    fold_sizes = {f: 0 for f in range(K)}
    cluster_to_fold = {}
    for cid, members in sorted_clusters:
        f = min(range(K), key=lambda x: (fold_sizes[x], x))
        fold_members[f].extend(members)
        fold_sizes[f] += len(members)
        cluster_to_fold[cid] = f
    out_dir = os.path.join(OUT_ROOT, "relationship_group", name)
    man = {"scheme": f"group_{K}fold_on_GRM_hier_clusters", "K": K, "seed": SEED,
           "n_fine_clusters": int(n_clusters), "linkage": "average",
           "distance": "max(G)-G", "folds": []}
    # cluster membership table
    crows = [{"sample_id": s, "cluster_id": int(lab)} for s, lab in zip(all_ids.tolist(), labels.tolist())]
    pd.DataFrame(crows).to_csv(_ensure(os.path.join(out_dir, "cluster_membership.csv")), index=False)
    for f in range(K):
        test_ids = sorted(fold_members[f])
        train_ids = sorted(set(all_ids.tolist()) - set(test_ids))
        tag = f"fold{f}"
        write_ids(os.path.join(out_dir, f"{tag}_train.csv"), train_ids)
        write_ids(os.path.join(out_dir, f"{tag}_test.csv"), test_ids)
        rel_rows, rel_sum = manifest_relatedness(train_ids, test_ids)
        pd.DataFrame(rel_rows).to_csv(os.path.join(out_dir, f"{tag}_test_relatedness.csv"), index=False)
        man["folds"].append({"fold_id": f, "tag": tag, "n_train": len(train_ids),
                             "n_test": len(test_ids), "n_clusters_in_test_fold":
                             int(sum(1 for c, ff in cluster_to_fold.items() if ff == f)),
                             "realized_max_train_relatedness": rel_sum})
    dump_json(os.path.join(out_dir, "manifest.json"), man)
    ntr = [x["n_train"] for x in man["folds"]]; nte = [x["n_test"] for x in man["folds"]]
    print(f"{name}: K={K} n_train[{min(ntr)}-{max(ntr)}] n_test[{min(nte)}-{max(nte)}] "
          f"fine_clusters={n_clusters}", flush=True)
    return man

def _ensure(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

group_kfold(8, 5, "group_5fold")
group_kfold(8, 10, "group_10fold")

# ---------------------------------------------------------------------------
# 4c. LEAVE_CLUSTER_OUT: leave one genetic cluster out (per cluster)
#     Use a coarse clustering so clusters are meaningful genetic groups.
# ---------------------------------------------------------------------------
print("\n=== STEP 4c: leave-cluster-out ===", flush=True)
LCO_K = 10  # number of genetic clusters to leave out one at a time
lco_labels = cluster_at_k(LCO_K)
lco_clusters = {}
for i, lab in enumerate(lco_labels):
    lco_clusters.setdefault(int(lab), []).append(int(all_ids[i]))
lco_dir = os.path.join(OUT_ROOT, "leave_cluster_out")
lco_man = {"scheme": "leave_one_genetic_cluster_out", "n_clusters": LCO_K, "seed": SEED,
           "linkage": "average", "distance": "max(G)-G", "folds": []}
pd.DataFrame([{"sample_id": s, "cluster_id": int(lab)} for s, lab in zip(all_ids.tolist(), lco_labels.tolist())]
             ).to_csv(_ensure(os.path.join(lco_dir, "cluster_membership.csv")), index=False)
for cid in sorted(lco_clusters.keys()):
    test_ids = sorted(lco_clusters[cid])
    train_ids = sorted(set(all_ids.tolist()) - set(test_ids))
    tag = f"cluster{cid:02d}"
    write_ids(os.path.join(lco_dir, f"{tag}_train.csv"), train_ids)
    write_ids(os.path.join(lco_dir, f"{tag}_test.csv"), test_ids)
    rel_rows, rel_sum = manifest_relatedness(train_ids, test_ids)
    pd.DataFrame(rel_rows).to_csv(os.path.join(lco_dir, f"{tag}_test_relatedness.csv"), index=False)
    lco_man["folds"].append({"left_out_cluster": cid, "tag": tag, "n_train": len(train_ids),
                             "n_test": len(test_ids), "realized_max_train_relatedness": rel_sum})
dump_json(os.path.join(lco_dir, "manifest.json"), lco_man)
nte = [x["n_test"] for x in lco_man["folds"]]
print(f"leave_cluster_out: {LCO_K} clusters, test sizes {min(nte)}-{max(nte)}", flush=True)

# ---------------------------------------------------------------------------
# 4d. RELATEDNESS_BUDGET (core new experiment)
#     For each test individual, max relatedness to the TRAIN set <= tau (a ceiling).
#
#     DESIGN NOTE (deviation from literal {q90,q75,q50,q25,q10}, documented & justified):
#     VanRaden GRM off-diagonals are population-structure-CENTERED around 0, so the lower
#     offdiag percentiles (q50~-0.001, q25~-0.025, q10~-0.052) are <=0 and are NOT meaningful
#     relatedness ceilings: requiring "max-train-relatedness <= a value below the median"
#     forces ~all individuals out (empirically -> empty train at q75 and below). A relatedness
#     *budget* must live in the UPPER tail, where positive values represent close-relative
#     cutoffs (q99~0.155 ~ half-sib/grandparent range; q90~0.044 ~ distant). We therefore
#     sweep tau over UPPER-tail quantiles {q99.5,q99,q97.5,q95,q90} (loose->strict = ceiling
#     descending), which yields the intended monotonic genetic-distance generalization ladder.
#     The literal lower-quantile values are still recorded in grm_stats for full audit.
# ---------------------------------------------------------------------------
print("\n=== STEP 4d: relatedness budget ===", flush=True)
# upper-tail tau ladder (loose -> strict): ceiling on max relatedness to train set
tau_pcts = [99.5, 99.0, 97.5, 95.0, 90.0]
tau_levels = [(f"q{('%g' % q)}", float(np.percentile(offdiag, q))) for q in tau_pcts]
budget_root = os.path.join(OUT_ROOT, "relatedness_budget")
rng = np.random.default_rng(SEED)

# We fix ONE held-out test set, then for each tau filter the eligible TRAIN individuals so that
# every test individual's max relatedness to the (retained) train set is <= tau. The test set is
# shared across all tau (the tightening of tau is the ONLY moving part). A 10% test set is used so
# the strictest tau still leaves a usable, non-empty training pool (verified empirically: q90 -> 97).
n_test_budget = int(round(0.10 * n_ind))
perm = rng.permutation(n_ind)
test_rows_budget = np.sort(perm[:n_test_budget])
pool_train_rows = np.sort(perm[n_test_budget:])   # candidate training pool (the other 90%)
test_ids_budget = all_ids[test_rows_budget].tolist()
pool_train_ids = all_ids[pool_train_rows].tolist()
print(f"budget: tau ladder = {[(n, round(t,4)) for n,t in tau_levels]}", flush=True)
print(f"budget: fixed test set n={len(test_ids_budget)}, candidate train pool n={len(pool_train_ids)}", flush=True)

# Precompute relatedness between every pool-train individual and every test individual.
# For a given tau, a TRAIN individual is admissible iff its max relatedness to ANY test
# individual <= tau  (guarantees every test individual's max-train-relatedness <= tau).
sub_pt = G[np.ix_(pool_train_rows, test_rows_budget)]  # (n_pool, n_test)
pool_max_to_test = sub_pt.max(axis=1)                  # per-pool-train max relatedness to test set

# Determine admissible train counts at each tau to find matched_size = min across tau
admissible = {}
for name, tau in tau_levels:
    adm = pool_max_to_test <= tau
    admissible[name] = adm
    print(f"  tau {name}={tau:.4f}: admissible_train={int(adm.sum())}", flush=True)
matched_n = min(int(admissible[name].sum()) for name, _ in tau_levels)
print(f"matched_size train n = {matched_n} (min over tau)", flush=True)

budget_summary = {"seed": SEED, "test_set_size": len(test_ids_budget),
                  "candidate_train_pool_size": len(pool_train_ids),
                  "tau_definition": "GRM offdiag UPPER-tail percentiles {q99.5,q99,q97.5,q95,q90} "
                                    "used as max-relatedness ceiling (loose->strict); lower "
                                    "percentiles omitted because centered GRM makes them <=0 "
                                    "and degenerate as ceilings",
                  "test_set_fraction": 0.10,
                  "tau_levels": {name: float(tau) for name, tau in tau_levels},
                  "admissible_train_counts": {name: int(admissible[name].sum()) for name, _ in tau_levels},
                  "matched_size_train_n": matched_n,
                  "natural": {}, "matched_size": {}}

# write shared test set once
write_ids(os.path.join(budget_root, "test_set.csv"), test_ids_budget)

for version in ("natural", "matched_size"):
    for name, tau in tau_levels:
        adm = admissible[name]
        adm_rows = pool_train_rows[adm]
        adm_ids = all_ids[adm_rows].tolist()
        if version == "natural":
            train_ids = sorted(adm_ids)
        else:
            # deterministic subsample to matched_n from admissible set (seeded)
            # use a stable per-tau offset (NOT Python's randomized hash) for reproducibility
            tau_offset = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100000
            sub_rng = np.random.default_rng(SEED + tau_offset)
            if len(adm_ids) >= matched_n:
                pick = sub_rng.choice(len(adm_ids), size=matched_n, replace=False)
                train_ids = sorted(int(adm_ids[j]) for j in pick)
            else:
                train_ids = sorted(adm_ids)
        out_dir = os.path.join(budget_root, version, name)
        write_ids(os.path.join(out_dir, "train.csv"), train_ids)
        write_ids(os.path.join(out_dir, "test.csv"), test_ids_budget)  # same test set
        rel_rows, rel_sum = manifest_relatedness(train_ids, test_ids_budget)
        rel_df = pd.DataFrame(rel_rows)
        rel_df.to_csv(os.path.join(out_dir, "test_relatedness.csv"), index=False)
        realized_max = float(rel_df.max_train_relatedness.max())
        # audit: every test individual's realized max-train-relatedness must be <= tau
        violations = int((rel_df.max_train_relatedness > tau + 1e-9).sum())
        man = {"version": version, "tau_level": name, "tau_value": float(tau), "seed": SEED,
               "n_train": len(train_ids), "n_test": len(test_ids_budget),
               "realized_max_train_relatedness_over_test": realized_max,
               "tau_violations": violations}
        dump_json(os.path.join(out_dir, "manifest.json"), man)
        budget_summary[version][name] = {"n_train": len(train_ids), "n_test": len(test_ids_budget),
                                          "tau": float(tau), "realized_max": realized_max,
                                          "violations": violations}
dump_json(os.path.join(budget_root, "budget_summary.json"), budget_summary)
print("budget natural n_train per tau:", {k: v["n_train"] for k, v in budget_summary["natural"].items()}, flush=True)
print("budget matched n_train per tau:", {k: v["n_train"] for k, v in budget_summary["matched_size"].items()}, flush=True)
print("budget violations (must be 0):",
      {k: v["violations"] for k, v in budget_summary["natural"].items()},
      {k: v["violations"] for k, v in budget_summary["matched_size"].items()}, flush=True)

# ---------------------------------------------------------------------------
# Top-level provenance manifest for Holstein
# ---------------------------------------------------------------------------
prov = {
    "dataset": "holstein_1092",
    "trait": TRAIT,
    "seed": SEED,
    "genotype_source_npy": os.path.join(QC_DIR, "holstein_genotype_matrix_qc.npy"),
    "genotype_shape": [n_ind, n_snp],
    "genotype_encoding": "0/1/2 allele dosage (float32)",
    "overall_missing_rate": overall_missing_rate,
    "phenotype_source": os.path.join(QC_DIR, "holstein_phenotype_table.csv"),
    "n_phenotype_finite": n_pheno,
    "grm_method": "VanRaden method 1: Z=M-2p, G=ZZ'/(2*sum p(1-p)), missing mean-imputed to 2p",
    "clustering": "scipy average-linkage on distance (max(G)-G), maxclust cut",
    "split_families": ["random (repeated 5-fold x10 = 50)",
                       "relationship_group (group5, group10)",
                       "leave_cluster_out (10 clusters)",
                       "relatedness_budget (tau in upper-tail {q99.5,q99,q97.5,q95,q90}; "
                       "natural + matched_size; 10% fixed test set)"],
    "relatedness_budget_design_note": "VanRaden GRM offdiag is centered ~0; lower percentiles "
                       "(q50/q25/q10) are <=0 and degenerate as relatedness ceilings. The budget "
                       "ladder therefore uses upper-tail percentiles as max-relatedness ceilings.",
    "v51_crossref": "v5.1 split types high_relatedness_cluster_group5 / relationship_aware_group5 "
                    "were performance-reported only; member IDs were NOT exported. This set is the "
                    "new authoritative export.",
}
dump_json(os.path.join(OUT_ROOT, "holstein_provenance.json"), prov)
print("\n=== DONE ===", flush=True)
print("Output root:", OUT_ROOT, flush=True)
