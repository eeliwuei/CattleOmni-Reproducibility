# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""A9-B Genome Behavior Recognition -- R2: genome-order block-feature sequence builder.
Holstein genotype (1092 x 164312) -> behavior sequence: sample x block x features,
blocks tiling chromosomes in (chr, position) order. Shared foundation for the Genome-TCN
(fast path) and the dual-scale SlowFast paths. Exploratory FORK; representations only
(no phenotype, no GEBV). Also emits genome-order index, block metadata, and a shuffled
block order for the C1 (block-order) ablation control. seed=20260529.
"""
import os, json, csv, time
import numpy as np
SEED = 20260529; np.random.seed(SEED); RNG = np.random.default_rng(SEED)
QC   = "data/processed/holstein/parse_qc"
ROOT = "."
GENO = f"{QC}/holstein_genotype_matrix_qc.npy"
SNPMETA = f"{QC}/holstein_snp_metadata.csv"
ALIGN = f"{QC}/holstein_sample_alignment.csv"
OUT  = f"{ROOT}/genome_behavior"
BLOCK_SNPS = 256
os.makedirs(OUT, exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

rows = []
with open(ALIGN) as f:
    for d in csv.DictReader(f): rows.append((int(d["sample_id"]), int(d["order"])))
rows.sort(key=lambda x: x[1]); sample_ids = np.array([r[0] for r in rows])

chrom = []; pos = []
with open(SNPMETA) as f:
    for d in csv.DictReader(f):
        chrom.append(int(d["chromosome"].replace("chr", ""))); pos.append(int(d["position"]))
chrom = np.array(chrom); pos = np.array(pos); p = len(chrom); log(f"SNPs={p}")

M = np.load(GENO).astype(np.float32)
assert M.shape[1] == p, f"{M.shape} vs {p}"
n = M.shape[0]; log(f"M{M.shape} samples={n}")

order = np.lexsort((pos, chrom))          # genome order: primary chrom, secondary pos
np.save(f"{OUT}/genome_order.npy", order)
Mo = M[:, order]; chrom_o = chrom[order]; pos_o = pos[order]

blocks = []
for ch in range(1, 30):
    idx = np.where(chrom_o == ch)[0]
    if len(idx) == 0: continue
    for s in range(0, len(idx), BLOCK_SNPS):
        blocks.append((ch, idx[s:s + BLOCK_SNPS]))
n_blocks = len(blocks); log(f"n_blocks={n_blocks}")

feat = ["dosage_mean","dosage_std","het_rate","homo_ref_rate","homo_alt_rate",
        "missing_rate","genotype_entropy","local_maf_dev","n_snps_norm","panel_coverage"]
F = len(feat); X = np.zeros((n, n_blocks, F), dtype=np.float32); meta = []
for bi, (ch, blk) in enumerate(blocks):
    Xb = Mo[:, blk]; nb = Xb.shape[1]
    dm = Xb.mean(1); hr = (Xb == 0).mean(1); het = (Xb == 1).mean(1); ha = (Xb == 2).mean(1)
    ent = -(np.where(hr > 0, hr*np.log(hr+1e-12), 0) + np.where(het > 0, het*np.log(het+1e-12), 0)
            + np.where(ha > 0, ha*np.log(ha+1e-12), 0))
    X[:, bi, 0] = dm; X[:, bi, 1] = Xb.std(1); X[:, bi, 2] = het; X[:, bi, 3] = hr; X[:, bi, 4] = ha
    X[:, bi, 5] = 0.0; X[:, bi, 6] = ent; X[:, bi, 7] = np.abs(dm - Xb.mean())
    X[:, bi, 8] = nb / BLOCK_SNPS; X[:, bi, 9] = 1.0
    meta.append((bi, ch, int(pos_o[blk].min()), int(pos_o[blk].max()), nb))

np.save(f"{OUT}/block_features.npy", X)
np.save(f"{OUT}/sample_ids.npy", sample_ids)
np.save(f"{OUT}/block_order_shuffle.npy", RNG.permutation(n_blocks))
json.dump(feat, open(f"{OUT}/feature_names.json", "w"))
with open(f"{OUT}/block_meta.csv", "w") as f:
    f.write("block_id,chromosome,start_pos,end_pos,n_snps\n")
    for r in meta: f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
bpc = {}
for ch, _ in blocks: bpc[ch] = bpc.get(ch, 0) + 1
log(f"blocks/chr: {dict(sorted(bpc.items()))}")
log(f"X{X.shape} -> {OUT}/block_features.npy")
log("feat means: " + ", ".join(f"{fn}={X[:,:,i].mean():.4f}" for i, fn in enumerate(feat)))
log("R2 DONE")
