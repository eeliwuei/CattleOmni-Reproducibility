# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/usr/bin/env python3
"""Aggregate Holstein benchmark fold metrics -> summary / relatedness-decay / significance / figures.
Pure numpy (no pandas/sklearn). Figures via matplotlib if available (Agg)."""
import os, sys, csv, json, math
import numpy as np

B = "benchmarks/holstein"
MET = sys.argv[1] if len(sys.argv) > 1 else f"{B}/fold_metrics_full.csv"

# family -> (stringency_order, nominal max train-test relatedness, kind)
FAM = {
    "random":            (0, 0.903, "categorical"),   # optimistic: relatives in train (A3 mean max-rel)
    "group10":           (1, 0.40,  "categorical"),
    "group5":            (2, 0.30,  "categorical"),
    "leave_cluster_out": (3, 0.20,  "categorical"),
    "budget_natural":    (4, None,  "budget"),         # tau read from label
    "budget_matched_size":(4, None, "budget"),
}
TAU = {"q99.5": 0.2121, "q99": 0.1548, "q97.5": 0.0961, "q95": 0.0658, "q90": 0.0438}  # A3 budget_summary

def load():
    rows = []
    with open(MET, newline="") as f:
        for d in csv.DictReader(line.replace("\x00", "") for line in f):
            if d["pearson"] in ("ERR", "", "nan"):
                pe = float("nan")
            else:
                pe = float(d["pearson"])
            rows.append(dict(mode=d["mode"], model=d["model"], family=d["family"], fold=d["fold"],
                             n_train=int(d["n_train"]), n_test=int(d["n_test"]), pearson=pe,
                             r2=(float(d["r2"]) if d["r2"] not in ("ERR","","nan") else float("nan")),
                             rmse=(float(d["rmse"]) if d["rmse"] not in ("ERR","","nan") else float("nan"))))
    return rows

def agg(rows):
    keys = {}
    for r in rows:
        keys.setdefault((r["mode"], r["model"], r["family"]), []).append(r)
    out = []
    for (mode, model, fam), rs in sorted(keys.items()):
        pe = np.array([x["pearson"] for x in rs], float); pe = pe[~np.isnan(pe)]
        r2 = np.array([x["r2"] for x in rs], float); r2 = r2[~np.isnan(r2)]
        out.append(dict(mode=mode, model=model, family=fam, n_folds=len(rs),
                        pearson_mean=(pe.mean() if len(pe) else float("nan")),
                        pearson_std=(pe.std() if len(pe) else float("nan")),
                        r2_mean=(r2.mean() if len(r2) else float("nan"))))
    return out

def paired_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 3: return (float("nan"), float("nan"), len(a))
    d = a - b; t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)) + 1e-12)
    # two-sided p via normal approx (df large enough for our fold counts)
    p = math.erfc(abs(t) / math.sqrt(2))
    return (float(d.mean()), float(t), float(p), len(d))

def fold_map(rows, mode, model, family):
    return {r["fold"]: r["pearson"] for r in rows if r["mode"]==mode and r["model"]==model and r["family"]==family}

def main():
    rows = load()
    summ = agg(rows)
    with open(f"{B}/model_split_summary.csv", "w") as f:
        f.write("mode,model,family,n_folds,pearson_mean,pearson_std,r2_mean\n")
        for s in summ:
            f.write(f"{s['mode']},{s['model']},{s['family']},{s['n_folds']},{s['pearson_mean']:.4f},{s['pearson_std']:.4f},{s['r2_mean']:.4f}\n")

    # relatedness decay table
    with open(f"{B}/relatedness_decay.csv", "w") as f:
        f.write("model,family,relatedness,pearson_mean,pearson_std,n_folds\n")
        for s in summ:
            if s["mode"] != "strict": continue
            fam = s["family"]
            if fam.startswith("budget"):
                # one row per tau handled below
                continue
            rel = FAM.get(fam, (9, float("nan"), ""))[1]
            f.write(f"{s['model']},{fam},{rel:.4f},{s['pearson_mean']:.4f},{s['pearson_std']:.4f},{s['n_folds']}\n")
        # budget: per-tau (family label encodes tau like natural_q90)
        for r in rows:
            if r["mode"]=="strict" and r["family"].startswith("budget"):
                tau = None
                for k,v in TAU.items():
                    if r["fold"].endswith(k): tau=v
                if tau is None: continue
                f.write(f"{r['model']},{r['family']},{tau:.4f},{r['pearson']:.4f},0.0,1\n")

    # significance: MLP vs GBLUP (strict) per family; MLP strict-vs-leak (random); SparseGate vs GBLUP
    with open(f"{B}/significance.csv", "w") as f:
        f.write("contrast,mode,family,delta_pearson,t_stat,p_value,n_pairs\n")
        for fam in ["random","group5","group10","leave_cluster_out"]:
            a = fold_map(rows,"strict","FullSNP_MLP_top50k",fam)
            b = fold_map(rows,"strict","GBLUP_full",fam)
            common = sorted(set(a)&set(b))
            res = paired_t([a[k] for k in common],[b[k] for k in common])
            if len(res)==4:
                f.write(f"MLP_vs_GBLUP,strict,{fam},{res[0]:.4f},{res[1]:.3f},{res[2]:.2e},{res[3]}\n")
        # MLP strict vs leaky on random (paired by fold)
        a = fold_map(rows,"strict","FullSNP_MLP_top50k","random")
        b = fold_map(rows,"historical","FullSNP_MLP_top50k","random")
        common = sorted(set(a)&set(b))
        res = paired_t([a[k] for k in common],[b[k] for k in common])
        if len(res)==4:
            f.write(f"MLP_strict_vs_leak,random,random,{res[0]:.4f},{res[1]:.3f},{res[2]:.2e},{res[3]}\n")

    # report
    def row(model, fam, mode="strict"):
        for s in summ:
            if s["mode"]==mode and s["model"]==model and s["family"]==fam: return s
        return None
    lines = ["# Holstein leakage-aware genomic prediction — benchmark report\n",
             f"Folds aggregated from `{MET}`. STRICT = official (train-fold-only preprocessing, variance SNP selection, LayerNorm MLP). LEAKY = diagnostic (global supervised selection).\n",
             "\n## Strict Pearson (mean±std) by model × validation stringency\n",
             "| model | random | group10 | group5 | leave_cluster_out |",
             "|---|---|---|---|---|"]
    for model in ["GBLUP_full","PCARidge_PCA80","Ridge_top50k","Ridge_top20k","FullSNP_MLP_top50k","SparseGate_MLP_top20k"]:
        cells=[]
        for fam in ["random","group10","group5","leave_cluster_out"]:
            s=row(model,fam)
            cells.append(f"{s['pearson_mean']:.3f}±{s['pearson_std']:.3f}" if s else "-")
        lines.append(f"| {model} | "+" | ".join(cells)+" |")
    # leak vs strict
    ms=row("FullSNP_MLP_top50k","random","strict"); ml=row("FullSNP_MLP_top50k","random","historical")
    if ms and ml:
        lines += ["\n## Leakage demonstration (random CV)\n",
                  f"- FullSNP_MLP strict (clean) = **{ms['pearson_mean']:.3f}**; leaky (global selection) = **{ml['pearson_mean']:.3f}**; inflation = **+{ml['pearson_mean']-ms['pearson_mean']:.3f}**"]
    open(f"{B}/HOLSTEIN_BENCHMARK_REPORT.md","w").write("\n".join(lines)+"\n")
    print("WROTE summary / decay / significance / report to", B)
    for s in summ:
        if s["model"] in ("FullSNP_MLP_top50k","GBLUP_full") and s["mode"]=="strict":
            print(f"  {s['model']:24s} {s['family']:20s} r={s['pearson_mean']:.3f}±{s['pearson_std']:.3f} (n={s['n_folds']})")

    # figures (optional)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        models=["GBLUP_full","PCARidge_PCA80","Ridge_top50k","FullSNP_MLP_top50k","SparseGate_MLP_top20k"]
        # Fig2 stringency
        fams=["random","group10","group5","leave_cluster_out"]; x=np.arange(len(fams))
        plt.figure(figsize=(8,5))
        for model in models:
            ys=[]; es=[]
            for fam in fams:
                s=row(model,fam); ys.append(s["pearson_mean"] if s else np.nan); es.append(s["pearson_std"] if s else 0)
            plt.errorbar(x, ys, yerr=es, marker="o", capsize=3, label=model)
        plt.xticks(x, ["random\n(rel~0.9)","group10","group5","leave-\ncluster-out"]); plt.ylabel("Pearson r"); plt.title("Holstein: prediction vs validation stringency (STRICT)")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{B}/fig2_stringency.png", dpi=140); plt.close()
        # Fig3 budget decay
        plt.figure(figsize=(8,5)); taus=[TAU[k] for k in ["q99.5","q99","q97.5","q95","q90"]]
        for model in models:
            ys=[]
            for k in ["q99.5","q99","q97.5","q95","q90"]:
                v=[r["pearson"] for r in rows if r["mode"]=="strict" and r["model"]==model and r["family"]=="budget_natural" and r["fold"].endswith(k)]
                ys.append(v[0] if v else np.nan)
            plt.plot(taus, ys, marker="o", label=model)
            sr=row(model,"random");
            if sr: plt.axhline(sr["pearson_mean"], ls=":", alpha=0.25)
        plt.gca().invert_xaxis(); plt.xlabel("max allowed train-test relatedness τ (stricter →)"); plt.ylabel("Pearson r")
        plt.title("Holstein: genetic-distance budget decay (STRICT, natural)"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{B}/fig3_budget_decay.png", dpi=140); plt.close()
        print("FIGURES_OK fig2_stringency.png fig3_budget_decay.png")
    except Exception as e:
        print("FIGURES_SKIPPED (matplotlib unavailable):", e)

if __name__ == "__main__":
    main()
