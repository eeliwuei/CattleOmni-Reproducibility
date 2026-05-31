# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/bin/bash
# Full BayesB ladder: 30 folds (random5 + group5 + group10 + lco10), full-164k, parallel.
cd .
SP=splits/holstein; LAD=/tmp/bglr_ladder; mkdir -p $LAD
export R_LIBS_USER=~/Rlibs
CAP=${CAP:-12}; NITER=${NITER:-12000}; BURNIN=${BURNIN:-2000}; MODE=${MODE:-BayesB_sparse}
export NITER BURNIN MODE
FOLDS=()
for i in 0 1 2 3 4; do FOLDS+=("$SP/random/rep00_fold${i}_train.csv|random|rep00_fold${i}"); done
for i in 0 1 2 3 4; do FOLDS+=("$SP/relationship_group/group_5fold/fold${i}_train.csv|group5|fold${i}"); done
for i in 0 1 2 3 4 5 6 7 8 9; do FOLDS+=("$SP/relationship_group/group_10fold/fold${i}_train.csv|group10|fold${i}"); done
for c in 01 02 03 04 05 06 07 08 09 10; do FOLDS+=("$SP/leave_cluster_out/cluster${c}_train.csv|leave_cluster_out|cluster${c}"); done
run_one(){
  IFS='|' read -r tcsv fam fold <<< "$1"
  wd=$LAD/${fam}_${fold}; mkdir -p $wd
  python3 scripts/bglr_prep2.py "$tcsv" "$wd" "$fam" "$fold" > $wd/prep.log 2>&1 || { echo "PREP FAIL $fam/$fold"; return; }
  nice -n 12 env NITER=$NITER BURNIN=$BURNIN MODE=$MODE Rscript scripts/bglr_run2.R "$wd" > $wd/run.log 2>&1 || echo "RUN FAIL $fam/$fold"
  cat $wd/run.log | grep -E "pearson|FAIL" || true
}
export -f run_one; export LAD R_LIBS_USER NITER BURNIN MODE
printf '%s\n' "${FOLDS[@]}" | xargs -P $CAP -I {} bash -c 'run_one "$@"' _ {}
echo "=== LADDER DONE — concat per-sample ==="
first=$(ls $LAD/*/persample.csv 2>/dev/null | head -1)
head -1 "$first" > $LAD/bayesb_per_sample.csv
for f in $LAD/*/persample.csv; do tail -n +2 "$f" >> $LAD/bayesb_per_sample.csv; done
echo "rows: $(wc -l < $LAD/bayesb_per_sample.csv)"
echo "=== pooled per-fold results ==="; cat $LAD/*/result.csv | grep -v "^.model.,.family." | sort
