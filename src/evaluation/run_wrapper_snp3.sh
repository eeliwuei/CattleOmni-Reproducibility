# CattleOmni reproducibility repository -- sanitised analysis script.
# Paths are relative to the repository root (see configs/default.yaml and data/README.md);
# all randomness uses fixed seeds. No server paths, hostnames, credentials, or raw data are included.
#!/bin/bash
cd .
O=benchmarks/snp3
mkdir -p "$O"; rm -f "$O/ALL_DONE_snp3"
for i in $(seq 1 30); do
  echo "[snp3-wrapper] attempt $i $(date +%H:%M:%S)" >> "$O/snp3_run.log"
  nice -n 10 python3 scripts/snp3_budget.py >> "$O/snp3_run.log" 2>&1
  [ -f "$O/ALL_DONE_snp3" ] && { echo "[snp3-wrapper] DONE attempt $i" >> "$O/snp3_run.log"; break; }
  echo "[snp3-wrapper] attempt $i no sentinel; resume 5s" >> "$O/snp3_run.log"; sleep 5
done
