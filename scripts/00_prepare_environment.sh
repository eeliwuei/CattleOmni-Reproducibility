#!/usr/bin/env bash
set -euo pipefail
# Create env: conda env create -f environment.yml && conda activate cattleomni
# Copy .env.example to .env and set CATTLEOMNI_DATA_ROOT to your processed-data location.
cp -n .env.example .env || true
echo "Environment notes printed; install deps via conda/pip and set .env."
