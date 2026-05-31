#!/usr/bin/env python3
"""Step 01: construct the GRM and all validation splits. Wraps src/data/build_holstein_splits.py."""
import runpy, os
os.environ.setdefault("CATTLEOMNI_DATA_ROOT","data/processed")
runpy.run_path("src/data/build_holstein_splits.py", run_name="__main__")
