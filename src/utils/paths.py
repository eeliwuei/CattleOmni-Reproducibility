"""Path resolution for the reproducibility repo.
All inputs are relative to the repository root by default; override with environment variables.
No server paths or credentials are embedded anywhere in this repository."""
import os
DATA_ROOT = os.environ.get("CATTLEOMNI_DATA_ROOT", "data/processed")
WORK_ROOT = os.environ.get("CATTLEOMNI_WORK_ROOT", ".")
SEED = 20260529
