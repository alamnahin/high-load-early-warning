"""
Shared configuration for the early-warning preprocessing pipeline.

Defaults: raw CSVs in <repo>/data/raw, outputs under <repo>. Override with the RAW_DIR / OUT_DIR
environment variables. Everything else in the pipeline follows from the settings below.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]                  # repository root
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))   # put the 3 original CSVs here
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT))             # generated files go below this folder

CLEAN_DIR = OUT_DIR / "data" / "clean"
MODEL_DIR = OUT_DIR / "data" / "model_ready"
FIG_DIR = OUT_DIR / "reports" / "figures"
TAB_DIR = OUT_DIR / "reports" / "tables"
for _d in (CLEAN_DIR, MODEL_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw files are found by prefix, so "_1" suffixes etc. do not matter.
DATASETS = {"D1": "dataset_01_*.csv", "D2": "dataset_02_*.csv", "D3": "dataset_03_*.csv"}
DATASET_NOTES = {
    "D1": "Baseline traffic (2-5 Jun 2025)",
    "D2": "Different distribution, heavy load (5-7 Jun 2025)",
    "D3": "Bursty / short high-load events (7-10 Jun 2025)",
}


def raw_path(ds_id: str) -> Path:
    hits = sorted(RAW_DIR.glob(DATASETS[ds_id]))
    if not hits:
        raise FileNotFoundError(f"No file matching {DATASETS[ds_id]} in {RAW_DIR}")
    return hits[0]


# ------------------------------------------------------------ time grid
SAMPLING_SEC = 10                       # nominal polling interval
VALID_TS_RANGE = ("2025-01-01", "2026-12-31")  # sanity window for Unix timestamps

# --------------------------------------------------------------- labels
# LABEL = bin of max(bw_rx, bw_tx over all interfaces) in Mbps (verified: 100 % agreement)
LABEL_EDGES_MBPS = [10, 40, 70, 90]
LABEL_NAMES = {0: "Low (<10)", 1: "Moderate (10-40)", 2: "Higher (40-70)",
               3: "High (70-90)", 4: "Very high (>=90)"}

IFACES = [3, 4, 5, 6]                   # interface suffixes present in the files
MAX_IMPUTE_GAP = 2                      # forward-fill at most 2 samples (20 s)

# ---------------------------------------------------- targets / features
HORIZONS_SEC = [10, 20, 30, 60]         # prediction horizons H
THRESHOLDS = [3, 4]                     # binary target: future LABEL >= thr
PRIMARY_THRESHOLD = 4
ROLL_STEPS = [3, 6]                     # causal rolling windows: 30 s and 60 s
LAG_STEPS = [1, 2, 3]                   # lagged load values

# Router groups. Inferred from the data (bw_max correlation structure, see EDA
# figure 6) - NOT from official topology documentation.
HOST_GROUPS = {
    "core": ["r1", "r5"],
    "grpA": ["r2", "r3", "r4"],
    "grpB": ["r6", "r8", "r9"],
    "idle": ["r7"],
}

# ------------------------------------------------------------ idle tail
# D1 and D2 end with hours of zero traffic (the traffic generator stopped). Rows after the
# last timestamp at which ANY router reached IDLE_TAIL_MIN_BW Mbps are flagged `idle_tail`.
IDLE_TAIL_MIN_BW = 10
DROP_IDLE_TAIL = True                   # drop them from the model-ready tables (kept in clean data)

# ---------------------------------------------------------------- splits
SPLIT_FRACS = (0.70, 0.15, 0.15)        # chronological train / val / test inside each dataset
SEED = 42
