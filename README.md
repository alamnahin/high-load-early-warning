# High-Load Early Warning - EDA & preprocessing

Exploratory analysis and leakage-safe preprocessing of router telemetry (SNMP-style counters, 9 routers, 10 s sampling,
three capture sessions `TC31`, June 2025) for **robust early warning of high-load network conditions**: predict, for each router,
whether its load will be in a high class (`LABEL >= 3` or `>= 4`, i.e. >= 70 / >= 90 Mbps) 10-60 s ahead, and how that
holds up across datasets with different traffic behavior.

This repository covers the data side (cleaning, EDA, features, targets, splits, checks). Model training is intentionally
not included; [`src/modeling_utils.py`](src/modeling_utils.py) loads the prepared tables for your experiments.

## Headline findings
Full write-up with figures: **[reports/EDA_REPORT.md](reports/EDA_REPORT.md)**

* `LABEL` is exactly the binned maximum interface bandwidth of the same sample (100 % agreement), so *persistence* ("stay in the current state") is a built-in baseline every model must beat.
* D1 and D2 end with hours of zero traffic (16.5 h / 7.6 h); keeping them makes a chronological test split meaningless, so they are flagged and removed from the model-ready tables.
* D2 is a *level* shift (mean load about 2x). D3 is a *burst* shift: LABEL >= 4 events have a median length of 10 s and 62 % last a single sample.
* Persistence F1 for "stay >= 90 Mbps" at 10 / 30 / 60 s: D1 0.78 / 0.61 / 0.32, D2 0.81 / 0.69 / 0.48, D3 0.49 / 0.34 / 0.16.
* Routers form groups ({r1, r5}, {r2, r3, r4}, {r6, r8, r9}, idle r7) and the two branches alternate load.

## Repository layout
```
.
├── README.md
├── Makefile                  # make all | test | clean
├── requirements.txt
├── src/
│   ├── config.py             # every setting: horizons, thresholds, windows, splits, router groups, paths
│   ├── common.py             # label re-derivation, run lengths
│   ├── 01_clean.py           # validate, drop constants, impute, flag idle tail, verify LABEL
│   ├── features.py           # feature / target / split construction (importable)
│   ├── 02_build_features.py  # -> data/model_ready/
│   ├── 03_eda.py             # -> reports/figures, reports/tables
│   ├── 04_validate.py        # causality (no-future-leak) test, target & split checks
│   ├── 05_sanity_baseline.py # optional: gradient boosting vs persistence
│   ├── modeling_utils.py     # load_split / load_full / scale_fit_on_train
│   └── run_all.py
├── data/
│   ├── raw/                  # put the 3 original CSVs here (git-ignored)
│   ├── clean/                # generated (git-ignored)
│   ├── model_ready/          # generated (git-ignored)
│   └── sample/               # small committed sample of the model-ready schema
├── reports/
│   ├── EDA_REPORT.md
│   ├── figures/              # fig01 ... fig10 (PNG)
│   └── tables/               # CSV/JSON behind every number in the report
├── docs/data_dictionary.md
├── tests/                    # unit tests on synthetic data + output checks
└── .github/workflows/ci.yml
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/dataset_0*_TC31_*_labeled_final*.csv data/raw/
make all          # ~4-5 min on one CPU, ~3 GB RAM
make test
```
Pipeline order: `01_clean` -> `02_build_features` -> `04_validate` -> `03_eda` -> `05_sanity_baseline`.
Raw data location and output location can be overridden with the `RAW_DIR` / `OUT_DIR` environment variables.

## Using the prepared data
```python
import sys; sys.path.append("src")
from modeling_utils import load_split, load_full, scale_fit_on_train

# within-dataset, chronological split (router r7 excluded by default)
Xtr, ytr, Xva, yva, Xte, yte, feats = load_split("D3", thr=4, H=30)

# cross-dataset: train on D1's train rows, test on all of D3
Xd1, yd1, *_ = load_split("D1", thr=4, H=30)
Xd3, yd3 = load_full("D3", thr=4, H=30)
```
Always compare against the persistence predictor (`cur_ge4` as the prediction) and report per-dataset prevalence.

## What the preprocessing guarantees
* every feature at time *t* uses only samples <= *t* (or other routers at the same *t*) - verified by a truncation test;
* targets are the same router's label exactly *H* seconds later - verified against the clean table;
* train < val < test in time, separated by a 60 s purge band; scaling is not baked in (fit on train rows only);
* nothing is clipped or resampled: loads above 100 Mbps (max 181 Mbps) are real measurements.

## Data
The raw CSVs and the generated tables are not stored in git - see [`data/README.md`](data/README.md) for how to regenerate or share them.
