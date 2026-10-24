"""
06_random_forest_baseline.py — Random Forest baseline for preprocessing comparison.

Trains a Random Forest on each dataset's train split and evaluates:
  - Same-dataset test split
  - Cross-dataset (train on one, test on another)

Horizons : 10, 30, 60 seconds
Threshold : LABEL >= 4

HOW TO RUN
----------
    python src/06_random_forest_baseline.py

Output:
    reports/tables/rf_baseline_results.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

# ── make sure src/ is on the path ─────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import MODEL_DIR, TAB_DIR, SEED
from modeling_utils import load_split, load_full, feature_columns, load_model_ready

# ── settings ──────────────────────────────────────────────────────────────
DATASETS  = ["D1", "D2", "D3"]
HORIZONS  = [10, 30, 60]
THR       = 4
IDLE_RTR  = "r7"

RF_PARAMS = dict(
    n_estimators    = 300,
    max_depth       = 20,
    min_samples_leaf= 10,
    class_weight    = "balanced",
    n_jobs          = -1,
    random_state    = SEED,
)

# ── helpers ───────────────────────────────────────────────────────────────
def scores(y_true, y_pred):
    p   = round(float(precision_score(y_true, y_pred, zero_division=0)), 3)
    r   = round(float(recall_score   (y_true, y_pred, zero_division=0)), 3)
    f1  = round(float(f1_score       (y_true, y_pred, zero_division=0)), 3)
    fpr = round(float(((y_pred == 1) & (y_true == 0)).sum()
                      / max((y_true == 0).sum(), 1)), 4)
    try:
        auc = round(float(roc_auc_score(y_true, y_pred)), 3)
    except Exception:
        auc = float("nan")
    return p, r, f1, fpr, auc

def persistence_f1(df, target):
    """Trivial baseline: predict current state persists into the future."""
    cur_col = f"cur_ge{THR}"
    if cur_col not in df.columns:
        return float("nan")
    return round(float(f1_score(df[target], df[cur_col], zero_division=0)), 3)

# ── main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading model-ready tables...")
    all_data = {}
    for ds in DATASETS:
        d = load_model_ready(ds, exclude_idle_router=True)
        d = d[d.split != "purge"]
        all_data[ds] = d
        print(f"  {ds}: {len(d):,} rows, {d.host.nunique()} routers")

    feats = feature_columns(all_data["D1"])
    feats = [f for f in feats if f in all_data["D1"].columns]
    print(f"\nFeatures used: {len(feats)}")

    rows = []

    for H in HORIZONS:
        target = f"y_ge{THR}_h{H}"
        print(f"\n{'='*60}")
        print(f"  Horizon: {H}s   Target: {target}")
        print(f"{'='*60}")

        for tr_ds in DATASETS:
            tr_df   = all_data[tr_ds]
            X_train = tr_df.loc[tr_df.split == "train", feats]
            y_train = tr_df.loc[tr_df.split == "train", target]

            if target not in tr_df.columns:
                print(f"  [{tr_ds}] target missing — skipping"); continue

            print(f"\n  Training on {tr_ds} ({len(X_train):,} rows, "
                  f"prevalence {y_train.mean()*100:.1f}%) ...")

            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(X_train, y_train)

            # same-dataset test split
            te_df  = tr_df[tr_df.split == "test"]
            y_pred = clf.predict(te_df[feats])
            p, r, f1, fpr, auc = scores(te_df[target].values, y_pred)
            pers   = persistence_f1(te_df, target)
            prev   = round(float(te_df[target].mean() * 100), 2)

            row = {
                "H_s"           : H,
                "train_ds"      : tr_ds,
                "test_ds"       : tr_ds,
                "setting"       : "same-dataset",
                "prevalence_%"  : prev,
                "RF_precision"  : p,
                "RF_recall"     : r,
                "RF_F1"         : f1,
                "RF_FPR"        : fpr,
                "RF_ROC_AUC"    : auc,
                "Persistence_F1": pers,
                "F1_gain"       : round(f1 - (pers if not np.isnan(pers) else 0), 3),
                "n_features"    : len(feats),
            }
            rows.append(row)
            print(f"    [{tr_ds}->{tr_ds}] RF F1={f1:.3f}  "
                  f"Prec={p:.3f}  Rec={r:.3f}  "
                  f"FPR={fpr:.4f}  AUC={auc:.3f}  "
                  f"Persistence F1={pers:.3f}  gain={f1-pers:+.3f}")

            # cross-dataset tests
            for te_ds in DATASETS:
                if te_ds == tr_ds:
                    continue
                te_df  = all_data[te_ds]           # all non-purge rows
                if target not in te_df.columns:
                    continue
                y_pred = clf.predict(te_df[feats])
                p, r, f1, fpr, auc = scores(te_df[target].values, y_pred)
                pers = persistence_f1(te_df, target)
                prev = round(float(te_df[target].mean() * 100), 2)

                row = {
                    "H_s"           : H,
                    "train_ds"      : tr_ds,
                    "test_ds"       : te_ds,
                    "setting"       : "cross-dataset",
                    "prevalence_%"  : prev,
                    "RF_precision"  : p,
                    "RF_recall"     : r,
                    "RF_F1"         : f1,
                    "RF_FPR"        : fpr,
                    "RF_ROC_AUC"    : auc,
                    "Persistence_F1": pers,
                    "F1_gain"       : round(f1 - (pers if not np.isnan(pers) else 0), 3),
                    "n_features"    : len(feats),
                }
                rows.append(row)
                print(f"    [{tr_ds}->{te_ds}] RF F1={f1:.3f}  "
                      f"Prec={p:.3f}  Rec={r:.3f}  "
                      f"FPR={fpr:.4f}  AUC={auc:.3f}  "
                      f"Persistence F1={pers:.3f}  gain={f1-pers:+.3f}")

    # ── save results ──────────────────────────────────────────────────────
    out = pd.DataFrame(rows)
    out_path = TAB_DIR / "rf_baseline_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # ── summary table ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY — RF F1 by horizon and dataset (same-dataset test)")
    print("="*60)
    same = out[out.setting == "same-dataset"].copy()
    pivot = same.pivot_table(
        index="train_ds", columns="H_s", values="RF_F1", aggfunc="mean")
    print(pivot.round(3).to_string())

    print("\nSUMMARY — Mean F1 gain over persistence (same-dataset)")
    pivot2 = same.pivot_table(
        index="train_ds", columns="H_s", values="F1_gain", aggfunc="mean")
    print(pivot2.round(3).to_string())

    print(f"\nShare  reports/tables/rf_baseline_results.csv  with our team.")
