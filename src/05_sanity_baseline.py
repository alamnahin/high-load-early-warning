"""
Step 5 (optional) - quick sanity baseline, NOT the final experiments.

Purpose: confirm the model-ready tables are usable and show how much headroom there is above the
trivial 'persistence' predictor ("future state = current state").  One gradient-boosting model per
(training dataset, horizon), target LABEL>=4, router r7 excluded from training and evaluation.
Scaling is unnecessary for tree models; for Logistic Regression fit the scaler on the TRAIN split only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

from config import DATASETS, MODEL_DIR, TAB_DIR, SEED
from features import feature_groups

THR, HORIZONS = 4, [10, 30, 60]


def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    fpr = ((p == 1) & (y == 0)).sum() / max(((y == 0)).sum(), 1)
    return pr, rc, f1, fpr


if __name__ == "__main__":
    data = {ds: pd.read_parquet(MODEL_DIR / f"{ds}_model_ready.parquet") for ds in DATASETS}
    data = {ds: d[d.host != "r7"] for ds, d in data.items()}
    feats = [c for v in feature_groups(data["D1"]).values() for c in v]
    rows = []
    for H in HORIZONS:
        y = f"y_ge{THR}_h{H}"
        for tr in DATASETS:
            d = data[tr]
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=.1, max_leaf_nodes=31,
                                                 class_weight="balanced", random_state=SEED)
            clf.fit(d.loc[d.split == "train", feats], d.loc[d.split == "train", y])
            for te in DATASETS:
                t = data[te]
                t = t[t.split == "test"] if te == tr else t[t.split != "purge"]     # other dataset: all rows
                p = clf.predict(t[feats])
                pr, rc, f1, fpr = scores(t[y].values, p)
                pp = scores(t[y].values, t[f"cur_ge{THR}"].values)
                rows.append({"H_s": H, "train": tr, "test": te, "setting": "same-dataset (test split)" if tr == te else "cross-dataset",
                             "precision": round(pr, 3), "recall": round(rc, 3), "F1": round(f1, 3), "FPR": round(fpr, 4),
                             "persistence_F1": round(pp[2], 3), "prevalence_%": round(t[y].mean() * 100, 2)})
                print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "sanity_baseline_ge4.csv", index=False)
