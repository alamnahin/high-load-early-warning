"""
Step 2 - build model-ready tables (features + future targets + chronological split).

Output: data/model_ready/<DS>_model_ready.parquet  (+ .csv.gz)
        tables/feature_groups.json, tables/model_ready_summary.csv
"""
import json
import pandas as pd

from config import DATASETS, CLEAN_DIR, MODEL_DIR, TAB_DIR, HORIZONS_SEC, THRESHOLDS
from features import build, feature_groups, WARMUP

ID_COLS = ["dataset", "host", "host_group", "ts_unix", "time", "label", "cur_ge3", "cur_ge4", "idle_tail", "split"]

if __name__ == "__main__":
    rows, groups = [], None
    for ds in DATASETS:
        clean = pd.read_parquet(CLEAN_DIR / f"{ds}_clean.parquet")
        mr = build(clean)
        groups = feature_groups(mr)
        feats = [c for v in groups.values() for c in v]
        targets = [c for c in mr.columns if c.startswith("y_")]

        n0 = len(mr)
        max_h = max(HORIZONS_SEC)
        mr = mr.dropna(subset=feats + [f"y_cls_h{max_h}"])        # warm-up rows + last H rows per router
        mr = mr[ID_COLS + feats + targets].copy()
        mr["host"] = mr["host"].astype(str)
        mr["label"] = mr["label"].astype("int8")
        mr[targets] = mr[targets].astype("int8")
        mr.to_parquet(MODEL_DIR / f"{ds}_model_ready.parquet", index=False)
        mr.to_csv(MODEL_DIR / f"{ds}_model_ready.csv.gz", index=False, compression="gzip")

        for sp, g in mr.groupby("split"):
            r = {"dataset": ds, "split": sp, "rows": len(g)}
            for thr in THRESHOLDS:
                for H in (10, 30, 60):
                    r[f"pos_rate_ge{thr}_h{H}_%"] = round(g[f"y_ge{thr}_h{H}"].mean() * 100, 2)
            rows.append(r)
        print(f"{ds}: {n0} -> {len(mr)} rows, {len(feats)} features, {len(targets)} target columns")

    (TAB_DIR / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    pd.DataFrame(rows).to_csv(TAB_DIR / "model_ready_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
