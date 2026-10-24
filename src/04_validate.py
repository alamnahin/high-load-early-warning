"""
Step 4 - automated sanity / leakage checks on the produced tables.  Raises AssertionError on failure.

  1. Causality (truncation) test: rebuild features on data cut at time T; every feature value for
     rows with t <= T must be IDENTICAL to the full-data run  ->  no feature can see the future.
  2. Target correctness: y_* equals the label of the same router exactly H seconds later.
  3. Split integrity: train < val < test in time, separated by >= max horizon (purge band).
  4. No NaN / inf in features; no feature column is a target.
  5. Timestamps: strictly 10 s spacing per router in the clean data.
"""
import numpy as np
import pandas as pd

from config import DATASETS, CLEAN_DIR, MODEL_DIR, HORIZONS_SEC, SAMPLING_SEC, THRESHOLDS
from features import build, feature_groups

ok = lambda msg: print("  [PASS]", msg)


def check_causality(ds):
    clean = pd.read_parquet(CLEAN_DIR / f"{ds}_clean.parquet")
    T = clean["ts_unix"].quantile(0.5)
    T = int(T - T % SAMPLING_SEC + (clean["ts_unix"].min() % SAMPLING_SEC))
    full = build(clean.copy(), drop_idle_tail=False)
    cut = build(clean[clean["ts_unix"] <= T].copy(), drop_idle_tail=False)
    feats = [c for v in feature_groups(full).values() for c in v]
    a = full[full["ts_unix"] <= T].sort_values(["host", "ts_unix"]).reset_index(drop=True)
    b = cut.sort_values(["host", "ts_unix"]).reset_index(drop=True)
    assert len(a) == len(b), "row count differs after truncation"
    diff = {c: float(np.nanmax(np.abs(a[c].astype(float).values - b[c].astype(float).values)))
            for c in feats if not np.allclose(a[c].astype(float).values, b[c].astype(float).values,
                                              equal_nan=True, atol=1e-9)}
    assert not diff, f"features depend on future data: {diff}"
    ok(f"{ds}: {len(feats)} features identical when the future is removed (causal)")


def check_targets(ds, mr):
    clean = pd.read_parquet(CLEAN_DIR / f"{ds}_clean.parquet", columns=["host", "ts_unix", "label"])
    clean["host"] = clean["host"].astype(str)
    lk = clean.set_index(["host", "ts_unix"])["label"]
    s = mr.sample(5000, random_state=0)
    for H in HORIZONS_SEC:
        key = pd.MultiIndex.from_arrays([s["host"], s["ts_unix"] + H])
        fut = lk.reindex(key).values
        assert (fut == s[f"y_cls_h{H}"].values).all(), f"y_cls_h{H} mismatch"
        for thr in THRESHOLDS:
            assert ((fut >= thr).astype(int) == s[f"y_ge{thr}_h{H}"].values).all(), f"y_ge{thr}_h{H} mismatch"
    ok(f"{ds}: targets equal the router's label exactly H s later (H={HORIZONS_SEC})")


def check_split(ds, mr):
    mx = max(HORIZONS_SEC)
    t = {s: mr.loc[mr["split"] == s, "ts_unix"] for s in ("train", "val", "test")}
    assert t["train"].max() + mx < t["val"].min() + 1, "train/val too close"
    assert t["val"].max() + mx < t["test"].min() + 1, "val/test too close"
    ok(f"{ds}: train < val < test with >= {mx}s purge gap "
       f"(rows {len(t['train'])}/{len(t['val'])}/{len(t['test'])})")


def check_features(ds, mr):
    feats = [c for v in feature_groups(mr).values() for c in v]
    X = mr[feats].astype(float)
    assert not X.isna().any().any(), "NaN in features"
    assert np.isfinite(X.values).all(), "inf in features"
    assert not any(c.startswith("y_") for c in feats), "target in features"
    ok(f"{ds}: no NaN/inf in {len(feats)} features, no target columns among them")


def check_grid(ds):
    c = pd.read_parquet(CLEAN_DIR / f"{ds}_clean.parquet", columns=["host", "ts_unix"])
    d = c.sort_values(["host", "ts_unix"]).groupby("host", observed=True)["ts_unix"].diff().dropna()
    assert (d == SAMPLING_SEC).all(), "irregular sampling in clean data"
    ok(f"{ds}: every router sampled exactly every {SAMPLING_SEC}s")


if __name__ == "__main__":
    for ds in DATASETS:
        print(ds)
        mr = pd.read_parquet(MODEL_DIR / f"{ds}_model_ready.parquet")
        check_grid(ds)
        check_features(ds, mr)
        check_targets(ds, mr)
        check_split(ds, mr)
        check_causality(ds)
    print("\nALL CHECKS PASSED")
