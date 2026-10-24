"""Unit tests on small synthetic data - they run in CI without the real datasets."""
import numpy as np
import pandas as pd

from common import label_from_bw, run_lengths
from config import HORIZONS_SEC, SAMPLING_SEC
from features import build, feature_groups

T0 = 1_748_860_136


def make_clean(n=600, hosts=("r1", "r2", "r3"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for h in hosts:
        load = np.zeros(n)
        i = 0
        while i < n:
            dur = int(rng.integers(1, 12))
            load[i:i + dur] = rng.choice([0, 5, 30, 60, 80, 100, 130])
            i += dur
        tx, rx = load, load * rng.uniform(.4, .6, size=n)
        df = pd.DataFrame({"dataset": "T", "host": h, "ts_unix": T0 + SAMPLING_SEC * np.arange(n),
                           "bw_rx_3": rx, "bw_tx_3": tx})
        df["bw_max"] = df[["bw_rx_3", "bw_tx_3"]].max(axis=1)
        df["label"] = label_from_bw(df["bw_max"]).astype("int8")
        df["has_if6"] = df["any_imputed"] = df["n_imputed"] = df["idle_tail"] = 0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_label_bins():
    x = pd.Series([0, 9.9, 10, 39.9, 40, 69.9, 70, 89.9, 90, 150])
    assert label_from_bw(x).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_run_lengths():
    assert run_lengths(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1])).tolist() == [2, 1, 3]
    assert run_lengths(np.array([0, 0])).tolist() == []


def test_features_are_causal():
    """Features at t must not change when everything after t is removed."""
    clean = make_clean()
    cut_ts = T0 + SAMPLING_SEC * 300
    full = build(clean.copy(), drop_idle_tail=False)
    cut = build(clean[clean.ts_unix <= cut_ts].copy(), drop_idle_tail=False)
    feats = [c for v in feature_groups(full).values() for c in v]
    a = full[full.ts_unix <= cut_ts].sort_values(["host", "ts_unix"]).reset_index(drop=True)
    b = cut.sort_values(["host", "ts_unix"]).reset_index(drop=True)
    assert len(a) == len(b)
    for c in feats:
        assert np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True), f"{c} uses future data"


def test_targets_equal_future_label():
    clean = make_clean()
    out = build(clean.copy(), drop_idle_tail=False)
    lk = clean.set_index(["host", "ts_unix"])["label"]
    for H in HORIZONS_SEC:
        ok = out[f"y_cls_h{H}"].notna()
        key = pd.MultiIndex.from_arrays([out.loc[ok, "host"], out.loc[ok, "ts_unix"] + H])
        assert (lk.reindex(key).values == out.loc[ok, f"y_cls_h{H}"].values).all()
        assert (out.loc[ok, "y_ge4_h%d" % H] == (out.loc[ok, f"y_cls_h{H}"] >= 4)).all()


def test_split_has_purge_gap():
    out = build(make_clean(), drop_idle_tail=False)
    mx = max(HORIZONS_SEC)
    t = {s: out.loc[out.split == s, "ts_unix"] for s in ("train", "val", "test")}
    assert t["train"].max() + mx < t["val"].min() + 1
    assert t["val"].max() + mx < t["test"].min() + 1


def test_idle_tail_removed_but_targets_use_it():
    clean = make_clean()
    last = clean.ts_unix.max()
    clean.loc[clean.ts_unix > last - SAMPLING_SEC * 100, "idle_tail"] = 1
    out = build(clean, drop_idle_tail=True)
    assert out.ts_unix.max() <= last - SAMPLING_SEC * 100
    # rows right before the tail still get a defined target (computed on the full series)
    assert out.loc[out.ts_unix == out.ts_unix.max(), "y_cls_h60"].notna().all()
