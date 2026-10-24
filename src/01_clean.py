"""
Step 1 - load, validate and clean the three raw CSV files.

What it does (per dataset)
  1. Renames awkward columns (e.g. 'Nº de interfacesdetectadas', 'bw_rx.3' -> 'bw_rx_3').
  2. Validates timestamps (numeric, plausible range, on the 10 s grid), removes
     duplicates, sorts by (host, time) and fills any missing grid rows.
  3. Drops columns that are constant in ALL three datasets (carry no information).
  4. Missing values
       - interface 6 does not exist on 7 of 9 routers  -> structural zeros + has_if6 flag
       - sporadic single-sample SNMP polling gaps       -> bw rebuilt from octet
         counters where possible, then causal forward-fill (<= 20 s), flagged.
  5. Verifies LABEL == bin(max interface bandwidth) and logs the result.

Outputs: data/clean/<DS>_clean.parquet and tables/cleaning_log.json
"""
import json
import numpy as np
import pandas as pd

from config import (DATASETS, CLEAN_DIR, TAB_DIR, SAMPLING_SEC, VALID_TS_RANGE,
                    IFACES, MAX_IMPUTE_GAP, IDLE_TAIL_MIN_BW, raw_path)
from common import bw_columns, label_from_bw

RENAMES = {"Host Name": "host", "Timestamp": "ts_unix", "LABEL": "label",
           "Nº de interfacesdetectadas": "n_interfaces_detected"}


def rename_columns(df):
    df = df.rename(columns=RENAMES)
    df.columns = [c.replace(".", "_") for c in df.columns]   # 'bw_rx.3' -> 'bw_rx_3'
    return df


def validate_timestamps(df, log):
    ts = pd.to_numeric(df["ts_unix"], errors="coerce")
    lo, hi = [pd.Timestamp(x, tz="UTC").timestamp() for x in VALID_TS_RANGE]
    bad = ts.isna() | (ts < lo) | (ts > hi)
    log["ts_invalid_dropped"] = int(bad.sum())
    df = df.loc[~bad].copy()
    df["ts_unix"] = ts[~bad].astype("int64")

    # snap off-grid timestamps to the modal 10 s phase
    phase = int((df["ts_unix"] % SAMPLING_SEC).mode().iloc[0])
    off = (df["ts_unix"] % SAMPLING_SEC) != phase
    log["ts_off_grid_snapped"] = int(off.sum())
    if off.any():
        df.loc[off, "ts_unix"] = ((df.loc[off, "ts_unix"] - phase) / SAMPLING_SEC).round().astype("int64") \
                                 * SAMPLING_SEC + phase

    n0 = len(df)
    df = df.drop_duplicates(["host", "ts_unix"], keep="first")
    log["duplicate_host_ts_dropped"] = int(n0 - len(df))
    return df.sort_values(["host", "ts_unix"]).reset_index(drop=True)


def regularize_grid(df, log):
    """Insert rows for missing 10 s slots so shifts == time offsets (no-op if no gaps)."""
    gaps = df.groupby("host")["ts_unix"].diff().dropna()
    log["grid_gaps_gt_10s"] = int((gaps > SAMPLING_SEC).sum())
    log["grid_max_gap_s"] = int(gaps.max()) if len(gaps) else 0
    log["grid_diffs_not_10s"] = int((gaps != SAMPLING_SEC).sum())
    df["row_inserted"] = 0
    if log["grid_gaps_gt_10s"] == 0:
        return df
    parts = []
    for h, g in df.groupby("host", sort=False):
        full = pd.RangeIndex(g.ts_unix.min(), g.ts_unix.max() + 1, SAMPLING_SEC)
        g = g.set_index("ts_unix").reindex(full)
        g["host"] = h
        g["row_inserted"] = g["row_inserted"].isna().astype(int)
        g.index.name = "ts_unix"
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def rebuild_bw_from_octets(df, log):
    """bw = floor(octets*8 / 10 s / 1e6, 1 decimal) - verified exact on all non-missing rows."""
    n_fixed = 0
    for i in IFACES:
        for d, o in (("rx", "ifInOctets"), ("tx", "ifOutOctets")):
            b, oc = f"bw_{d}_{i}", f"{o}_{i}"
            if b not in df.columns or oc not in df.columns:   # e.g. bw_tx_6 is constant -> dropped
                continue
            m = df[b].isna() & df[oc].notna()
            df.loc[m, b] = np.floor(df.loc[m, oc] * 8 / (SAMPLING_SEC * 1e6) * 10) / 10
            n_fixed += int(m.sum())
    log["bw_values_rebuilt_from_octets"] = n_fixed
    return df


def clean_one(ds_id, const_cols_union):
    log = {}
    raw = pd.read_csv(raw_path(ds_id), low_memory=False)
    log["rows_raw"], log["cols_raw"] = int(raw.shape[0]), int(raw.shape[1])
    df = rename_columns(raw)
    df = validate_timestamps(df, log)
    df = regularize_grid(df, log)

    df = df.drop(columns=[c for c in const_cols_union if c in df.columns])
    log["constant_cols_dropped"] = sorted(const_cols_union)

    # ---- interface 6: structural absence
    has6 = df.groupby("host")["bw_rx_6"].transform(lambda s: s.notna().mean()) > 0.5
    df["has_if6"] = has6.astype("int8")
    if6_cols = [c for c in df.columns if c.endswith("_6")]
    df.loc[~has6, if6_cols] = 0.0
    log["hosts_with_if6"] = sorted(df.loc[has6, "host"].unique().tolist())

    # ---- sporadic missing values
    df = rebuild_bw_from_octets(df, log)
    num = [c for c in df.columns if c not in ("host", "ts_unix", "label", "row_inserted", "has_if6", "bw_max")]
    na_before = df[num].isna()
    # bw_max follows the dataset's own LABEL definition: max over the interfaces that were
    # actually observed in that sample (computed BEFORE forward-fill).
    df["bw_max"] = df[bw_columns(df)].max(axis=1)
    log["missing_cells_before_imputation"] = int(na_before.values.sum())
    log["rows_with_missing_before_imputation"] = int(na_before.any(axis=1).sum())
    df[num] = df.groupby("host")[num].ffill(limit=MAX_IMPUTE_GAP)
    still = df[num].isna().any(axis=1)
    log["rows_dropped_unfillable"] = int(still.sum())
    imputed_mask = na_before & df[num].notna()
    df["n_imputed"] = imputed_mask.sum(axis=1).astype("int16")
    df["any_imputed"] = (df["n_imputed"] > 0).astype("int8")
    df = df.loc[~still & df["label"].notna()].reset_index(drop=True)

    # ---- label consistency
    agree = (label_from_bw(df["bw_max"]) == df["label"])
    log["label_equals_bin_of_bw_max_pct"] = round(float(agree.mean() * 100), 4)
    log["label_mismatch_rows_imputed_only"] = int((~agree & (df["any_imputed"] == 1)).sum())
    log["label_mismatch_rows_total"] = int((~agree).sum())

    # ---- idle tail: everything after the last time ANY router carried >= IDLE_TAIL_MIN_BW Mbps
    net_max = df.groupby("ts_unix")["bw_max"].max()
    active_ts = net_max[net_max >= IDLE_TAIL_MIN_BW].index
    last_active = int(active_ts.max()) if len(active_ts) else int(df["ts_unix"].max())
    df["idle_tail"] = (df["ts_unix"] > last_active).astype("int8")
    log["idle_tail_rows"] = int(df["idle_tail"].sum())
    log["idle_tail_pct"] = round(float(df["idle_tail"].mean() * 100), 2)
    log["idle_tail_hours"] = round((int(df["ts_unix"].max()) - last_active) / 3600, 2)
    log["last_active_time"] = str(pd.to_datetime(last_active, unit="s", utc=True))

    # ---- time & dtypes
    df.insert(0, "dataset", ds_id)
    df.insert(3, "time", pd.to_datetime(df["ts_unix"], unit="s", utc=True))
    df["label"] = df["label"].astype("int8")
    df["host"] = df["host"].astype("category")
    log["rows_clean"], log["cols_clean"] = int(df.shape[0]), int(df.shape[1])
    log["time_start"], log["time_end"] = str(df["time"].min()), str(df["time"].max())
    return df, log


def constant_columns_across_all():
    """Columns whose value never changes in ANY of the three datasets."""
    seen = None
    for ds in DATASETS:
        d = rename_columns(pd.read_csv(raw_path(ds), low_memory=False))
        nun = d.drop(columns=["host", "ts_unix"]).nunique(dropna=True)
        const = set(nun[nun <= 1].index)
        seen = const if seen is None else seen & const
    return sorted(c for c in seen if c != "label")


if __name__ == "__main__":
    const_cols = constant_columns_across_all()
    print("Constant in all datasets -> dropped:", const_cols)
    full_log = {}
    for ds in DATASETS:
        df, log = clean_one(ds, const_cols)
        df.to_parquet(CLEAN_DIR / f"{ds}_clean.parquet", index=False)
        full_log[ds] = log
        print(ds, json.dumps({k: v for k, v in log.items() if k != "constant_cols_dropped"}, indent=1))
    (TAB_DIR / "cleaning_log.json").write_text(json.dumps(full_log, indent=2))
