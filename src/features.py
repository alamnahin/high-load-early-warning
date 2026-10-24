"""
Feature + target construction (importable, so the leakage test can re-use it).

Every feature at row (host, t) is computed ONLY from samples with time <= t
(current sample, lags, trailing windows) or from other routers at the SAME time t.
Targets look strictly forward: label at t+H of the same router.
"""
import numpy as np
import pandas as pd

from config import (SAMPLING_SEC, HORIZONS_SEC, THRESHOLDS, ROLL_STEPS, LAG_STEPS,
                    HOST_GROUPS, IFACES, SPLIT_FRACS, DROP_IDLE_TAIL)

SINCE_CAP = 360          # "steps since" is capped at 1 hour of 10 s samples
WARMUP = max(max(ROLL_STEPS), max(LAG_STEPS))


def _sum(df, prefix, ifaces=IFACES):
    cols = [f"{prefix}_{i}" for i in ifaces if f"{prefix}_{i}" in df.columns]
    return df[cols].sum(axis=1)


def _run_and_since(b: np.ndarray, cap=SINCE_CAP):
    n = len(b)
    run = np.zeros(n, dtype=np.int32)
    since = np.zeros(n, dtype=np.int32)
    r, s = 0, cap
    for k in range(n):
        if b[k]:
            r += 1
            s = 0
        else:
            r = 0
            s = min(s + 1, cap)
        run[k], since[k] = r, s
    return run, since


def add_current_aggregates(df):
    df["bw_rx_sum"] = _sum(df, "bw_rx")
    df["bw_tx_sum"] = _sum(df, "bw_tx")
    df["bw_total"] = df["bw_rx_sum"] + df["bw_tx_sum"]
    df["bw_rx_share"] = df["bw_rx_sum"] / (df["bw_total"] + 1e-6)
    df["eth_pkts_sum"] = _sum(df, "etherStatsPkts")
    df["eth_octets_sum"] = _sum(df, "etherStatsOctets")
    df["ucast_in_sum"] = _sum(df, "ifInUcastPkts")
    df["ucast_out_sum"] = _sum(df, "ifOutUcastPkts")
    df["nucast_in_sum"] = _sum(df, "ifInNUcastPkts")
    df["discards_in_sum"] = _sum(df, "ifInDiscards")
    df["discards_out_sum"] = _sum(df, "ifOutDiscards")
    df["discards_sum"] = df["discards_in_sum"] + df["discards_out_sum"]
    # error-like RMON counters (Jabbers is excluded: in these files it behaves like a packet counter)
    err = ["etherStatsCRCAlignErrors", "etherStatsUndersizePkts", "etherStatsOversizePkts",
           "etherStatsFragments", "etherStatsCollisions"]
    df["rmon_err_sum"] = sum(_sum(df, e) for e in err)
    df["cur_ge3"] = (df["label"] >= 3).astype("int8")
    df["cur_ge4"] = (df["label"] >= 4).astype("int8")
    return df


def add_temporal_features(df):
    """Causal lags / trailing windows / burst-dynamics, computed per router."""
    g = df.groupby("host", observed=True, sort=False)
    out = {}
    for k in LAG_STEPS:
        out[f"bw_max_lag{k}"] = g["bw_max"].shift(k)
    out["bw_max_diff1"] = df["bw_max"] - out["bw_max_lag1"]
    out["bw_max_diff3"] = df["bw_max"] - out["bw_max_lag3"]
    for w in ROLL_STEPS:
        s = w * SAMPLING_SEC
        for c in ("bw_max", "bw_total"):
            r = g[c].rolling(w, min_periods=w)
            out[f"{c}_mean{s}s"] = r.mean().reset_index(level=0, drop=True)
            out[f"{c}_max{s}s"] = r.max().reset_index(level=0, drop=True)
            out[f"{c}_std{s}s"] = r.std().reset_index(level=0, drop=True)
        for c in ("eth_pkts_sum", "discards_sum"):
            out[f"{c}_mean{s}s"] = g[c].rolling(w, min_periods=w).mean().reset_index(level=0, drop=True)
        out[f"n_ge4_last{s}s"] = g["cur_ge4"].rolling(w, min_periods=w).sum().reset_index(level=0, drop=True)
    df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    # time-in-state features for LABEL>=3 and >=4
    run3 = np.zeros(len(df), dtype=np.int32); since3 = run3.copy()
    run4 = run3.copy(); since4 = run3.copy()
    for _, idx in df.groupby("host", observed=True, sort=False).indices.items():
        run3[idx], since3[idx] = _run_and_since(df["cur_ge3"].values[idx])
        run4[idx], since4[idx] = _run_and_since(df["cur_ge4"].values[idx])
    df["run_len_ge3"], df["steps_since_ge3"] = run3, since3
    df["run_len_ge4"], df["steps_since_ge4"] = run4, since4
    return df


def add_network_context(df):
    """Load of the other routers at the SAME timestamp (available at prediction time)."""
    p = df.pivot_table(index="ts_unix", columns="host", values="bw_max", observed=True)
    pf = df.pivot_table(index="ts_unix", columns="host", values="cur_ge4", observed=True)
    ctx = pd.DataFrame(index=p.index)
    for name, hosts in HOST_GROUPS.items():
        hosts = [h for h in hosts if h in p.columns]
        if name != "idle" and hosts:
            ctx[f"ctx_{name}_bw_max_mean"] = p[hosts].mean(axis=1)
    active = [h for h in p.columns if h not in HOST_GROUPS["idle"]]
    ctx["ctx_net_bw_max_mean"] = p[active].mean(axis=1)
    ctx["ctx_net_bw_max_max"] = p[active].max(axis=1)
    ctx["ctx_net_n_ge4"] = pf[active].sum(axis=1)
    return df.merge(ctx, left_on="ts_unix", right_index=True, how="left")


def add_targets(df):
    g = df.groupby("host", observed=True, sort=False)
    for H in HORIZONS_SEC:
        k = H // SAMPLING_SEC
        fut_label = g["label"].shift(-k)
        ok = (g["ts_unix"].shift(-k) - df["ts_unix"]) == H          # guards against gaps
        fut_label = fut_label.where(ok)
        df[f"y_cls_h{H}"] = fut_label
        for thr in THRESHOLDS:
            df[f"y_ge{thr}_h{H}"] = (fut_label >= thr).astype("float").where(fut_label.notna())
    return df


def add_split(df, purge_sec=max(HORIZONS_SEC)):
    """Chronological train/val/test inside ONE dataset, with a purge band so that
    no training target window reaches into the validation/test period."""
    ts_u = np.sort(df["ts_unix"].unique())
    t1 = ts_u[int(len(ts_u) * SPLIT_FRACS[0])]
    t2 = ts_u[int(len(ts_u) * (SPLIT_FRACS[0] + SPLIT_FRACS[1]))]
    t = df["ts_unix"]
    split = np.where(t < t1 - purge_sec, "train",
            np.where(t < t1, "purge",
            np.where(t < t2 - purge_sec, "val",
            np.where(t < t2, "purge", "test"))))
    df["split"] = split
    return df


def build(df, drop_idle_tail=DROP_IDLE_TAIL):
    """clean dataframe (one dataset) -> model-ready dataframe.

    Features/targets are computed on the FULL series first (so the last active rows still get
    correct targets); the idle tail is removed afterwards and the chronological split is defined
    on the remaining active period only."""
    df = df.sort_values(["host", "ts_unix"]).reset_index(drop=True)
    df = add_current_aggregates(df)
    df = add_temporal_features(df)
    df = add_network_context(df)
    df = add_targets(df)
    if drop_idle_tail and "idle_tail" in df.columns:
        df = df[df["idle_tail"] == 0].reset_index(drop=True)
    df = add_split(df)
    df["host_group"] = df["host"].astype(str).map(
        {h: grp for grp, hs in HOST_GROUPS.items() for h in hs}).fillna("other")
    return df


def feature_groups(df):
    base_bw = [c for c in df.columns if c.startswith("bw_rx_") or c.startswith("bw_tx_")]
    groups = {
        "current_bandwidth": base_bw + ["bw_max", "bw_rx_sum", "bw_tx_sum", "bw_total", "bw_rx_share"],
        "current_packets_errors": ["eth_pkts_sum", "eth_octets_sum", "ucast_in_sum", "ucast_out_sum",
                                   "nucast_in_sum", "discards_in_sum", "discards_out_sum",
                                   "discards_sum", "rmon_err_sum"],
        "current_tcp_udp": [c for c in ("tcpOutRsts", "tcpInSegs", "tcpOutSegs", "tcpActiveOpens",
                                        "udpInDatagrams", "udpOutDatagrams") if c in df.columns],
        "lags_trend": [c for c in df.columns if "_lag" in c or "_diff" in c],
        "trailing_windows": [c for c in df.columns if any(t in c for t in ("_mean", "_max", "_std", "n_ge4_last"))
                             and not c.startswith("ctx_") and c not in ("bw_max",)],
        "burst_dynamics": ["run_len_ge3", "run_len_ge4", "steps_since_ge3", "steps_since_ge4"],
        "network_context": [c for c in df.columns if c.startswith("ctx_")],
        "meta_flags": ["has_if6", "any_imputed", "n_imputed"],
    }
    # bw_max appears in two patterns above; keep unique ordering
    seen = set()
    for k, v in groups.items():
        groups[k] = [c for c in v if not (c in seen or seen.add(c))]
    return groups
