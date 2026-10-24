# Data dictionary

## Raw files (104 columns, one row = one router at one 10 s polling instant)
* `Host Name` (r1..r9), `Timestamp` (Unix seconds, UTC), `LABEL` (0-4), `Nº de interfacesdetectadas`, `Interfaces` (constant 6).
* SNMP-style counters with an interface suffix `.3 .4 .5 .6` (`ifInOctets.3`, `etherStatsPkts.4`, ...). Values are **per-interval deltas**, not cumulative counters.
* `bw_rx.i` / `bw_tx.i` - bandwidth in Mbps: `floor(octets x 8 / 10 s / 1e6, 1 decimal)`.
* `tcp*`, `udp*` - host-level counters (low information in these files).
* **`LABEL` = bin of `max(bw_rx.i, bw_tx.i)` over the router's interfaces:**

| LABEL | max interface load |
|---|---|
| 0 | < 10 Mbps |
| 1 | 10 - 40 |
| 2 | 40 - 70 |
| 3 | 70 - 90 |
| 4 | >= 90 |

Interface 6 exists only on r6 and r8. Some counter names do not match their behavior (e.g. `hostOutPkts` ~ bytes, `etherStatsJabbers` ~ packets) - see the EDA report.

## Clean tables (`data/clean/*_clean.parquet`)
Same information as the raw file with: columns renamed (`bw_rx.3` -> `bw_rx_3`, `Host Name` -> `host`, `Timestamp` -> `ts_unix`, `LABEL` -> `label`, `Nº de interfacesdetectadas` -> `n_interfaces_detected`), 15 all-constant columns dropped, plus

| Column | Meaning |
|---|---|
| `dataset` | D1 / D2 / D3 |
| `time` | UTC datetime of `ts_unix` |
| `bw_max` | max over observed interface bandwidths (before imputation) - exactly the quantity `label` bins |
| `has_if6` | 1 if the router has interface 6 (r6, r8); otherwise the `*_6` columns are structural zeros |
| `any_imputed`, `n_imputed` | row had >= 1 forward-filled value / how many |
| `row_inserted` | row created to fill a missing 10 s slot (none in these files) |
| `idle_tail` | 1 for the final period in which no router carries >= 10 Mbps (D1 16.5 h, D2 7.6 h) |

## Model-ready tables (`data/model_ready/*_model_ready.*`)

**Identifiers / state**

| Column | Meaning |
|---|---|
| `dataset`, `host`, `ts_unix`, `time` | as above |
| `host_group` | `core` (r1, r5), `grpA` (r2, r3, r4), `grpB` (r6, r8, r9), `idle` (r7); inferred from correlations, not documented topology |
| `label`, `cur_ge3`, `cur_ge4` | current LABEL and indicators LABEL >= 3 / >= 4 |
| `split` | `train` / `val` / `test` (70/15/15 chronological per dataset) or `purge` (60 s band between splits - exclude) |
| `idle_tail` | 0 for all rows unless `DROP_IDLE_TAIL=False` |

**Targets** (look strictly forward on the same router; H in {10, 20, 30, 60} s)

| Column | Meaning |
|---|---|
| `y_cls_h{H}` | LABEL at t + H |
| `y_ge3_h{H}`, `y_ge4_h{H}` | 1 if LABEL at t + H >= 3 / >= 4 |

**Features (63, all computed from samples at or before t, or from other routers at the same t)**

| Group | Columns |
|---|---|
| current bandwidth | `bw_rx_3..6`, `bw_tx_3..5`, `bw_max`, `bw_rx_sum`, `bw_tx_sum`, `bw_total`, `bw_rx_share` |
| packets / errors (sums over interfaces) | `eth_pkts_sum`, `eth_octets_sum`, `ucast_in_sum`, `ucast_out_sum`, `nucast_in_sum`, `discards_in_sum`, `discards_out_sum`, `discards_sum`, `rmon_err_sum` |
| TCP / UDP | `tcpOutRsts`, `tcpInSegs`, `tcpOutSegs`, `tcpActiveOpens`, `udpInDatagrams`, `udpOutDatagrams` |
| lags / trend | `bw_max_lag1..3`, `bw_max_diff1`, `bw_max_diff3` |
| trailing windows (30 s, 60 s) | `{bw_max,bw_total}_{mean,max,std}{30,60}s`, `{eth_pkts_sum,discards_sum}_mean{30,60}s`, `n_ge4_last{30,60}s` |
| burst dynamics | `run_len_ge3`, `run_len_ge4` (current consecutive run), `steps_since_ge3`, `steps_since_ge4` (capped at 360 = 1 h) |
| network context (same t) | `ctx_core_bw_max_mean`, `ctx_grpA_bw_max_mean`, `ctx_grpB_bw_max_mean`, `ctx_net_bw_max_mean`, `ctx_net_bw_max_max`, `ctx_net_n_ge4` |
| data quality | `has_if6`, `any_imputed`, `n_imputed` |

The first 6 rows of every router (rolling-window warm-up) and the last 6 (no 60 s future) are dropped. Nothing is scaled; fit scalers on the `train` rows only (`src/modeling_utils.py`).
