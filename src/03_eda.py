"""
Step 3 - exploratory data analysis: figures (outputs/figures) + tables (outputs/tables).

Needs the outputs of steps 1 and 2.  Everything that describes *load* uses the active
period only (idle tail removed) and, where stated, excludes router r7 (always idle).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import (DATASETS, DATASET_NOTES, CLEAN_DIR, MODEL_DIR, FIG_DIR, TAB_DIR, HORIZONS_SEC,
                    SAMPLING_SEC, LABEL_NAMES, HOST_GROUPS, SEED)
from common import run_lengths

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COL = {"D1": "#1f77b4", "D2": "#d62728", "D3": "#2ca02c"}
HOST_ORDER = HOST_GROUPS["core"] + HOST_GROUPS["grpA"] + HOST_GROUPS["grpB"]     # r7 (idle) excluded
rng = np.random.default_rng(SEED)
S = {}                                                                          # summary numbers


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


clean = {ds: pd.read_parquet(CLEAN_DIR / f"{ds}_clean.parquet") for ds in DATASETS}
for ds in clean:
    clean[ds]["host"] = clean[ds]["host"].astype(str)
active = {ds: d[d["idle_tail"] == 0] for ds, d in clean.items()}
model = {ds: pd.read_parquet(MODEL_DIR / f"{ds}_model_ready.parquet") for ds in DATASETS}

# ------------------------------------------------------------------ overview table
rows = []
for ds, d in clean.items():
    a = active[ds]
    rows.append({"dataset": ds, "rows": len(d), "routers": d.host.nunique(),
                 "start": d.time.min().strftime("%Y-%m-%d %H:%M"), "end": d.time.max().strftime("%Y-%m-%d %H:%M"),
                 "duration_h": round((d.ts_unix.max() - d.ts_unix.min()) / 3600, 1),
                 "idle_tail_h": round(float(d.ts_unix.max() - a.ts_unix.max()) / 3600, 2),
                 "idle_tail_%": round(d.idle_tail.mean() * 100, 1),
                 "max_bw_mbps": round(d.bw_max.max(), 1)})
overview = pd.DataFrame(rows)
overview.to_csv(TAB_DIR / "dataset_overview.csv", index=False)
S["overview"] = overview.to_dict("records")
print(overview.to_string(index=False))

# ------------------------------------------------------------------ Fig 1 label distribution
lab = {}
for ds, a in active.items():
    lab[ds] = {"all_routers": a.label.value_counts(normalize=True).sort_index() * 100,
               "excl_r7": a[a.host != "r7"].label.value_counts(normalize=True).sort_index() * 100}
lc = pd.concat({ds: pd.DataFrame(v) for ds, v in lab.items()}, axis=1).round(2)
lc.to_csv(TAB_DIR / "label_distribution_pct.csv")
S["label_pct_excl_r7"] = {ds: lab[ds]["excl_r7"].round(2).to_dict() for ds in DATASETS}
S["prevalence_excl_r7"] = {ds: {"ge3_%": round(float((a[a.host != "r7"].label >= 3).mean() * 100), 2),
                                "ge4_%": round(float((a[a.host != "r7"].label >= 4).mean() * 100), 2)}
                           for ds, a in active.items()}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, key, ttl in zip(axs, ["all_routers", "excl_r7"], ["All 9 routers", "Excluding always-idle router r7"]):
    w = 0.26
    for j, ds in enumerate(DATASETS):
        v = lab[ds][key].reindex(range(5)).fillna(0)
        ax.bar(np.arange(5) + (j - 1) * w, v.values, w, label=ds, color=COL[ds])
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{k}\n{LABEL_NAMES[k].split(' ')[0]}" for k in range(5)])
    ax.set_title(ttl); ax.set_xlabel("LABEL"); ax.legend()
axs[0].set_ylabel("% of samples (active period)")
save(fig, "fig01_label_distribution.png")

# ------------------------------------------------------------------ Fig 2 network load timeline
fig, ax = plt.subplots(figsize=(11, 3.6))
tl = []
for ds, d in clean.items():
    x = d[d.host != "r7"]
    h = x.groupby(((x.ts_unix - x.ts_unix.min()) // 3600)).agg(t=("time", "first"), bw=("bw_max", "mean"),
                                                              ge4=("label", lambda s: (s >= 4).mean() * 100))
    ax.plot(h.t, h.bw, color=COL[ds], label=f"{ds}")
    if d.idle_tail.any():
        t0 = d.loc[d.idle_tail == 1, "time"].min()
        ax.axvspan(t0, d.time.max(), color=COL[ds], alpha=.12)
        ax.annotate("idle tail", (t0, 2), fontsize=8, color=COL[ds])
    tl.append(h.assign(dataset=ds))
ax.set_ylabel("Mean bw_max over routers (Mbps)"); ax.set_xlabel("Time (UTC)"); ax.legend(ncol=3)
ax.set_title("Hourly mean interface load, three consecutive capture sessions")
save(fig, "fig02_network_load_timeline.png")
pd.concat(tl).to_csv(TAB_DIR / "hourly_load.csv", index=False)

# ------------------------------------------------------------------ Fig 3 burst example
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharey=True)
for ax, ds in zip(axs, DATASETS):
    d = active[ds]
    r = d[(d.host == "r2")].sort_values("ts_unix")
    t0 = r.ts_unix.min() + 3600
    w = r[(r.ts_unix >= t0) & (r.ts_unix < t0 + 20 * 60)]
    ax.step(w.time, w.bw_max, where="post", color=COL[ds], lw=1.2)
    for thr, c in zip([10, 40, 70, 90], ["#999"] * 3 + ["k"]):
        ax.axhline(thr, ls="--", lw=.7, color=c)
    ax.fill_between(w.time, 0, w.bw_max.where(w.bw_max >= 90), step="post", color="k", alpha=.25)
    ax.set_ylabel("bw_max (Mbps)"); ax.set_title(f"{ds} - router r2, 20 min window", loc="left", fontsize=10)
axs[-1].set_xlabel("Time (UTC)")
save(fig, "fig03_burst_example.png")

# ------------------------------------------------------------------ Fig 4 event durations
ev = {}
rows = []
for ds, a in active.items():
    for thr in (3, 4):
        L = []
        for h, g in a[a.host != "r7"].sort_values(["host", "ts_unix"]).groupby("host"):
            L += list(run_lengths((g.label >= thr).values))
        L = np.array(L) * SAMPLING_SEC
        ev[(ds, thr)] = L
        rows.append({"dataset": ds, "event": f"LABEL>={thr}", "n_events": len(L), "median_s": np.median(L),
                     "mean_s": round(L.mean(), 1), "p90_s": np.percentile(L, 90), "max_s": L.max(),
                     "pct_single_sample_(10s)": round((L == 10).mean() * 100, 1),
                     "pct_<=30s": round((L <= 30).mean() * 100, 1), "pct_<=60s": round((L <= 60).mean() * 100, 1)})
evt = pd.DataFrame(rows)
evt.to_csv(TAB_DIR / "event_duration_stats.csv", index=False)
S["events"] = evt.to_dict("records")
print(evt.to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, thr in zip(axs, (4, 3)):
    for ds in DATASETS:
        L = np.sort(ev[(ds, thr)]); ax.step(L, np.arange(1, len(L) + 1) / len(L), where="post", color=COL[ds], label=ds)
    ax.set_xscale("log"); ax.set_xlabel("Event duration (s, consecutive samples with LABEL>=%d)" % thr)
    ax.set_title(f"LABEL >= {thr}"); ax.legend()
axs[0].set_ylabel("ECDF")
save(fig, "fig04_event_duration_ecdf.png")

# ------------------------------------------------------------------ Fig 5 transitions
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
trans = {}
for ax, (ds, a) in zip(axs, active.items()):
    a = a[a.host != "r7"].sort_values(["host", "ts_unix"])
    nxt = a.groupby("host")["label"].shift(-1)
    T = pd.crosstab(a.label, nxt, normalize="index") * 100
    trans[ds] = T.round(1)
    sns.heatmap(T, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=100)
    ax.set_title(f"{ds}: P(LABEL at t+10 s | LABEL at t) %"); ax.set_xlabel("next"); ax.set_ylabel("current")
S["transitions_pct"] = {ds: t.to_dict() for ds, t in trans.items()}
save(fig, "fig05_label_transitions.png")

# ------------------------------------------------------------------ Fig 6 host x label, Fig 7 cross-host corr
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
hl = {}
for ax, (ds, a) in zip(axs, active.items()):
    T = pd.crosstab(a.host, a.label, normalize="index") * 100
    hl[ds] = T.round(1)
    sns.heatmap(T, annot=True, fmt=".0f", cmap="YlOrRd", cbar=False, ax=ax)
    ax.set_title(f"{ds}: % of router samples per LABEL")
save(fig, "fig06_router_label_mix.png")
pd.concat(hl, axis=1).to_csv(TAB_DIR / "router_label_mix_pct.csv")

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
corrs = {}
for ax, (ds, a) in zip(axs, active.items()):
    P = a.pivot_table(index="ts_unix", columns="host", values="bw_max")[HOST_ORDER]
    C = P.corr(); corrs[ds] = C.round(2)
    sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, cbar=False, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"{ds}: corr of bw_max across routers (same t)")
save(fig, "fig07_cross_router_correlation.png")
pd.concat(corrs, axis=1).to_csv(TAB_DIR / "cross_router_corr.csv")
# lead-lag: for every router B, is the best OTHER router's load at t a better predictor of B at t+10 s
# than B's own current load?  (own lag-1 correlation vs best cross-router lag-1 correlation)
ll = []
for ds, a in active.items():
    P = a.pivot_table(index="ts_unix", columns="host", values="bw_max")[HOST_ORDER]
    for B in HOST_ORDER:
        own = P[B].corr(P[B].shift(-1))
        cand = {A: P[A].corr(P[B].shift(-1)) for A in HOST_ORDER if A != B}
        A = max(cand, key=cand.get)
        ll.append({"dataset": ds, "target": B, "own_lag1_corr": round(own, 3), "best_other_router": A,
                   "best_other_corr_to_target_t+10s": round(cand[A], 3)})
lead = pd.DataFrame(ll)
lead.to_csv(TAB_DIR / "lead_lag_vs_own_persistence.csv", index=False)
S["lead_lag_mean"] = {ds: {"own": round(g.own_lag1_corr.mean(), 3), "best_other": round(g["best_other_corr_to_target_t+10s"].mean(), 3)}
                      for ds, g in lead.groupby("dataset")}

# ------------------------------------------------------------------ Fig 8 persistence baseline
rows = []
for ds, m in model.items():
    m = m[m.host != "r7"]
    for thr in (3, 4):
        cur = m[f"cur_ge{thr}"].values
        for H in HORIZONS_SEC:
            y = m[f"y_ge{thr}_h{H}"].values
            tp = ((cur == 1) & (y == 1)).sum(); fp = ((cur == 1) & (y == 0)).sum(); fn = ((cur == 0) & (y == 1)).sum()
            tn = ((cur == 0) & (y == 0)).sum()
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            rows.append({"dataset": ds, "threshold": f">={thr}", "H_s": H, "prevalence_%": round(y.mean() * 100, 2),
                         "precision": round(p, 3), "recall": round(r, 3), "F1": round(2 * p * r / max(p + r, 1e-9), 3),
                         "FPR": round(fp / max(fp + tn, 1), 4)})
pers = pd.DataFrame(rows)
pers.to_csv(TAB_DIR / "persistence_baseline.csv", index=False)
S["persistence"] = pers.to_dict("records")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, thr in zip(axs, (">=4", ">=3")):
    for ds in DATASETS:
        q = pers[(pers.dataset == ds) & (pers.threshold == thr)]
        ax.plot(q.H_s, q.F1, "o-", color=COL[ds], label=ds)
    ax.set_title(f"'Label stays {thr}' (persistence) F1"); ax.set_xlabel("Horizon H (s)"); ax.legend()
axs[0].set_ylabel("F1 of trivial persistence predictor")
save(fig, "fig08_persistence_baseline.png")

# ------------------------------------------------------------------ Fig 9 distribution shift
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
bins = np.arange(0, 190, 5)
for ds, a in active.items():
    x = a[a.host != "r7"].bw_max
    axs[0].hist(x, bins=bins, histtype="step", lw=1.6, color=COL[ds], label=ds, density=True)
axs[0].set_yscale("log"); axs[0].set_xlabel("bw_max (Mbps)"); axs[0].set_ylabel("density (log)"); axs[0].legend()
axs[0].set_title("Load distribution (active period, excl. r7)")
for thr in (10, 40, 70, 90):
    axs[0].axvline(thr, ls=":", lw=.7, color="k")
feat_sets = ["bw_max", "bw_total", "bw_rx_sum", "bw_tx_sum", "eth_pkts_sum", "eth_octets_sum", "ucast_in_sum",
             "nucast_in_sum", "discards_sum", "rmon_err_sum", "udpInDatagrams", "udpOutDatagrams", "tcpInSegs",
             "tcpOutSegs", "bw_max_mean60s", "bw_max_std60s", "bw_max_diff1", "run_len_ge4", "steps_since_ge4",
             "ctx_net_bw_max_mean", "ctx_net_n_ge4"]
samp = {ds: m[m.host != "r7"].sample(25000, random_state=SEED) for ds, m in model.items()}
rows = []
for f in feat_sets:
    for a_, b_ in [("D1", "D2"), ("D1", "D3"), ("D2", "D3")]:
        ks = stats.ks_2samp(samp[a_][f], samp[b_][f]).statistic
        rows.append({"feature": f, "pair": f"{a_}-{b_}", "KS": round(ks, 3)})
ksd = pd.DataFrame(rows)
ksd.to_csv(TAB_DIR / "ks_distribution_shift.csv", index=False)
piv = ksd.pivot(index="feature", columns="pair", values="KS")
piv["max"] = piv.max(axis=1)
piv = piv.sort_values("max")
piv.drop(columns="max").plot.barh(ax=axs[1], width=.8, color=["#9467bd", "#8c564b", "#e377c2"])
axs[1].set_xlabel("Kolmogorov-Smirnov statistic (0 = same distribution)"); axs[1].set_title("Feature shift between datasets")
axs[1].tick_params(axis="y", labelsize=7)
save(fig, "fig09_distribution_shift.png")
S["ks_top"] = piv.sort_values("max", ascending=False).head(8).round(3).to_dict("index")

# ------------------------------------------------------------------ Fig 10 feature <-> future target association
meta = {"dataset", "host", "host_group", "ts_unix", "time", "split"}
feats = [c for c in model["D1"].columns if c not in meta and not c.startswith("y_") and c not in ("label", "cur_ge3", "cur_ge4", "idle_tail")]
tgt = "y_ge4_h30"
assoc = {}
for ds, m in model.items():
    x = m[m.host != "r7"].sample(40000, random_state=SEED)
    assoc[ds] = x[feats].apply(lambda c: stats.spearmanr(c, x[tgt])[0] if c.nunique() > 1 else np.nan)
A = pd.DataFrame(assoc)
A["mean_abs"] = A.abs().mean(axis=1)
A["min_abs"] = A[list(DATASETS)].abs().min(axis=1)
A = A.sort_values("mean_abs", ascending=False)
A.round(3).to_csv(TAB_DIR / "feature_target_spearman_ge4_h30.csv")
top = A.head(15).drop(columns=["mean_abs", "min_abs"]).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.2))
top.plot.barh(ax=ax, width=.8, color=[COL[d] for d in DATASETS])
ax.set_xlabel(f"Spearman correlation with target {tgt}"); ax.set_title("Top-15 features: association with future LABEL>=4 (H=30 s)")
save(fig, "fig10_feature_target_association.png")
S["assoc_top10"] = A.head(10).round(3).to_dict("index")

# ------------------------------------------------------------------ redundancy among model features
x = pd.concat([m[m.host != "r7"].sample(30000, random_state=SEED)[feats] for m in model.values()])
C = x.corr().abs()
pairs = [(a_, b_, round(C.loc[a_, b_], 4)) for i, a_ in enumerate(C.columns) for b_ in C.columns[i + 1:] if C.loc[a_, b_] > .98]
pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_corr"]).sort_values("abs_corr", ascending=False)\
    .to_csv(TAB_DIR / "redundant_feature_pairs_gt098.csv", index=False)
S["n_redundant_pairs_gt098"] = len(pairs)

# ------------------------------------------------------------------ data quality summary
log = json.loads((TAB_DIR / "cleaning_log.json").read_text())
S["cleaning"] = {ds: {k: v for k, v in l.items() if k != "constant_cols_dropped"} for ds, l in log.items()}
S["constant_cols_dropped"] = log["D1"]["constant_cols_dropped"]
(TAB_DIR / "eda_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("EDA done")
