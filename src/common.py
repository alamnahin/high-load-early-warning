"""Small helpers shared by several scripts."""
import numpy as np
import pandas as pd

from config import LABEL_EDGES_MBPS, IFACES


def bw_columns(df: pd.DataFrame, include_if6: bool = True):
    cols = []
    for i in IFACES:
        if i == 6 and not include_if6:
            continue
        cols += [f"bw_rx_{i}", f"bw_tx_{i}"]
    return [c for c in cols if c in df.columns]


def label_from_bw(bw_max: pd.Series) -> pd.Series:
    """Re-derive LABEL from the max interface bandwidth (Mbps)."""
    edges = [-np.inf] + list(LABEL_EDGES_MBPS) + [np.inf]
    return pd.cut(bw_max, bins=edges, labels=False, right=False)


def run_lengths(binary: np.ndarray):
    """Lengths (in samples) of consecutive runs of 1s in a 0/1 array."""
    x = np.asarray(binary).astype(np.int8)
    if x.size == 0:
        return np.array([], dtype=int)
    padded = np.r_[0, x, 0]
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return ends - starts
