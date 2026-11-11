"""Workloads: real NSE+BSE daily-volume panel (two independent exchange sources per stock-day)
and a synthetic generator with the same schema. Builds decision sets over shared evidence.

Real data are read-only from the ICRITO-2027 package (official exchange bhavcopies); they are
not redistributed. Truth for each item = the exchange-reported daily volume.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

ICRITO = r"C:\Users\mailt\Downloads\ICRITO-2027"
CAL_DAYS = 120          # calibration prefix (thresholds, domain caps); never evaluated
WINDOWS = (5, 20)       # sliding-window lengths (trading days)
MULTS = (1.0, 1.5)      # threshold multiples of the calibration median
import os
CAP_MULT = float(os.environ.get("CAP_MULT", "3.0"))  # declared domain bound multiplier


def load_real():
    nse = pd.read_parquet(f"{ICRITO}/data/derived/official.parquet", columns=["date", "isin", "series", "volume"])
    bse = pd.read_parquet(f"{ICRITO}/data/derived/bse.parquet", columns=["date", "isin", "volume"])
    nse = nse[nse.series == "EQ"].drop(columns="series")
    ind = pd.read_csv(f"{ICRITO}/data/raw/ind_nifty500list.csv")[["ISIN Code", "Industry"]]
    ind.columns = ["isin", "industry"]
    a = nse.pivot_table(index="isin", columns="date", values="volume", aggfunc="sum")
    b = bse.pivot_table(index="isin", columns="date", values="volume", aggfunc="sum")
    dates = a.columns.intersection(b.columns)
    keys = a.index.intersection(b.index).intersection(pd.Index(ind["isin"]))
    a, b = a.loc[keys, dates], b.loc[keys, dates]
    ok = (a.notna().all(axis=1) & b.notna().all(axis=1))       # complete truth on both sources
    a, b = a[ok], b[ok]
    industry = ind.set_index("isin").loc[a.index, "industry"].values
    V = np.stack([a.values, b.values], axis=1).astype(float)    # (K, 2 sources, T)
    return V, industry, list(a.index), list(dates)


def synthetic(K=400, T=300, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(12, 1.2, size=(K, 2)) * np.array([1.0, 0.15])
    ar = np.zeros((K, 2, T))
    x = np.zeros((K, 2))
    for t in range(T):
        x = 0.8 * x + rng.normal(0, 0.45, size=(K, 2))
        ar[:, :, t] = x
    spikes = rng.random((K, 2, T)) < 0.03
    V = base[:, :, None] * np.exp(ar) * np.where(spikes, rng.uniform(2, 6, (K, 2, T)), 1.0)
    industry = np.array([f"G{g:02d}" for g in rng.integers(0, n_groups, K)])
    return np.round(V), industry


class Calib:
    def __init__(self, V, industry):
        cal = V[:, :, :CAL_DAYS]
        self.med = np.median(cal.sum(axis=1), axis=1)             # per-key combined median
        self.src_med = np.median(cal, axis=2)                      # (K,2) for imputation baseline
        self.cap = CAP_MULT * cal.max(axis=2)                      # (K,2) declared domain bound
        self.groups = {g: np.where(industry == g)[0] for g in np.unique(industry)}


def build_episode(V, cal, t, Wmax=max(WINDOWS)):
    """Decision set for windows ending at day t. Items = (key, source, lag 0..Wmax-1)."""
    K = V.shape[0]
    lags = np.arange(Wmax)
    # item index: ((k*2)+s)*Wmax + lag
    n_items = K * 2 * Wmax
    truth = V[:, :, t - lags]                                      # (K,2,Wmax)
    truth = truth.reshape(-1)
    rows, cols, theta, meta = [], [], [], []
    d = 0
    item = lambda k, s, l: (k * 2 + s) * Wmax + l
    for W in WINDOWS:
        mem = [(s, l) for s in (0, 1) for l in range(W)]
        for c in MULTS:
            for k in range(K):                                     # stock-level decisions
                for s, l in mem:
                    rows.append(d); cols.append(item(k, s, l))
                theta.append(c * W * cal.med[k]); meta.append(("stock", W, c)); d += 1
            for g, ks in cal.groups.items():                      # industry roll-ups
                for k in ks:
                    for s, l in mem:
                        rows.append(d); cols.append(item(k, s, l))
                theta.append(c * W * cal.med[ks].sum()); meta.append(("industry", W, c)); d += 1
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(d, n_items))
    cap = np.repeat(cal.cap[:, :, None], Wmax, axis=2).reshape(-1)
    smed = np.repeat(cal.src_med[:, :, None], Wmax, axis=2).reshape(-1)
    lag = np.tile(lags, K * 2)
    return M, np.array(theta), truth, cap, smed, lag, meta


def evidence(truth, cap, smed, lag, scenario, rng, p=0.3, q=0.3, eps=0.10):
    """Return (lo, hi, point_estimate, kind). kind: 0 exact, 1 noisy interval, 2 missing."""
    n = len(truth)
    if scenario == "mcar":
        pm = np.full(n, p)
    elif scenario == "late":                    # late-arriving feeds: recent days mostly missing
        pm = 0.9 * 0.7 ** lag
    elif scenario == "mnar":                    # adversarial: spikes are the ones that go missing
        pm = np.where(truth > 1.5 * smed, 0.6, 0.15)
    else:
        raise ValueError(scenario)
    miss = rng.random(n) < pm
    noisy = (~miss) & (rng.random(n) < q)
    r = truth * (1 + rng.uniform(-eps, eps, n))                   # noisy reading (e.g. OCR / derived)
    lo, hi = truth.copy(), truth.copy()
    lo[noisy], hi[noisy] = r[noisy] / (1 + eps), r[noisy] / (1 - eps)
    lo[miss], hi[miss] = 0.0, np.maximum(cap[miss], 0)
    est = truth.copy(); est[noisy] = r[noisy]; est[miss] = smed[miss]
    kind = np.zeros(n, np.int8); kind[noisy] = 1; kind[miss] = 2
    return lo, hi, est, kind
