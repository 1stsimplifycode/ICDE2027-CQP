"""Build aligned per-symbol panels from the raw downloads.

official(): NSE bhavcopy rows for the universe (series EQ/BE/BZ, i.e. the normal-market cash listing), keyed by
            symbol *and* ISIN so renames are visible. Columns: open high low close prev_close volume isin series.
yahoo():    Yahoo snapshot for one suffix (NS or BO). Columns: open high low close adj_close volume div split.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "data", "derived")


def universe() -> pd.DataFrame:
    u = pd.read_csv(os.path.join(ROOT, "data", "raw", "ind_nifty500list.csv"))
    return u.rename(columns={"Symbol": "symbol", "ISIN Code": "isin", "Industry": "industry"})


def official(rebuild: bool = False) -> pd.DataFrame:
    p = os.path.join(DER, "official.parquet")
    if os.path.exists(p) and not rebuild:
        return pd.read_parquet(p)
    u = universe()
    isins, syms = set(u["isin"]), set(u["symbol"])
    keep = ["TradDt", "ISIN", "TckrSymb", "SctySrs", "OpnPric", "HghPric", "LwPric", "ClsPric", "PrvsClsgPric",
            "TtlTradgVol"]
    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "bhav", "*.csv.zip"))):
        d = pd.read_csv(f, usecols=keep)
        # ISINs change on splits (face-value change), so match on current ISIN OR current ticker
        d = d[(d["ISIN"].isin(isins) | d["TckrSymb"].isin(syms)) & d.SctySrs.isin(["EQ", "BE", "BZ"])]
        parts.append(d)
    o = pd.concat(parts, ignore_index=True)
    o.columns = ["date", "isin", "bhav_symbol", "series", "open", "high", "low", "close", "prev_close", "volume"]
    o["date"] = pd.to_datetime(o["date"])
    sym = dict(zip(u["isin"], u["symbol"]))
    o["symbol"] = o["isin"].map(sym).fillna(o["bhav_symbol"])  # current symbol; bhav_symbol = historical one
    o["isin_is_current"] = o["isin"].isin(isins)
    o = o.sort_values(["symbol", "date"]).reset_index(drop=True)
    os.makedirs(DER, exist_ok=True)
    o.to_parquet(p)
    return o


def bse(rebuild: bool = False) -> pd.DataFrame:
    """Official BSE bhavcopy rows for the universe, matched on ISIN (BSE tickers differ from NSE tickers).
    Pre-split ISINs are mapped through the NSE file's ISIN->symbol pairs (same ISIN on both exchanges)."""
    p = os.path.join(DER, "bse.parquet")
    if os.path.exists(p) and not rebuild:
        return pd.read_parquet(p)
    o = official()
    isin2sym = dict(zip(o["isin"], o["symbol"]))
    keep = ["TradDt", "ISIN", "TckrSymb", "SctySrs", "OpnPric", "HghPric", "LwPric", "ClsPric", "PrvsClsgPric",
            "TtlTradgVol"]
    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "bse", "*.csv.gz"))):
        d = pd.read_csv(f, usecols=keep)
        d = d[d["ISIN"].isin(set(isin2sym))]
        parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    b.columns = ["date", "isin", "bse_symbol", "series", "open", "high", "low", "close", "prev_close", "volume"]
    b["date"] = pd.to_datetime(b["date"])
    b["symbol"] = b["isin"].map(isin2sym)
    b = b.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"]).reset_index(drop=True)
    b.to_parquet(p)
    return b


def yahoo(snap: str, suffix: str = "NS") -> dict[str, pd.DataFrame]:
    out = {}
    for f in glob.glob(os.path.join(ROOT, "data", "raw", "yahoo", snap, f"*.{suffix}.csv")):
        sym = os.path.basename(f)[: -len(f".{suffix}.csv")]
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        d = d.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                              "Adj Close": "adj_close", "Volume": "volume", "Dividends": "div",
                              "Stock Splits": "split"})
        d.index = d.index.normalize()
        out[sym] = d
    return out
