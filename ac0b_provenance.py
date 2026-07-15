"""Which source covers which dates? The 2020->2021 cliff must be understood before
choosing a 12-1 momentum test window: pre-2021 history may be Kaggle-NYSE (the data we
had to split-fix + winsorize), while 2021+ is yfinance-SP500 (clean)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SRC = Path("git_ignore_folder/factor_implementation_source_data")

def span(df, label):
    c = df["$close"].unstack("instrument")
    d = c.index
    print(f"{label:28} {c.shape[1]:>5} insts   {d.min().date()} .. {d.max().date()}   rows={len(c):,}")
    return c

def main():
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp_bak = pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
    c_comb = span(comb, "combined daily_pv.h5")
    c_sp = span(sp_bak, "SP500-only backup (yfinance)")

    sp500 = sorted(set(sp_bak.index.get_level_values("instrument").unique()))
    nyse = sorted(set(comb.index.get_level_values("instrument").unique()) - set(sp500))
    print(f"\nNYSE-only (Kaggle) instruments: {len(nyse)}")

    print(f"\n{'year':>6}{'SP500 w/ data':>15}{'NYSE w/ data':>14}")
    for y in range(2000, 2027):
        rs = c_comb[c_comb.index.year == y]
        if len(rs) == 0:
            continue
        nsp = int(rs[[c for c in sp500 if c in rs.columns]].notna().any(axis=0).sum())
        nny = int(rs[[c for c in nyse if c in rs.columns]].notna().any(axis=0).sum())
        print(f"{y:>6}{nsp:>15}{nny:>14}")

    # per-instrument first/last date for the yfinance SP500 source
    fl = []
    for t in sp500:
        s = c_sp[t].dropna()
        if len(s):
            fl.append((t, s.index[0], s.index[-1], len(s)))
    fl = pd.DataFrame(fl, columns=["t", "first", "last", "n"])
    print(f"\n=== yfinance SP500 source: per-instrument history ===")
    print(f"  instruments with any data: {len(fl)}")
    print(f"  first-date  min={fl['first'].min().date()}  median={fl['first'].median().date()}  max={fl['first'].max().date()}")
    print(f"  last-date   min={fl['last'].min().date()}   median={fl['last'].median().date()}  max={fl['last'].max().date()}")
    for cut in ["2001-01-01", "2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"]:
        n = int((fl["first"] <= cut).sum())
        print(f"  have history starting on/before {cut}: {n:>4} instruments")
    alive = int((fl["last"] >= "2026-06-01").sum())
    print(f"  still alive in Jun-2026: {alive}")

if __name__ == "__main__":
    main()
