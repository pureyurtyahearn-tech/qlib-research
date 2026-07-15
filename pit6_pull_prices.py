"""Pull SHARADAR/SEP prices for the full 822-ticker point-in-time SP500 universe.

Window starts 2008-06-01 (not 2010-01-01): the 12-1 momentum factor needs 252 trading days
of lookback before the first evaluable date, so the panel must start ~18 months early.

Adjustment (per Sharadar docs):
  close      -> adjusted for splits + stock dividends ONLY
  closeadj   -> splits + stock dividends + CASH DIVIDENDS + SPINOFFS  <-- our primary
  closeunadj -> raw
We take closeadj as the price series. OHLC are only split/stock-div adjusted, so to keep
them on the SAME basis as closeadj we rescale: openadj = open * closeadj/close (the
formula Sharadar gives in its own FAQ). Otherwise open and close would live on different
adjustment bases -- exactly the kind of mixed-lineage bug that bit us with the NYSE data.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time
import numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nasdaqdatalink as ndl

load_dotenv(".env", override=True)
ndl.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
OUT = Path("git_ignore_folder/sharadar")
START, END = "2008-06-01", "2026-06-29"
CHUNK = 40


def main():
    mat = pd.read_hdf(OUT / "sp500_pit_membership.h5")
    tick = sorted(mat.columns[mat.any(axis=0)])
    print(f"PIT universe: {len(tick)} tickers   window {START} .. {END}")

    cache = OUT / "sep_prices.h5"
    if cache.exists():
        df = pd.read_hdf(cache)
        print(f"cached: {len(df):,} rows, {df.ticker.nunique()} tickers")
    else:
        parts = []
        t0 = time.time()
        for i in range(0, len(tick), CHUNK):
            ch = tick[i:i + CHUNK]
            for attempt in range(3):
                try:
                    d = ndl.get_table("SHARADAR/SEP", ticker=ch,
                                      date={"gte": START, "lte": END}, paginate=True)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(5)
            parts.append(d)
            got = d.ticker.nunique() if len(d) else 0
            print(f"  [{i+len(ch):>4}/{len(tick)}] {len(d):>7,} rows, {got}/{len(ch)} tickers"
                  f"   ({time.time()-t0:.0f}s)")
        df = pd.concat(parts, ignore_index=True)
        df.to_hdf(cache, key="p", complevel=5)
        print(f"\nsaved {len(df):,} rows -> {cache}")

    df["date"] = pd.to_datetime(df["date"])
    have = set(df.ticker.unique())
    missing = [t for t in tick if t not in have]
    print(f"\n=== coverage ===")
    print(f"  tickers requested : {len(tick)}")
    print(f"  tickers returned  : {len(have)}")
    print(f"  NO DATA           : {len(missing)}  {missing[:12]}")

    # rows / adjustment sanity
    print(f"  rows              : {len(df):,}")
    print(f"  date range        : {df.date.min().date()} .. {df.date.max().date()}")
    bad = df[(df.closeadj <= 0) | (df.closeadj.isna())]
    print(f"  non-positive/NaN closeadj rows: {len(bad):,}")

    # build wide panels on a consistent adjustment basis
    scale = (df["closeadj"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        "ticker": df.ticker, "date": df.date,
        "$open": df["open"] * scale, "$high": df["high"] * scale,
        "$low": df["low"] * scale, "$close": df["closeadj"],
        "$volume": df["volume"],
    })
    out = out.dropna(subset=["$close"]).set_index(["date", "ticker"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_hdf(OUT / "sep_panel.h5", key="d", complevel=5)
    print(f"\n  panel -> {OUT/'sep_panel.h5'}   {len(out):,} rows, "
          f"{out.index.get_level_values('ticker').nunique()} tickers, "
          f"{out.index.get_level_values('date').nunique()} dates")

    # how many of the previously-MISSING names did we just recover?
    ours = set(pd.read_hdf("git_ignore_folder/factor_implementation_source_data/"
                           "daily_pv_sp500_backup.h5").index.get_level_values("instrument").unique())
    newly = [t for t in have if t not in ours]
    print(f"\n  names we did NOT have before and now DO: {len(newly)}")
    dead = [t for t in ["CELG", "RHT", "EKDKQ", "ETFC", "AGN", "NBL", "PCG", "M", "JWN", "HOG"]
            if t in have]
    print(f"  survivorship-hole spot check present: {dead}")


if __name__ == "__main__":
    main()
