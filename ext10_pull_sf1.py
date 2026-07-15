"""Pull SHARADAR/SF1 fundamentals, dimension=ART (As Reported, Trailing-Twelve-Month) for
the full PIT S&P 500 universe (1159 ever-members).

WHY ART (not MR): AR excludes restatements and is time-indexed to the actual SEC FILING
date (datekey), the genuine point-in-time view for backtesting. MR uses restated numbers
that were not known at the time -> look-ahead. This is the entire reason for using this data.
WHY TTM (not Q): trailing-twelve-month flows avoid quarterly seasonality for revenue/eps/
roe/fcf; still filing-date-indexed, still updates each quarter.

datekey = filing date (what we merge on). calendardate/reportperiod = fiscal period (NOT
used for merging -- using it would leak ~1-3 months of future information).
"""
import warnings; warnings.filterwarnings("ignore")
import os, time
import numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nasdaqdatalink as ndl

load_dotenv(".env", override=True)
ndl.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
SH = Path("git_ignore_folder/sharadar")
DIM = "ART"
COLS = ["ticker", "dimension", "calendardate", "datekey", "reportperiod",
        "eps", "revenue", "de", "roe", "fcf", "marketcap", "equity", "bvps",
        "sharesbas", "pe", "pb", "netinc", "ncfo", "debt"]
CHUNK = 100


def main():
    tick = pd.read_csv(SH / "ever_members_full.csv")["ticker"].tolist()
    print(f"pulling SF1 dimension={DIM} for {len(tick)} tickers")
    ckpt = SH / "_sf1_batches"; ckpt.mkdir(exist_ok=True)
    cache = SH / "sf1_art_raw.h5"
    if not cache.exists():
        n_b = (len(tick) + CHUNK - 1) // CHUNK
        t0 = time.time()
        for bi in range(n_b):
            bf = ckpt / f"b{bi:03d}.pkl"
            if bf.exists():
                continue
            ch = tick[bi * CHUNK:(bi + 1) * CHUNK]
            for att in range(4):
                try:
                    d = ndl.get_table("SHARADAR/SF1", ticker=ch, dimension=DIM,
                                      qopts={"columns": COLS}, paginate=True)
                    break
                except Exception:
                    if att == 3:
                        raise
                    time.sleep(8)
            d.to_pickle(bf)
            print(f"  [{min((bi+1)*CHUNK,len(tick)):>4}/{len(tick)}] {len(d):>6} rows ({time.time()-t0:.0f}s)", flush=True)
        df = pd.concat([pd.read_pickle(p) for p in sorted(ckpt.glob("b*.pkl"))], ignore_index=True)
        df.to_hdf(cache, key="f", complevel=5)
        print(f"saved {len(df):,} rows -> {cache}")
    else:
        df = pd.read_hdf(cache)
        print(f"cached: {len(df):,} rows")

    # ---- confirm dimension + point-in-time property ----
    assert set(df.dimension.unique()) == {DIM}, f"UNEXPECTED dimensions: {df.dimension.unique()}"
    df["datekey"] = pd.to_datetime(df["datekey"])
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    lag = (df["datekey"] - df["calendardate"]).dt.days
    print(f"\n=== DIMENSION CONFIRMED: {DIM} (As Reported TTM), {df.ticker.nunique()} tickers ===")
    print(f"  filing lag (datekey - calendardate): median {int(lag.median())}d, "
          f"1%..99% = {int(lag.quantile(.01))}..{int(lag.quantile(.99))}d  (all >0 => filing-indexed)")
    print(f"  negative-lag rows (would be look-ahead!): {int((lag < 0).sum())}")
    print(f"  datekey range: {df.datekey.min().date()} .. {df.datekey.max().date()}")
    print(f"  rows/ticker: median {int(df.groupby('ticker').size().median())}")
    print(f"\n  indicator null rates:")
    for k in ["eps", "revenue", "de", "roe", "fcf", "marketcap", "equity", "bvps", "sharesbas"]:
        print(f"    {k:11} {df[k].isna().mean():.1%}")


if __name__ == "__main__":
    main()
