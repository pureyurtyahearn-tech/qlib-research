"""Pull SHARADAR/SF1 dimension=ART for the NASDAQ-only liquid universe (3,272 names that
have NEVER been in the S&P 500 screen). Mechanically identical to ext10 (SP500 pull) --
same ART (As-Reported TTM) dimension, same datekey (filing-date) indexing, same columns,
same resumable per-batch checkpointing. Only the input list and output paths differ.

WHY ART not MR: AR excludes restatements, indexed to the actual SEC filing date (datekey) =
the genuine point-in-time view. MR would leak future restatements. (Same reason as SP500.)
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
TICK_CSV = SH / "nasdaq_only_tickers.csv"
CKPT = SH / "_sf1_nasdaq_batches"
CACHE = SH / "sf1_nasdaq_raw.h5"


def main():
    tick = pd.read_csv(TICK_CSV)["ticker"].astype(str).tolist()
    print(f"pulling SF1 dimension={DIM} for {len(tick)} NASDAQ-only tickers", flush=True)
    CKPT.mkdir(exist_ok=True)
    if not CACHE.exists():
        n_b = (len(tick) + CHUNK - 1) // CHUNK
        t0 = time.time()
        for bi in range(n_b):
            bf = CKPT / f"b{bi:03d}.pkl"
            if bf.exists():
                continue
            ch = tick[bi * CHUNK:(bi + 1) * CHUNK]
            for att in range(5):
                try:
                    d = ndl.get_table("SHARADAR/SF1", ticker=ch, dimension=DIM,
                                      qopts={"columns": COLS}, paginate=True)
                    break
                except Exception:
                    if att == 4:
                        raise
                    time.sleep(10)
            d.to_pickle(bf)
            print(f"  [{min((bi+1)*CHUNK,len(tick)):>4}/{len(tick)}] batch {bi:03d}: "
                  f"{len(d):>6} rows, {d.ticker.nunique() if len(d) else 0:>3} tickers "
                  f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.concat([pd.read_pickle(p) for p in sorted(CKPT.glob("b*.pkl"))], ignore_index=True)
        df.to_hdf(CACHE, key="f", complevel=5)
        print(f"saved {len(df):,} rows -> {CACHE}", flush=True)
    else:
        df = pd.read_hdf(CACHE)
        print(f"cached: {len(df):,} rows", flush=True)

    # ---- confirm dimension + point-in-time property + ACTUAL coverage ----
    assert set(df.dimension.unique()) == {DIM}, f"UNEXPECTED dimensions: {df.dimension.unique()}"
    df["datekey"] = pd.to_datetime(df["datekey"])
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    lag = (df["datekey"] - df["calendardate"]).dt.days
    n_target = len(tick)
    n_data = df.ticker.nunique()
    print(f"\n=== DIMENSION CONFIRMED: {DIM} (As-Reported TTM) ===")
    print(f"  ACTUAL SF1 coverage: {n_data} of {n_target} target NASDAQ-only names "
          f"({n_data/n_target:.0%}) returned data; {n_target-n_data} have NO SF1")
    print(f"  filing lag (datekey - calendardate): median {int(lag.median())}d, "
          f"1%..99% = {int(lag.quantile(.01))}..{int(lag.quantile(.99))}d")
    print(f"  negative-lag rows (look-ahead!): {int((lag < 0).sum())}")
    print(f"  datekey range: {df.datekey.min().date()} .. {df.datekey.max().date()}")
    rpt = df.groupby("ticker").size()
    print(f"  filings/ticker: median {int(rpt.median())}, p10 {int(rpt.quantile(.1))} "
          f"(names with <8 filings ~ <2yr history: {int((rpt < 8).sum())})")
    print(f"\n  indicator null rates (of all filing rows):")
    for k in ["eps", "revenue", "de", "roe", "fcf", "marketcap", "equity", "bvps", "sharesbas"]:
        print(f"    {k:11} {df[k].isna().mean():.1%}")


if __name__ == "__main__":
    main()
