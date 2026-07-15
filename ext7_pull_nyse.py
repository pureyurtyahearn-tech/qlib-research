"""Consolidation item 2: pull the NYSE-only broad universe from Sharadar SEP (full history),
so daily_pv.h5 can be rebuilt from a SINGLE source instead of yfinance+Kaggle+Sharadar.

Ticker set = the liquidity-filtered NYSE-only names (nyse_store_universe.csv, 1502) MINUS
any already in the SP500 ever-member pull. Sharadar handles splits/divs natively, retiring
fix_and_build_nyse.py's manual yfinance-metadata adjustment.
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
SRC = Path("git_ignore_folder/factor_implementation_source_data")
START, END = "1997-06-01", "2026-06-29"
CHUNK = 40


def main():
    nyse = pd.read_csv(SRC / "nyse_store_universe.csv")["ticker"].tolist()
    sp = set(pd.read_csv(SH / "ever_members_full.csv")["ticker"])
    tick = sorted(set(nyse) - sp)
    print(f"NYSE-only names: {len(nyse)}; not already in SP500 pull: {len(tick)}")

    cache = SH / "sep_nyse_full.h5"
    ckpt = SH / "_nyse_batches"; ckpt.mkdir(exist_ok=True)   # per-batch checkpoints (resumable)
    if cache.exists():
        df = pd.read_hdf(cache)
        print(f"cached: {len(df):,} rows")
    else:
        t0 = time.time()
        n_batches = (len(tick) + CHUNK - 1) // CHUNK
        for bi in range(n_batches):
            bf = ckpt / f"b{bi:04d}.pkl"
            if bf.exists():          # resume: skip batches already pulled
                continue
            ch = tick[bi * CHUNK:(bi + 1) * CHUNK]
            for att in range(4):
                try:
                    d = ndl.get_table("SHARADAR/SEP", ticker=ch,
                                      date={"gte": START, "lte": END}, paginate=True)
                    break
                except Exception:
                    if att == 3:
                        raise
                    time.sleep(8)
            d.to_pickle(bf)          # checkpoint immediately, before moving on
            print(f"  [{min((bi+1)*CHUNK, len(tick)):>4}/{len(tick)}] {len(d):>7,} rows  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        parts = [pd.read_pickle(p) for p in sorted(ckpt.glob("b*.pkl"))]
        df = pd.concat(parts, ignore_index=True)
        df.to_hdf(cache, key="p", complevel=5)
        print(f"saved {len(df):,} rows -> {cache}  ({len(list(ckpt.glob('b*.pkl')))} batches)")

    df["date"] = pd.to_datetime(df["date"])
    have = set(df.ticker.unique())
    print(f"\nreturned {len(have)}/{len(tick)} tickers; not in Sharadar: {len(tick)-len(have)}")
    scale = (df["closeadj"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        "ticker": df.ticker, "date": df.date,
        "$open": df["open"] * scale, "$high": df["high"] * scale,
        "$low": df["low"] * scale, "$close": df["closeadj"], "$volume": df["volume"],
    }).dropna(subset=["$close"]).set_index(["date", "ticker"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_hdf(SH / "sep_nyse_panel.h5", key="d", complevel=5)
    print(f"panel -> sep_nyse_panel.h5  {len(out):,} rows, {out.index.get_level_values('ticker').nunique()} tickers")


if __name__ == "__main__":
    main()
