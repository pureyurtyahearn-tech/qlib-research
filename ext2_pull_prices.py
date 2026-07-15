"""Pull SHARADAR/SEP for the FULL ever-member universe (1159 tickers), full history.
Start 1997-06-01 so a 252d momentum lookback is defined before the 1998-03-31 membership
start. Writes raw -> sep_prices_full.h5 and a consistent-basis panel -> sep_panel_full.h5.

Adjustment identical to pit6: closeadj is the price (splits+stockdivs+cashdivs+spinoffs);
OHLC rescaled by closeadj/close so all four share closeadj's basis.
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
START, END = "1997-06-01", "2026-06-29"
CHUNK = 40


def main():
    tick = pd.read_csv(SH / "ever_members_full.csv")["ticker"].tolist()
    print(f"pulling {len(tick)} tickers  {START}..{END}")
    cache = SH / "sep_prices_full.h5"
    if cache.exists():
        df = pd.read_hdf(cache)
        print(f"cached: {len(df):,} rows")
    else:
        parts = []
        t0 = time.time()
        for i in range(0, len(tick), CHUNK):
            ch = tick[i:i + CHUNK]
            for att in range(3):
                try:
                    d = ndl.get_table("SHARADAR/SEP", ticker=ch,
                                      date={"gte": START, "lte": END}, paginate=True)
                    break
                except Exception:
                    if att == 2:
                        raise
                    time.sleep(5)
            parts.append(d)
            print(f"  [{i+len(ch):>4}/{len(tick)}] {len(d):>7,} rows  ({time.time()-t0:.0f}s)", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_hdf(cache, key="p", complevel=5)
        print(f"saved {len(df):,} rows -> {cache}")

    df["date"] = pd.to_datetime(df["date"])
    have = set(df.ticker.unique())
    missing = [t for t in tick if t not in have]
    print(f"\ncoverage: {len(have)}/{len(tick)} tickers returned data; missing {len(missing)}: {missing[:15]}")
    print(f"rows={len(df):,}  dates {df.date.min().date()}..{df.date.max().date()}")
    print(f"non-positive/NaN closeadj: {int((df.closeadj <= 0).sum() + df.closeadj.isna().sum())}")

    scale = (df["closeadj"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        "ticker": df.ticker, "date": df.date,
        "$open": df["open"] * scale, "$high": df["high"] * scale,
        "$low": df["low"] * scale, "$close": df["closeadj"], "$volume": df["volume"],
    }).dropna(subset=["$close"]).set_index(["date", "ticker"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_hdf(SH / "sep_panel_full.h5", key="d", complevel=5)
    print(f"panel -> sep_panel_full.h5  {len(out):,} rows, "
          f"{out.index.get_level_values('ticker').nunique()} tickers, "
          f"{out.index.get_level_values('date').nunique()} dates")


if __name__ == "__main__":
    main()
