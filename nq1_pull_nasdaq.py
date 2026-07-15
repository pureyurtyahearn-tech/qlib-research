"""Overnight resumable pull: SHARADAR/SEP prices for ALL NASDAQ common-stock names.

Universe = 11,872 NASDAQ-listed common stocks (primary class; warrants/preferred/secondary
excluded), incl. delisted -> survivorship-free. NO index membership (Sharadar has none for
NASDAQ); eligibility = "listed & trading on NASDAQ on date t" (a name has a price bar iff it
traded). Liquidity filter (median $vol >= $1M store / $5M trade) applied AFTER, on prices.

RESUMABILITY (non-negotiable): each batch of 100 tickers is pickled to disk IMMEDIATELY
(_nasdaq_batches/bNNNN.pkl) before moving on. On restart, existing batch files are skipped,
so a death/sleep/crash loses at most one in-flight batch. A progress line is flushed per
batch for the paired Monitor.
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
CHUNK = 100
COLS = ["ticker", "date", "open", "high", "low", "close", "closeadj", "closeunadj", "volume"]


def main():
    tick = pd.read_csv(SH / "nasdaq_candidates.csv", low_memory=False)["ticker"].astype(str).tolist()
    tick = sorted(set(tick))
    print(f"NASDAQ common-stock pull: {len(tick)} tickers  {START}..{END}", flush=True)
    ckpt = SH / "_nasdaq_batches"; ckpt.mkdir(exist_ok=True)
    cache = SH / "sep_nasdaq_raw.h5"
    n_b = (len(tick) + CHUNK - 1) // CHUNK

    if not cache.exists():
        t0 = time.time(); done = len(list(ckpt.glob("b*.pkl")))
        print(f"resuming: {done}/{n_b} batches already checkpointed", flush=True)
        for bi in range(n_b):
            bf = ckpt / f"b{bi:04d}.pkl"
            if bf.exists():
                continue
            ch = tick[bi * CHUNK:(bi + 1) * CHUNK]
            for att in range(5):
                try:
                    d = ndl.get_table("SHARADAR/SEP", ticker=ch, date={"gte": START, "lte": END},
                                      qopts={"columns": COLS}, paginate=True)
                    break
                except Exception as e:
                    if att == 4:
                        print(f"  BATCH {bi} FAILED after retries: {str(e)[:80]}", flush=True)
                        raise
                    time.sleep(10)
            d.to_pickle(bf)                        # checkpoint IMMEDIATELY
            got = d.ticker.nunique() if len(d) else 0
            print(f"  [{min((bi+1)*CHUNK, len(tick)):>6}/{len(tick)}] batch {bi+1}/{n_b}  "
                  f"{len(d):>7,} rows, {got}/{len(ch)} tickers  ({time.time()-t0:.0f}s)", flush=True)
        print("all batches pulled; concatenating...", flush=True)
        df = pd.concat([pd.read_pickle(p) for p in sorted(ckpt.glob("b*.pkl"))], ignore_index=True)
        df.to_hdf(cache, key="p", complevel=5)
        print(f"saved {len(df):,} rows -> {cache}", flush=True)
    else:
        df = pd.read_hdf(cache)
        print(f"cached raw: {len(df):,} rows", flush=True)

    df["date"] = pd.to_datetime(df["date"])
    have = set(df.ticker.unique())
    missing = [t for t in tick if t not in have]
    print(f"\ncoverage: {len(have)}/{len(tick)} tickers returned data; no-data: {len(missing)}", flush=True)
    print(f"rows={len(df):,}  dates {df.date.min().date()}..{df.date.max().date()}", flush=True)
    print(f"non-positive/NaN closeadj: {int((df.closeadj <= 0).sum() + df.closeadj.isna().sum())}", flush=True)

    # consistent-basis panel: closeadj is price; OHLC rescaled by closeadj/close (Sharadar FAQ)
    scale = (df["closeadj"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        "ticker": df.ticker, "date": df.date,
        "$open": df["open"] * scale, "$high": df["high"] * scale,
        "$low": df["low"] * scale, "$close": df["closeadj"], "$volume": df["volume"],
    }).dropna(subset=["$close"]).set_index(["date", "ticker"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_hdf(SH / "sep_nasdaq_panel.h5", key="d", complevel=5)
    print(f"panel -> sep_nasdaq_panel.h5  {len(out):,} rows, "
          f"{out.index.get_level_values('ticker').nunique()} tickers, "
          f"{out.index.get_level_values('date').nunique()} dates", flush=True)
    print("PULL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
