"""
Build NYSE daily_pv.h5 from Kaggle data with yfinance-sourced split+dividend adjustments.

Pipeline:
  1. Load Kaggle NYSE CSV (3168 tickers, 2019-2024), filter holiday rows (volume=0)
  2. Per ticker: fetch split and dividend history from yfinance
  3. Apply backward adjustments:
       - Splits first  (exact ratios, divide OHLC / ratio, multiply volume)
       - Dividends second (yfinance amounts are split-adjusted; adj_factor uses split-adjusted price)
  4. SHOP verification: confirm Jun 29 2022 split is removed from the price series
  5. Convert to daily_pv.h5 format: MultiIndex (datetime, instrument), float32, $factor=1.0
  6. Merge with existing SP500 daily_pv.h5 (SP500 wins on any overlap)
  7. Save combined file; backup SP500 first
"""
import warnings; warnings.filterwarnings("ignore")
import time, pickle, shutil, os, sys
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
KAGGLE_CSV  = "/home/codespace/.cache/kagglehub/datasets/mousemover/quant-finance-nyse-5-years/versions/1/NYSE_fully_cleaned_2019_2024.csv"
SP500_H5    = "git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
OUT_H5      = "git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
NYSE_ONLY_H5 = "git_ignore_folder/nyse_adjusted_daily_pv.h5"   # intermediate; kept for inspection
CHECKPOINT   = "git_ignore_folder/nyse_adj_checkpoint.pkl"
SLEEP        = 0.05   # seconds between yfinance calls; keeps us well under rate limits
CKPT_EVERY   = 100   # checkpoint every N tickers

# ── Step 1: Load Kaggle data ───────────────────────────────────────────────────
print("Loading Kaggle NYSE CSV...", flush=True)
raw = pd.read_csv(KAGGLE_CSV)
raw["date"] = pd.to_datetime(raw["date"], format="%d-%b-%Y")
raw = raw[raw["volume"] > 0].reset_index(drop=True)   # drop holiday carry-forward rows
print(f"  {len(raw):,} trading-day rows, {raw['ticker'].nunique()} tickers "
      f"({raw['date'].min().date()} → {raw['date'].max().date()})", flush=True)

# Build per-ticker dict for fast O(1) access
print("Indexing per-ticker...", flush=True)
ticker_groups: dict[str, pd.DataFrame] = {}
for ticker, grp in raw.groupby("ticker"):
    g = grp[["date","open","high","low","close","volume"]].set_index("date").sort_index()
    ticker_groups[ticker] = g

all_tickers = sorted(ticker_groups.keys())
print(f"  Ready: {len(all_tickers)} tickers", flush=True)

# yfinance uses dashes for class/warrant suffixes (BRK.B → BRK-B)
def to_yf(ticker: str) -> str:
    return ticker.replace(".", "-")

# ── Step 2: Checkpoint resume ──────────────────────────────────────────────────
adjusted_frames: list[pd.DataFrame] = []
start_idx = 0
split_count = 0
div_count = 0
skip_count = 0

if Path(CHECKPOINT).exists():
    with open(CHECKPOINT, "rb") as f:
        ckpt = pickle.load(f)
    adjusted_frames = ckpt["frames"]
    start_idx  = ckpt["next_idx"]
    split_count = ckpt.get("splits", 0)
    div_count   = ckpt.get("divs", 0)
    skip_count  = ckpt.get("skips", 0)
    print(f"Resuming from checkpoint: {start_idx}/{len(all_tickers)} done "
          f"({len(adjusted_frames)} frames, {split_count} splits, {div_count} divs applied)", flush=True)
else:
    print("No checkpoint — starting fresh", flush=True)

# ── Adjustment logic ───────────────────────────────────────────────────────────
def apply_adjustments(
    df: pd.DataFrame,
    splits: pd.Series,
    dividends: pd.Series,
) -> tuple[pd.DataFrame, int, int]:
    """
    Backward-adjust OHLCV for splits then dividends.

    Processing order is critical:
      Splits first: gives us split-adjusted prices.
      Dividends second: yfinance dividend amounts are already in split-adjusted share terms,
        so we compute adj_factor against the split-adjusted prev_close.

    Returns (adjusted_df, n_splits_applied, n_divs_applied).
    """
    df = df.copy()
    price_cols = ["open", "high", "low", "close"]
    n_splits = 0
    n_divs   = 0

    df_start = df.index.min()
    df_end   = df.index.max()

    # — Splits: oldest to newest —
    for raw_date, ratio in splits.sort_index().items():
        split_date = pd.Timestamp(raw_date).tz_localize(None).normalize()
        # Only care about splits within our data window
        if split_date <= df_start or split_date > df_end:
            continue
        if ratio <= 0 or ratio == 1.0:
            continue
        mask = df.index < split_date
        if not mask.any():
            continue
        df.loc[mask, price_cols] = df.loc[mask, price_cols] / ratio
        df.loc[mask, "volume"]   = df.loc[mask, "volume"]   * ratio
        n_splits += 1

    # — Dividends: oldest to newest (against now-split-adjusted prices) —
    for raw_date, div_amount in dividends.sort_index().items():
        div_date = pd.Timestamp(raw_date).tz_localize(None).normalize()
        if div_date <= df_start or div_date > df_end:
            continue
        if div_amount <= 0:
            continue
        # Close on the last trading day before the ex-div date
        pre = df[df.index < div_date]
        if pre.empty:
            continue
        prev_close = float(pre["close"].iloc[-1])
        if prev_close <= 0 or prev_close <= div_amount:
            continue
        adj_factor = (prev_close - div_amount) / prev_close
        # Sanity: a single dividend shouldn't represent more than 30% of share price
        if not (0.70 < adj_factor < 1.0):
            continue
        mask = df.index < div_date
        df.loc[mask, price_cols] = df.loc[mask, price_cols] * adj_factor
        n_divs += 1

    return df, n_splits, n_divs

# ── Step 3: Fetch adjustments and apply ───────────────────────────────────────
print(f"\nProcessing {len(all_tickers)} tickers (starts at #{start_idx})...", flush=True)

for i, kaggle_t in enumerate(all_tickers[start_idx:], start=start_idx):
    yf_t = to_yf(kaggle_t)
    df   = ticker_groups[kaggle_t]

    try:
        tk = yf.Ticker(yf_t)
        splits    = tk.get_splits()
        dividends = tk.get_dividends()

        if splits is None:    splits    = pd.Series(dtype=float)
        if dividends is None: dividends = pd.Series(dtype=float)

        adj_df, ns, nd = apply_adjustments(df, splits, dividends)
        split_count += ns
        div_count   += nd

        # Rename columns to qlib convention
        adj_df = adj_df.rename(columns={
            "open":   "$open",
            "high":   "$high",
            "low":    "$low",
            "close":  "$close",
            "volume": "$volume",
        })
        adj_df["$factor"]    = np.float32(1.0)
        adj_df["instrument"] = kaggle_t
        adj_df.index.name    = "datetime"
        adjusted_frames.append(
            adj_df.reset_index().set_index(["datetime", "instrument"])
        )

    except Exception as e:
        skip_count += 1
        # Still include unadjusted data for this ticker
        adj_df = df.rename(columns={
            "open":"$open","high":"$high","low":"$low",
            "close":"$close","volume":"$volume"
        })
        adj_df["$factor"]    = np.float32(1.0)
        adj_df["instrument"] = kaggle_t
        adj_df.index.name    = "datetime"
        adjusted_frames.append(
            adj_df.reset_index().set_index(["datetime", "instrument"])
        )

    if (i + 1) % 50 == 0:
        pct = (i + 1) / len(all_tickers) * 100
        print(f"  [{i+1:4d}/{len(all_tickers)}] {pct:5.1f}%  "
              f"splits={split_count}  divs={div_count}  skips={skip_count}", flush=True)

    if (i + 1) % CKPT_EVERY == 0:
        with open(CHECKPOINT, "wb") as f:
            pickle.dump({"frames": adjusted_frames, "next_idx": i + 1,
                         "splits": split_count, "divs": div_count, "skips": skip_count}, f)

    time.sleep(SLEEP)

print(f"\nDone: {len(adjusted_frames)} ticker frames, "
      f"{split_count} splits applied, {div_count} divs applied, {skip_count} errors", flush=True)

# ── Step 4: Combine and cast ───────────────────────────────────────────────────
print("Combining frames...", flush=True)
nyse = pd.concat(adjusted_frames).sort_index()
nyse = nyse[~nyse.index.duplicated(keep="first")]

for col in ["$open", "$high", "$low", "$close", "$factor"]:
    nyse[col] = nyse[col].astype(np.float32)
nyse["$volume"] = nyse["$volume"].astype(np.float32)

nyse_insts = set(nyse.index.get_level_values(1))
print(f"NYSE adjusted: {nyse.shape[0]:,} rows, {len(nyse_insts)} instruments", flush=True)
print(f"Date range: {nyse.index.get_level_values(0).min().date()} → "
      f"{nyse.index.get_level_values(0).max().date()}", flush=True)

# Save intermediate NYSE-only file for inspection
nyse.to_hdf(NYSE_ONLY_H5, key="data", complevel=5)
mb = os.path.getsize(NYSE_ONLY_H5) / 1e6
print(f"Saved intermediate: {NYSE_ONLY_H5}  ({mb:.1f} MB)", flush=True)

# ── Step 5: SHOP verification ──────────────────────────────────────────────────
print("\n=== SHOP verification (split should be gone) ===", flush=True)
try:
    shop_adj = nyse.xs("SHOP", level="instrument").sort_index()
    window = shop_adj.loc["2022-06-15":"2022-07-15"]
    print(window[["$open","$close","$volume"]].to_string(), flush=True)

    pre  = shop_adj.loc[:"2022-06-28", "$close"].iloc[-1]
    post = shop_adj.loc["2022-06-29":, "$close"].iloc[0]
    ratio = pre / post
    print(f"\nPre-split close (adj):  ${pre:.4f}")
    print(f"Post-split close:       ${post:.4f}")
    print(f"Ratio pre/post:         {ratio:.4f}x  (should be ~1.0 if split removed)", flush=True)
except Exception as e:
    print(f"SHOP check failed: {e}", flush=True)

# ── Step 6: Merge with SP500 ───────────────────────────────────────────────────
print(f"\nLoading SP500 from {SP500_H5}...", flush=True)
sp500 = pd.read_hdf(SP500_H5, key="data")
sp500_insts = set(sp500.index.get_level_values(1))
print(f"SP500: {sp500.shape[0]:,} rows, {len(sp500_insts)} instruments", flush=True)

new_only = nyse_insts - sp500_insts
overlap  = nyse_insts & sp500_insts
print(f"Overlap (SP500 wins): {len(overlap)} instruments", flush=True)
print(f"NYSE-only (new):     {len(new_only)} instruments", flush=True)

# SP500 first in concat → SP500 wins on duplicate (date, instrument) pairs
combined = pd.concat([sp500, nyse])
combined = combined[~combined.index.duplicated(keep="first")]
combined = combined.sort_index()
combined_insts = combined.index.get_level_values(1).nunique()

print(f"\nCombined: {combined.shape[0]:,} rows, {combined_insts} instruments", flush=True)
print(f"Date range: {combined.index.get_level_values(0).min().date()} → "
      f"{combined.index.get_level_values(0).max().date()}", flush=True)

# ── Step 7: Backup SP500 and write combined ────────────────────────────────────
backup = SP500_H5.replace(".h5", "_sp500_backup.h5")
shutil.copy(SP500_H5, backup)
print(f"\nSP500 backed up → {backup}", flush=True)

combined.to_hdf(OUT_H5, key="data", complevel=5)
mb = os.path.getsize(OUT_H5) / 1e6
print(f"Saved combined → {OUT_H5}  ({mb:.1f} MB)", flush=True)

# Clean up checkpoint
if Path(CHECKPOINT).exists():
    os.remove(CHECKPOINT)

print("\n=== COMPLETE ===", flush=True)
print(f"  SP500 instruments (2010-2026): {len(sp500_insts)}", flush=True)
print(f"  NYSE-only instruments (2019-2024): {len(new_only)}", flush=True)
print(f"  Total in daily_pv.h5: {combined_insts}", flush=True)
