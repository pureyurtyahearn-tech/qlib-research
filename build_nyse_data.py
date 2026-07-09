"""
Download 3168 NYSE tickers from yfinance (split/dividend-adjusted) and merge with
the existing SP500 daily_pv.h5.

Source ticker universe: Kaggle 'mousemover/quant-finance-nyse-5-years'
Date range downloaded: 2019-01-01 → 2026-06-30 (adjusted, yfinance)

Merge rule: SP500 takes priority on any overlapping (date, instrument) pair.
NYSE-only instruments are added with history starting from 2019.
"""
import warnings; warnings.filterwarnings("ignore")
import time, pickle, shutil, os
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
KAGGLE_CSV  = "/home/codespace/.cache/kagglehub/datasets/mousemover/quant-finance-nyse-5-years/versions/1/NYSE_fully_cleaned_2019_2024.csv"
SP500_H5    = "git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
OUT_H5      = "git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
CHECKPOINT  = "git_ignore_folder/nyse_download_checkpoint.pkl"

FULL_START  = "2019-01-01"
FULL_END    = "2026-06-30"
BATCH       = 20
SLEEP       = 0.3   # seconds between batches; yfinance handles bursts fine
CKPT_EVERY  = 10   # save checkpoint every N batches

# ── Step 1: ticker list from Kaggle CSV ───────────────────────────────────────
print("Loading ticker list from Kaggle CSV...")
kaggle_tickers = sorted(
    pd.read_csv(KAGGLE_CSV, usecols=["ticker"])["ticker"].unique().tolist()
)
print(f"  {len(kaggle_tickers)} unique tickers")

# yfinance uses dashes for share classes / warrants; Kaggle uses dots
# Store instrument name as the original Kaggle ticker (dots preserved)
# but call yfinance with dashes so the API resolves correctly
def to_yf(ticker: str) -> str:
    return ticker.replace(".", "-")

# Build bidirectional maps: yf_ticker → kaggle_ticker (for result labelling)
yf_tickers  = [to_yf(t) for t in kaggle_tickers]
yf_to_kaggle = dict(zip(yf_tickers, kaggle_tickers))
batches = [
    (kaggle_tickers[i:i+BATCH], yf_tickers[i:i+BATCH])
    for i in range(0, len(kaggle_tickers), BATCH)
]
print(f"  {len(batches)} batches of {BATCH}")

# ── Step 2: checkpoint resume ──────────────────────────────────────────────────
done_frames: list[pd.DataFrame] = []
start_batch = 0

if Path(CHECKPOINT).exists():
    with open(CHECKPOINT, "rb") as f:
        ckpt = pickle.load(f)
    done_frames = ckpt["frames"]
    start_batch = ckpt["next_batch"]
    pct = start_batch * BATCH / len(kaggle_tickers) * 100
    print(f"Resuming from checkpoint: batch {start_batch}/{len(batches)} "
          f"({pct:.0f}% done, {len(done_frames)} frames so far)")
else:
    print("No checkpoint found — starting fresh")

# ── Step 3: download ───────────────────────────────────────────────────────────
print(f"\nDownloading {len(kaggle_tickers)} tickers, "
      f"{FULL_START} → {FULL_END}, auto_adjust=True\n")

def extract_frames(batch_kaggle, batch_yf, df):
    """Pull per-ticker DataFrames from a yfinance multi-ticker download."""
    frames = []
    for yf_t, kag_t in zip(batch_yf, batch_kaggle):
        try:
            tk = df[yf_t]
        except KeyError:
            continue
        if tk.empty:
            continue
        tk = tk.rename(columns={
            "Open":  "$open",
            "High":  "$high",
            "Low":   "$low",
            "Close": "$close",
            "Volume":"$volume",
        })
        # Drop rows where close is NaN (non-trading or delisted)
        tk = tk[[c for c in ["$open","$high","$low","$close","$volume"] if c in tk.columns]]
        tk = tk.dropna(subset=["$close"])
        if tk.empty:
            continue
        tk["$factor"]    = np.float32(1.0)
        tk["instrument"] = kag_t
        tk.index         = pd.to_datetime(tk.index)
        tk.index.name    = "datetime"
        frames.append(tk.reset_index().set_index(["datetime", "instrument"]))
    return frames

failed_batches = []
for i, (kag_batch, yf_batch) in enumerate(batches[start_batch:], start=start_batch):
    label = f"{kag_batch[0]}…{kag_batch[-1]}"
    print(f"  Batch {i+1:3d}/{len(batches)} [{label}]", end=" ", flush=True)
    try:
        df = yf.download(
            yf_batch,
            start=FULL_START,
            end=FULL_END,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if df.empty:
            print("→ empty")
            failed_batches.append(kag_batch)
        else:
            frames = extract_frames(kag_batch, yf_batch, df)
            done_frames.extend(frames)
            print(f"→ {len(frames)}/{len(kag_batch)} tickers")
    except Exception as e:
        print(f"→ ERROR: {e}")
        failed_batches.append(kag_batch)

    # Checkpoint
    if (i + 1) % CKPT_EVERY == 0 or i == len(batches) - 1:
        with open(CHECKPOINT, "wb") as f:
            pickle.dump({"frames": done_frames, "next_batch": i + 1}, f)
        print(f"    [checkpoint @ batch {i+1}]")

    time.sleep(SLEEP)

# ── Step 4: retry failed batches individually ──────────────────────────────────
if failed_batches:
    print(f"\nRetrying {sum(len(b) for b in failed_batches)} tickers from {len(failed_batches)} failed batches...")
    for kag_batch in failed_batches:
        for kag_t in kag_batch:
            yf_t = to_yf(kag_t)
            try:
                df = yf.download(yf_t, start=FULL_START, end=FULL_END,
                                 auto_adjust=True, group_by="ticker",
                                 progress=False)
                if not df.empty:
                    frames = extract_frames([kag_t], [yf_t], df)
                    done_frames.extend(frames)
                    if frames:
                        print(f"  {kag_t} → OK")
            except Exception:
                pass
            time.sleep(0.2)

# ── Step 5: build NYSE DataFrame ───────────────────────────────────────────────
print(f"\nCombining {len(done_frames)} per-ticker frames...")
if not done_frames:
    raise RuntimeError("No data downloaded — check network or ticker list")

nyse = pd.concat(done_frames).sort_index()
nyse = nyse[~nyse.index.duplicated(keep="first")]

# Cast to float32 to match SP500 file
for col in ["$open", "$high", "$low", "$close", "$factor"]:
    nyse[col] = nyse[col].astype(np.float32)
nyse["$volume"] = nyse["$volume"].astype(np.float32)

nyse_insts = set(nyse.index.get_level_values(1))
print(f"NYSE: {nyse.shape[0]:,} rows, {len(nyse_insts)} instruments")
print(f"NYSE date range: {nyse.index.get_level_values(0).min().date()} → "
      f"{nyse.index.get_level_values(0).max().date()}")

# ── Step 6: merge with existing SP500 ─────────────────────────────────────────
print(f"\nLoading SP500 from {SP500_H5} ...")
sp500 = pd.read_hdf(SP500_H5, key="data")
sp500_insts = set(sp500.index.get_level_values(1))
print(f"SP500: {sp500.shape[0]:,} rows, {len(sp500_insts)} instruments")

new_only  = nyse_insts - sp500_insts
overlap   = nyse_insts & sp500_insts
print(f"\nOverlap (SP500 wins):     {len(overlap):4d} instruments")
print(f"NYSE-only (new to add):  {len(new_only):4d} instruments")

# concat: SP500 first → SP500 wins on any duplicate (date, instrument) index
combined = pd.concat([sp500, nyse])
combined = combined[~combined.index.duplicated(keep="first")]
combined = combined.sort_index()
combined_insts = combined.index.get_level_values(1).nunique()

print(f"\nCombined: {combined.shape[0]:,} rows, {combined_insts} instruments")
print(f"Date range: {combined.index.get_level_values(0).min().date()} → "
      f"{combined.index.get_level_values(0).max().date()}")

# ── Step 7: backup SP500 and write combined ────────────────────────────────────
backup = SP500_H5.replace(".h5", "_sp500_backup.h5")
shutil.copy(SP500_H5, backup)
print(f"\nSP500 backed up → {backup}")

combined.to_hdf(OUT_H5, key="data", complevel=5)
mb = os.path.getsize(OUT_H5) / 1e6
print(f"Saved combined → {OUT_H5}  ({mb:.1f} MB)")

# ── Step 8: clean up checkpoint ───────────────────────────────────────────────
if Path(CHECKPOINT).exists():
    os.remove(CHECKPOINT)
    print("Checkpoint removed")

print("\nDone.")
print(f"  SP500 instruments (2010-2026): {len(sp500_insts)}")
print(f"  NYSE-only instruments (2019-2026): {len(new_only)}")
print(f"  Total in daily_pv.h5: {combined_insts}")
