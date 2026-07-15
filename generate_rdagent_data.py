"""
Generate daily_pv.h5 for RD-Agent factor code covering 2010-2026.
Uses yfinance (auto-adjusted, fully consistent) for the full period.
Active S&P 500 instruments (~505 tickers with the 2099-12-31 sentinel in sp500.txt).
This produces the SP500-only base that fix_and_build_nyse.py merges NYSE into.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf, os
from pathlib import Path

FULL_START = "2010-01-01"
FULL_END   = "2026-06-30"
DATA_DIR   = "git_ignore_folder/factor_implementation_source_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Active S&P 500 tickers from qlib instrument list
sp500_file = Path.home() / ".qlib" / "qlib_data" / "us_data" / "instruments" / "sp500.txt"
with open(sp500_file) as f:
    active_tickers = [ln.split("\t")[0].strip() for ln in f if "2099-12-31" in ln]
print(f"Active S&P 500 tickers: {len(active_tickers)}")

# Some tickers were renamed; yfinance handles most automatically,
# but FB→META needs explicit mapping since 'FB' no longer trades
TICKER_MAP = {"FB": "META"}
yf_tickers = [TICKER_MAP.get(t, t) for t in active_tickers]
qlib_to_yf = dict(zip(active_tickers, yf_tickers))

# Download all at once in batches to avoid memory issues
BATCH = 20
all_frames = []

for i in range(0, len(yf_tickers), BATCH):
    batch = yf_tickers[i:i+BATCH]
    print(f"Downloading batch {i//BATCH + 1}/{(len(yf_tickers)-1)//BATCH + 1}: {batch[:5]}...")
    df = yf.download(
        batch,
        start=FULL_START,
        end=FULL_END,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if df.empty:
        print(f"  WARNING: empty batch")
        continue

    # Normalise MultiIndex columns -> (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        for ticker in batch:
            try:
                tk_df = df[ticker].copy()
            except KeyError:
                continue
            tk_df = tk_df.rename(columns={
                "Open":   "$open",
                "High":   "$high",
                "Low":    "$low",
                "Close":  "$close",
                "Volume": "$volume",
            })
            # Find the qlib ticker for this yf ticker
            qlib_tick = next((q for q, y in qlib_to_yf.items() if y == ticker), ticker)
            tk_df["$factor"] = 1.0   # prices fully adjusted, factor=1
            tk_df.index = pd.to_datetime(tk_df.index)
            valid = tk_df[["$open","$high","$low","$close","$volume","$factor"]].dropna(how="all")
            if valid.empty:
                continue
            valid.index.name = "datetime"
            valid["instrument"] = qlib_tick
            all_frames.append(valid.reset_index().set_index(["datetime","instrument"]))
    else:
        # Single ticker download
        ticker = batch[0]
        tk_df = df.rename(columns={
            "Open":"$open","High":"$high","Low":"$low",
            "Close":"$close","Volume":"$volume"
        })
        qlib_tick = next((q for q, y in qlib_to_yf.items() if y == ticker), ticker)
        tk_df["$factor"] = 1.0
        tk_df.index = pd.to_datetime(tk_df.index)
        valid = tk_df[["$open","$high","$low","$close","$volume","$factor"]].dropna(how="all")
        if not valid.empty:
            valid.index.name = "datetime"
            valid["instrument"] = qlib_tick
            all_frames.append(valid.reset_index().set_index(["datetime","instrument"]))

print(f"\nCombining {len(all_frames)} instrument frames...")
if not all_frames:
    raise RuntimeError("No data downloaded!")

combined = pd.concat(all_frames).sort_index()
combined = combined[~combined.index.duplicated(keep="first")]
print(f"Combined shape: {combined.shape}")
print(f"Date range: {combined.index.get_level_values(0).min()} to {combined.index.get_level_values(0).max()}")
print(f"Instruments: {combined.index.get_level_values(1).nunique()}")

out_path = f"{DATA_DIR}/daily_pv.h5"
combined.to_hdf(out_path, key="data", complevel=5)
print(f"\nSaved: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
