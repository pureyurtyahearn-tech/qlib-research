"""
Convert NASDAQ_2020-2024 daily CSVs to the HDF5 format expected by RD-Agent.

Source:  nasdaq-data/NASDAQ_2020-2024/NASDAQ_YYYYMMDD.csv
         Columns (no header): ticker, date, open, high, low, close, volume
         Date format: "02-Jan-2020"

Target:  git_ignore_folder/nasdaq_factor_data/daily_pv.h5
         MultiIndex (datetime, instrument), sorted datetime-first
         Columns: $open, $close, $high, $low, $volume, $factor — all float32
         $factor = 1.0 (raw unadjusted prices; no split/dividend data in source)
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

SRC_DIR = "/workspaces/qlib-research/nasdaq-data/NASDAQ_2020-2024"
OUT_DIR = "/workspaces/qlib-research/git_ignore_folder/nasdaq_factor_data"
OUT_FILE = os.path.join(OUT_DIR, "daily_pv.h5")

os.makedirs(OUT_DIR, exist_ok=True)

CSV_COLS = ["instrument", "date_str", "$open", "$high", "$low", "$close", "$volume"]

files = sorted(glob.glob(os.path.join(SRC_DIR, "NASDAQ_*.csv")))
print(f"Found {len(files)} CSV files spanning {os.path.basename(files[0])} → {os.path.basename(files[-1])}")

chunks = []
for i, path in enumerate(files):
    if i % 100 == 0:
        print(f"  Reading {i}/{len(files)}: {os.path.basename(path)}")
    try:
        df = pd.read_csv(path, header=None, names=CSV_COLS, dtype=str)
        # Parse date from first row (same for all rows in file — but parse per row to be safe)
        df["datetime"] = pd.to_datetime(df["date_str"], format="%d-%b-%Y")
        df = df.drop(columns=["date_str"])
        chunks.append(df)
    except Exception as e:
        print(f"  WARNING: skipped {path}: {e}")

print(f"\nConcatenating {len(chunks)} day DataFrames...")
full = pd.concat(chunks, ignore_index=True)
print(f"  Raw shape: {full.shape}")

# Cast price/volume columns to float32
for col in ["$open", "$high", "$low", "$close", "$volume"]:
    full[col] = pd.to_numeric(full[col], errors="coerce").astype("float32")

# Add $factor = 1.0 (unadjusted data)
full["$factor"] = np.float32(1.0)

# Reorder columns to match target schema: $open, $close, $high, $low, $volume, $factor
full = full[["datetime", "instrument", "$open", "$close", "$high", "$low", "$volume", "$factor"]]

# Drop rows with NaN in price columns
before = len(full)
full = full.dropna(subset=["$open", "$close", "$high", "$low", "$volume"])
if before != len(full):
    print(f"  Dropped {before - len(full)} rows with NaN prices")

# Set MultiIndex (datetime, instrument) and sort
full = full.set_index(["datetime", "instrument"]).sort_index()
print(f"  Final shape: {full.shape}")
print(f"  Date range: {full.index.get_level_values('datetime').min()} → {full.index.get_level_values('datetime').max()}")
print(f"  Unique instruments: {full.index.get_level_values('instrument').nunique()}")
print(f"  Dtypes:\n{full.dtypes}")
print(f"\nSample:\n{full.head(5)}")

print(f"\nSaving to {OUT_FILE} ...")
full.to_hdf(OUT_FILE, key="data", mode="w", complevel=9, complib="blosc")
size_mb = os.path.getsize(OUT_FILE) / 1e6
print(f"Done. File size: {size_mb:.1f} MB")
