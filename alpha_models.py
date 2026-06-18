"""
Qlib Alpha Models demo.
Uses Qlib expression engine to compute factors, evaluates IC (Information Coefficient).
"""
import warnings
warnings.filterwarnings("ignore")

import qlib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
from qlib.data import D

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "JPM",  "BAC",  "XOM",  "JNJ",  "PG",
            "NVDA", "V",    "MA",   "HD"]
START, END = "2017-01-01", "2020-11-08"

# ── 1. Fetch factors via Qlib expression engine ──────────────────────────────
fields = [
    "$close",
    "$volume",
    "Ref($close, 1)",          # yesterday's close
    "Mean($close, 5)",         # 5-day MA
    "Mean($close, 20)",        # 20-day MA
    "Std($close, 20)",         # 20-day std
    "($close - Mean($close, 20)) / (Std($close, 20) + 1e-8)",  # Bollinger z-score
    "($close / Ref($close, 20)) - 1",                           # 20-day momentum
    "($close / Ref($close, 5))  - 1",                           # 5-day momentum
    "($close / Ref($close, 60)) - 1",                           # 60-day momentum
    "Mean($volume, 5) / (Mean($volume, 20) + 1e-8)",           # volume ratio
]
names = ["close", "volume", "prev_close",
         "ma5", "ma20", "std20",
         "bb_z", "mom20", "mom5", "mom60",
         "vol_ratio"]

df = D.features(UNIVERSE, fields, start_time=START, end_time=END, freq="day")
df.columns = names
print(f"Factor data shape: {df.shape}")
print(df.head(3).to_string())

# ── 2. Forward return (target) ────────────────────────────────────────────────
close_df = {}
for t in UNIVERSE:
    close_df[t] = df.xs(t, level="instrument")["close"]
px_df  = pd.DataFrame(close_df)
fwd1   = px_df.pct_change().shift(-1)   # 1-day forward return

# ── 3. IC analysis per factor ─────────────────────────────────────────────────
factor_cols = ["bb_z", "mom5", "mom20", "mom60", "vol_ratio"]

ic_results = {}
for fcol in factor_cols:
    factor_df = {}
    for t in UNIVERSE:
        factor_df[t] = df.xs(t, level="instrument")[fcol]
    fac_df = pd.DataFrame(factor_df)

    # Cross-sectional Spearman IC per day
    daily_ic = []
    for date in fac_df.index:
        row_f = fac_df.loc[date].dropna()
        row_r = fwd1.loc[date].dropna() if date in fwd1.index else pd.Series()
        common = row_f.index.intersection(row_r.index)
        if len(common) < 4:
            continue
        rho, _ = spearmanr(row_f[common], row_r[common])
        daily_ic.append(rho)

    ic_series = pd.Series(daily_ic, dtype=float)
    ic_results[fcol] = {
        "Mean IC" : ic_series.mean(),
        "ICIR"    : ic_series.mean() / (ic_series.std() + 1e-9),
        "IC>0"    : (ic_series > 0).mean(),
    }

ic_df = pd.DataFrame(ic_results).T
print("\n=== Cross-sectional Spearman IC (1-day fwd return) ===")
print(ic_df.to_string(float_format="{:.4f}".format))

# ── 4. Composite alpha: equal-weight factors with positive mean IC ─────────────
good_factors = ic_df[ic_df["Mean IC"] > 0].index.tolist()
print(f"\nFactors with positive IC: {good_factors}")

if good_factors:
    composite_frames = []
    for t in UNIVERSE:
        row = df.xs(t, level="instrument")[good_factors]
        # Cross-sectionally rank each factor, average
        ranked = row.rank(pct=True)
        composite_frames.append(ranked.mean(axis=1).rename(t))
    composite_df = pd.DataFrame(composite_frames).T

    # IC of composite
    comp_ic = []
    for date in composite_df.index:
        row_f = composite_df.loc[date].dropna()
        row_r = fwd1.loc[date].dropna() if date in fwd1.index else pd.Series()
        common = row_f.index.intersection(row_r.index)
        if len(common) < 4:
            continue
        rho, _ = spearmanr(row_f[common], row_r[common])
        comp_ic.append(rho)
    comp_ic_s = pd.Series(comp_ic)
    print(f"\nComposite Alpha IC:  {comp_ic_s.mean():.4f}")
    print(f"Composite Alpha ICIR: {comp_ic_s.mean()/comp_ic_s.std():.4f}")
    print(f"Composite IC>0:      {(comp_ic_s>0).mean():.2%}")
