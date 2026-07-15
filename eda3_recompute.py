"""EDA part 3: independently recompute vwap_deviation_10d and amihud_illiquidity_10d
from raw daily_pv.h5, compare to the result.h5 RD-Agent's generated code produced."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

SRC = "git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
df = pd.read_hdf(SRC).sort_index()
close  = df["$close"].unstack("instrument")
volume = df["$volume"].unstack("instrument")

def compare(name, mine, produced_path):
    prod = pd.read_hdf(produced_path)
    prod = prod.iloc[:, 0] if prod.ndim > 1 else prod
    mine = mine.dropna()
    mine.index.names = ["datetime", "instrument"]
    # align on shared index
    idx = mine.index.intersection(prod.index)
    a = mine.reindex(idx).astype(float)
    b = prod.reindex(idx).astype(float)
    diff = (a - b).abs()
    denom = b.abs().replace(0, np.nan)
    rel = (diff / denom).dropna()
    print(f"\n=== {name} ===")
    print(f"  my rows={len(mine):,}  produced rows={len(prod):,}  shared index={len(idx):,}")
    print(f"  max abs diff   = {diff.max():.3e}")
    print(f"  mean abs diff  = {diff.mean():.3e}")
    print(f"  max rel diff   = {rel.max():.3e}")
    print(f"  Pearson corr   = {a.corr(b):.10f}")
    print(f"  fraction within 1e-9 abs: {100*(diff<1e-9).mean():.4f}%")
    print(f"  fraction within 1e-6 abs: {100*(diff<1e-6).mean():.4f}%")

# --- vwap_deviation_10d : VWAP over [t-10, t-1], vs close_{t-1} ---
dollar_vol = close * volume
vwap = (dollar_vol.rolling(10, min_periods=10).sum().shift(1)
        / volume.rolling(10, min_periods=10).sum().shift(1))
vwap_dev = (close.shift(1) - vwap) / vwap
compare("vwap_deviation_10d", vwap_dev.stack(),
        "git_ignore_folder/RD-Agent_workspace/31fdccc307aa4633a18579f0b29ef923/result.h5")

# --- amihud_illiquidity_10d : mean over [t-10,t-1] of |ret|/(close*vol) ---
daily_illiq = close.pct_change().abs() / (close * volume)
amihud = daily_illiq.shift(1).rolling(10, min_periods=10).mean()
compare("amihud_illiquidity_10d", amihud.stack(),
        "git_ignore_folder/RD-Agent_workspace/073a8026258c4efd820d6354e3aed833/result.h5")
