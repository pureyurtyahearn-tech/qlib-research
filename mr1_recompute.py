"""Recompute the mean-reversion family (and a momentum control group) from PRIOR-RUN
generated factor code, executed against the CORRECTED (split-fixed + winsorized) dataset.
Reuses the LLM-generated code; only the values are recomputed."""
import warnings; warnings.filterwarnings("ignore")
import re, shutil, subprocess, sys, os
from pathlib import Path
import pandas as pd

WS = Path("git_ignore_folder/RD-Agent_workspace")
DATA = Path("git_ignore_folder/factor_implementation_source_data/daily_pv.h5")
TMP = Path("git_ignore_folder/_mr_recompute"); TMP.mkdir(exist_ok=True)
OUT = Path("git_ignore_folder/_mr_factors"); OUT.mkdir(exist_ok=True)

MEANREV = ["composite_reversal_zscore","reversal_20d","price_zscore_20d","rsi_14d","rsi_7d",
           "rsi_14d_centered","williams_r_10d","stochastic_k_14d","z_ema_dev_5d",
           "volatility_normalized_reversal_1d","ts_percentile_rank_5d_return_20d",
           "price_channel_position_20d","kaufman_efficiency_ratio_10d"]
MOMENTUM = ["vw_momentum_5d","vol_adj_momentum_10d","price_trend_slope_10d","obv_momentum_20d",
            "vpt_momentum_10d","PriceToHigh20","VolumeWeightedMom10"]

# map factor name -> the workspace factor.py that produces it
def find_src(name):
    for f in WS.glob("*/factor.py"):
        t = f.read_text(errors="ignore")
        if re.search(rf"to_frame\(name=['\"]{re.escape(name)}['\"]", t):
            return f
    return None

if not (TMP/"daily_pv.h5").exists():
    shutil.copy(DATA, TMP/"daily_pv.h5")

def recompute(name):
    dst = OUT/f"{name}.h5"
    if dst.exists():
        return pd.read_hdf(dst).iloc[:,0]
    src = find_src(name)
    if src is None:
        print(f"  {name:38} NO CODE FOUND"); return None
    shutil.copy(src, TMP/"factor.py")
    (TMP/"result.h5").unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "factor.py"], cwd=TMP, capture_output=True, text=True,
                       env={**os.environ, "PYTHONUTF8":"1"})
    if not (TMP/"result.h5").exists():
        print(f"  {name:38} FAILED: {r.stderr.strip().splitlines()[-1][:70] if r.stderr else '?'}")
        return None
    s = pd.read_hdf(TMP/"result.h5")
    s = s.iloc[:,0] if s.ndim > 1 else s
    s.to_frame(name).to_hdf(dst, key="f", complevel=5)
    print(f"  {name:38} OK  n={len(s):,}")
    return s

if __name__ == "__main__":
    print("=== MEAN-REVERSION family ===")
    for n in MEANREV: recompute(n)
    print("\n=== MOMENTUM control group ===")
    for n in MOMENTUM: recompute(n)
    print(f"\nsaved factor values -> {OUT}/")
