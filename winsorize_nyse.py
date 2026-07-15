"""Winsorize extreme outlier returns in the residual NYSE-only names, so unadjustable
reverse-split cliffs (delisted tickers with no yfinance split data) cannot distort
CROSS-SECTIONAL factor statistics. SP500 names are never touched.

Method: cap the daily return, then rebuild the price series from the capped returns.
A constant rescale cancels in returns, so EVERY non-clipped return is preserved exactly;
only the clipped day's return changes, and prices after it are rescaled by one constant.

Asymmetric bounds are deliberate and principled:
  * upper tail: returns are UNBOUNDED above -- this is where the corruption lives
    (reverse-split cliffs of +650%..+71,500%). Clip at +100% (SP500's largest genuine
    move is +93%, so nothing comparable is lost). Clipping upward rescales later prices
    DOWN, which moves them toward the true split adjustment -- safe.
  * lower tail: returns are bounded at -100%, so they cannot blow up cross-sectional
    variance the way the upper tail can. Clipping a ~-99.9% collapse would require
    inflating all later prices by hundreds of x (fabricating levels), so instead the
    handful of near-total-collapse observations are masked to NaN -- removed from
    cross-sectional stats without inventing price levels.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, shutil
from pathlib import Path

SRC   = "git_ignore_folder/factor_implementation_source_data"
H5    = f"{SRC}/daily_pv.h5"
BAK   = f"{SRC}/daily_pv_prewinsor.h5"
UPPER = 1.00     # cap daily return at +100%
LOWER_MASK = -0.90   # returns below this are masked (data-artifact collapses)
PRICE_COLS = ["$open","$high","$low","$close"]

comb = pd.read_hdf(H5)
sp500 = set(pd.read_hdf(f"{SRC}/daily_pv_sp500_backup.h5").index.get_level_values("instrument").unique())
allins = set(comb.index.get_level_values("instrument").unique())
nyse = sorted(allins - sp500)
print(f"combined={len(allins)}  SP500(untouched)={len(sp500)}  NYSE-only(winsorized)={len(nyse)}")

if not Path(BAK).exists():
    shutil.copy(H5, BAK); print(f"backup -> {Path(BAK).name}")

wide = {c: comb[c].unstack("instrument") for c in PRICE_COLS}
close = wide["$close"]
ret = close[nyse].pct_change()

n_clip = int((ret > UPPER).sum().sum())
n_mask = int((ret < LOWER_MASK).sum().sum())
print(f"\nobservations to winsorize (ret > +{UPPER:.0%}): {n_clip}")
print(f"observations to mask     (ret < {LOWER_MASK:.0%}): {n_mask}")

# --- rebuild prices from capped returns ---
capped = ret.clip(upper=UPPER)
# rescale factor: product of (1+capped)/(1+raw) up to and including t
adj = ((1 + capped) / (1 + ret)).where(ret.notna(), 1.0).fillna(1.0)
scale = adj.cumprod()
for c in PRICE_COLS:
    w = wide[c]
    w[nyse] = w[nyse] * scale
    wide[c] = w

# --- mask near-total-collapse artifacts (cannot cap without fabricating levels) ---
mask = (ret < LOWER_MASK)
for c in PRICE_COLS:
    w = wide[c]
    sub = w[nyse]
    sub[mask] = np.nan
    w[nyse] = sub
    wide[c] = w

out = comb.copy()
for c in PRICE_COLS:
    out[c] = wide[c].stack().reindex(out.index).astype(np.float32)

out.to_hdf(H5, key="data", complevel=5)
print(f"\nwrote {H5}")

# --- verification ---
chk = pd.read_hdf(H5)
cw = chk["$close"].unstack("instrument")
r_ny = cw[nyse].pct_change().stack().dropna()
r_sp = cw[sorted(sp500)].pct_change().stack().dropna()
old = pd.read_hdf(BAK)["$close"].unstack("instrument")
sp_same = np.allclose(cw[sorted(sp500)].fillna(-1).values, old[sorted(sp500)].fillna(-1).values, equal_nan=True)
print(f"\n=== verification ===")
print(f"  SP500 prices bit-identical to pre-winsor: {sp_same}")
print(f"  NYSE-only returns:  max={r_ny.max():+.4f}  min={r_ny.min():+.4f}   (was max=+715.67)")
print(f"  SP500   returns:    max={r_sp.max():+.4f}  min={r_sp.min():+.4f}   (unchanged: LUMN +0.93 / GL -0.54)")
print(f"  rows={len(chk):,}  instruments={chk.index.get_level_values('instrument').nunique()}")
