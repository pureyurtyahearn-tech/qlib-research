"""Shared loaders for the NASDAQ-only fundamentals factor test (nqf1_dev / nqf2_holdout).

Universe: the 3,218 small/mid NASDAQ-only names (never S&P 500) that carry SF1 fundamentals
(fundamentals_nasdaq_daily.h5, 0 look-ahead leaks). Prices from sep_nasdaq_panel.h5.

PIT / tradeability (pre-registered, fixed before seeing data):
  elig[t,i] = finite close AND causal trailing-63d median dollar-volume >= $2M.
  The ext6 pandas simulator force-exits any holding that loses eligibility on date t --
  the same PIT discipline as PITTopkDropoutStrategy (defeats the ghost-position bug), and
  the SAME engine used for every SP500 holdout test, so results are directly comparable.

Reuses sn_common's generic metric functions (rank_ic, placebo_sd, quintiles, phased_book,
evaluate, print_eval) -- all operate on plain arrays, universe-agnostic.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import sn_common as C
from ext6_momentum_full import ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
LIQ_GATE = 2e6      # $2M causal trailing-median dollar volume
LIQ_WIN = 63        # ~3 trading months


def load_nasdaq():
    p = pd.read_hdf(SH / "sep_nasdaq_panel.h5")
    fund = pd.read_hdf(SH / "fundamentals_nasdaq_daily.h5")
    names = sorted(set(fund.index.get_level_values("instrument").unique()))
    close = p["$close"].unstack("ticker").sort_index()
    vol = p["$volume"].unstack("ticker").reindex(index=close.index)
    names = [n for n in names if n in close.columns]
    close = close[names]
    vol = vol[names]
    return close, vol, fund


def build_elig(close, vol):
    """Causal trailing-63d median dollar-volume gate + finite price."""
    dv = (close * vol).astype("float32")
    trail = dv.rolling(LIQ_WIN, min_periods=20).median().shift(1)   # causal (t-1 and earlier)
    elig = (trail >= LIQ_GATE) & np.isfinite(close)
    return elig.fillna(False)


def make_windows(close):
    retv_full = close.pct_change()
    fwd_full = (close.shift(-22) / close.shift(-1) - 1)              # 21d fwd, skip today
    return retv_full, fwd_full


def slice_window(close, elig, retv_full, fwd_full, s, e):
    w = (close.index >= s) & (close.index <= e)
    d = close.index[w]
    eligv = elig.loc[d].values & np.isfinite(close.loc[d].values)
    retv = retv_full.loc[d].values
    fwd = fwd_full.loc[d].values
    ew = np.array([np.nanmean(np.where(eligv[t], retv[t], np.nan)) for t in range(len(d))])
    return d, eligv, retv, fwd, ew


def factor_wide(fund, factor, index, cols):
    return (fund[factor].unstack("instrument")
            .reindex(index=index, columns=cols).astype(float))
