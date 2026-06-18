"""
Mean-reversion backtest (long-only) using Qlib.
Signal: 5-day z-score of close returns
Long the most oversold stock (z < -1) each day
Execution: buy at next close, sell at close 5 days later
Period: 2015-01-01 to 2020-11-10 (full sample range)
"""
import warnings
warnings.filterwarnings("ignore")

import qlib
import pandas as pd
import numpy as np

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
from qlib.data import D

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START    = "2015-01-01"
END      = "2020-11-10"

# ── 1. Fetch close prices ────────────────────────────────────────────────────
df = D.features(UNIVERSE, ["$close"], start_time=START, end_time=END, freq="day")
df.columns = ["close"]

prices = {}
for t in UNIVERSE:
    prices[t] = df.xs(t, level="instrument").sort_index()["close"]
px_df = pd.DataFrame(prices)

# ── 2. Compute z-score signal ────────────────────────────────────────────────
ret5     = px_df.pct_change(5)
roll_std = ret5.rolling(20).std()
z_df     = ret5 / roll_std

# ── 3. Forward returns (signal at close[t] → trade at close[t+1], exit at close[t+6])
#    Simplified: use 1-day forward close-to-close return
fwd_ret  = px_df.pct_change().shift(-1)          # return from t to t+1

# Align
idx = z_df.dropna(how="all").index.intersection(fwd_ret.dropna(how="all").index)
z   = z_df.loc[idx]
fwd = fwd_ret.loc[idx]

# ── 4. Long-short portfolio ─────────────────────────────────────────────────
# Long: bottom quintile z (most oversold); Short: top quintile z (most overbought)
pos = pd.DataFrame(0.0, index=z.index, columns=z.columns)

for date in z.index:
    row = z.loc[date].dropna()
    if len(row) < 3:
        continue
    q_lo = row.quantile(0.2)
    q_hi = row.quantile(0.8)
    for t in row.index:
        if row[t] <= q_lo:
            pos.loc[date, t] = 1.0
        elif row[t] >= q_hi:
            pos.loc[date, t] = -1.0

# Normalize
row_abs = pos.abs().sum(axis=1).replace(0, np.nan)
pos     = pos.div(row_abs, axis=0).fillna(0)

# ── 5. Daily P&L ─────────────────────────────────────────────────────────────
daily_pnl = (pos * fwd).sum(axis=1)
cumret    = (1 + daily_pnl).cumprod()

# Long-only benchmark (equally weighted)
bah_cum = (1 + fwd.mean(axis=1)).cumprod()

# ── 6. Metrics ────────────────────────────────────────────────────────────────
n          = len(daily_pnl)
total_ret  = cumret.iloc[-1] - 1
ann_ret    = (1 + total_ret) ** (252 / n) - 1
vol        = daily_pnl.std() * np.sqrt(252)
sharpe     = ann_ret / vol if vol > 0 else np.nan
dd         = (cumret / cumret.cummax() - 1).min()
win_rate   = (daily_pnl > 0).mean()
bah_total  = bah_cum.iloc[-1] - 1

print("=" * 52)
print("MEAN-REVERSION BACKTEST  (2015-01-01 → 2020-11-10)")
print("L/S Quintile | 5-day z-score | Close-to-close")
print("=" * 52)
print(f"Total Return         : {total_ret:>8.2%}")
print(f"Annual Return        : {ann_ret:>8.2%}")
print(f"Annual Volatility    : {vol:>8.2%}")
print(f"Sharpe Ratio         : {sharpe:>8.2f}")
print(f"Max Drawdown         : {dd:>8.2%}")
print(f"Win Rate             : {win_rate:>8.2%}")
print(f"Trading Days         : {n}")
print(f"EW Benchmark Return  : {bah_total:>8.2%}")
print()
print("Yearly Returns (Strategy vs EW Benchmark):")
yr_strat = daily_pnl.groupby(daily_pnl.index.year).apply(lambda x: (1+x).prod()-1)
yr_bah   = fwd.mean(axis=1).groupby(fwd.index.year).apply(lambda x: (1+x).prod()-1)
yr_df    = pd.DataFrame({"Strategy": yr_strat, "EW_Bench": yr_bah})
print(yr_df.to_string())
