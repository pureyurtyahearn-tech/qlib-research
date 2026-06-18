"""
Qlib built-in backtesting engine demo.
Uses TopkDropoutStrategy with synthetic momentum signal on S&P 500-like universe.
"""
import warnings
warnings.filterwarnings("ignore")

import qlib
import pandas as pd
import numpy as np

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
from qlib.data import D
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily
from qlib.backtest.executor import SimulatorExecutor

# ── 1. Universe & prices ─────────────────────────────────────────────────────
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "JPM",  "BAC",  "XOM",  "JNJ",  "PG",
            "NVDA", "META", "V",    "MA",   "HD"]
START    = "2017-01-01"
END      = "2020-11-08"   # last full day in sample

df = D.features(UNIVERSE, ["$close", "$volume"], start_time=START, end_time=END, freq="day")
df.columns = ["close", "volume"]

prices, avail = {}, []
for t in UNIVERSE:
    try:
        prices[t] = df.xs(t, level="instrument").sort_index()["close"]
        avail.append(t)
    except KeyError:
        pass
px_df = pd.DataFrame(prices)
print(f"Stocks available: {avail}")

# ── 2. Momentum alpha signal ─────────────────────────────────────────────────
mom20 = px_df.pct_change(20)

# ── 3. Qlib-format prediction scores ─────────────────────────────────────────
# shape: (datetime index) × (instrument columns)  — expected by TopkDropoutStrategy
pred_score = mom20.copy()
pred_score.index = pd.to_datetime(pred_score.index)

# ── 4. Build strategy ─────────────────────────────────────────────────────────
strategy = TopkDropoutStrategy(signal=pred_score, topk=5, n_drop=2)

# ── 5. Executor config ────────────────────────────────────────────────────────
executor_config = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
}

# ── 6. Exchange config (US equities — no limit) ───────────────────────────────
exchange_kwargs = {
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
    "limit_threshold": None,   # no limit-up/limit-down rules in US
    "codes": avail,
}

# ── 7. Run backtest ───────────────────────────────────────────────────────────
bt_start = str(pred_score.dropna(how="all").index[5].date())
bt_end   = str(pred_score.index[-2].date())
print(f"\nBacktest period: {bt_start} → {bt_end}")

try:
    report_normal, positions = backtest_daily(
        start_time=bt_start,
        end_time=bt_end,
        strategy=strategy,
        executor=executor_config,
        account=1_000_000,
        benchmark="",          # no benchmark
        exchange_kwargs=exchange_kwargs,
    )

    print("\n=== Qlib Portfolio Report ===")
    print(report_normal.tail(5))

    pnl = report_normal["return"]
    cum = (1 + pnl).cumprod()
    n   = len(pnl)
    ann = (cum.iloc[-1]) ** (252/n) - 1
    vol = pnl.std() * np.sqrt(252)
    sr  = ann / vol
    dd  = (cum / cum.cummax() - 1).min()
    print(f"\nTotal Return  : {cum.iloc[-1]-1:.2%}")
    print(f"Annual Return : {ann:.2%}")
    print(f"Volatility    : {vol:.2%}")
    print(f"Sharpe        : {sr:.2f}")
    print(f"Max Drawdown  : {dd:.2%}")

except Exception as e:
    print(f"\nBacktest error: {e}")
    print("\nFalling back to manual simulation...")

    fwd = px_df.pct_change().shift(-1)
    pos = pd.DataFrame(0.0, index=mom20.index, columns=px_df.columns)
    for date in mom20.dropna(how="all").index:
        row = mom20.loc[date].dropna()
        if len(row) < 5:
            continue
        top = row.nlargest(5).index
        pos.loc[date, top] = 1.0 / 5

    daily_pnl = (pos * fwd).sum(axis=1).dropna()
    # Subtract 10bps transaction cost per turnover
    turnover  = pos.diff().abs().sum(axis=1) * 0.001
    daily_pnl -= turnover

    cumret = (1 + daily_pnl).cumprod()
    n      = len(daily_pnl)
    total  = cumret.iloc[-1] - 1
    ann    = (1 + total) ** (252/n) - 1
    vol    = daily_pnl.std() * np.sqrt(252)
    sr     = ann / vol
    dd     = (cumret / cumret.cummax() - 1).min()

    print("\n=== Top-5 Momentum, Long-Only (with ~10bps cost) ===")
    print(f"Total Return  : {total:.2%}")
    print(f"Annual Return : {ann:.2%}")
    print(f"Volatility    : {vol:.2%}")
    print(f"Sharpe        : {sr:.2f}")
    print(f"Max Drawdown  : {dd:.2%}")
    print(f"Win Rate      : {(daily_pnl>0).mean():.2%}")

    yr = daily_pnl.groupby(daily_pnl.index.year).apply(lambda x: (1+x).prod()-1)
    print("\nYearly Returns:")
    print(yr.to_string())
