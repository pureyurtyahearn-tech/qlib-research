"""
Order execution simulation using Qlib data.
Clean vectorized approach: compare gross vs net (with costs) returns.
Models:
  - Bid-ask spread (5bps)
  - Linear market impact: impact_bps = eta * (order_size / ADV * 10000)
  - Turnover analysis
"""
import warnings
warnings.filterwarnings("ignore")

import qlib
import pandas as pd
import numpy as np

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
from qlib.data import D

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "JPM",  "BAC",  "XOM",  "JNJ",  "PG",
            "NVDA", "V",    "MA",   "HD"]
START, END = "2019-01-01", "2020-11-08"
ACCOUNT    = 1_000_000
TOP_K      = 5
SPREAD_BPS = 5.0
ETA        = 0.1     # linear impact coefficient

# ── 1. Prices & volumes ───────────────────────────────────────────────────────
df = D.features(UNIVERSE, ["$close", "$volume"], start_time=START, end_time=END, freq="day")
df.columns = ["close", "volume"]
px = {t: df.xs(t, level="instrument").sort_index()["close"] for t in UNIVERSE}
vl = {t: df.xs(t, level="instrument").sort_index()["volume"] for t in UNIVERSE}
px_df = pd.DataFrame(px)
vl_df = pd.DataFrame(vl)

# ── 2. Signal: 20-day momentum, monthly rebalance ────────────────────────────
mom20 = px_df.pct_change(20)

# Monthly: first trading day of each month
monthly_idx = px_df.resample("MS").first().index.intersection(px_df.index)
# Rebalance signals and target weights (forward-fill between rebalances)
weights_df = pd.DataFrame(0.0, index=px_df.index, columns=UNIVERSE)
for dt in monthly_idx:
    if dt not in mom20.index:
        continue
    row = mom20.loc[dt].dropna()
    if len(row) < TOP_K:
        continue
    top = row.nlargest(TOP_K).index
    for t in UNIVERSE:
        weights_df.loc[dt, t] = (1.0 / TOP_K) if t in top else 0.0

weights_df = weights_df.replace(0.0, np.nan)
weights_df = weights_df.ffill().fillna(0.0)
# Exclude warmup period
weights_df.iloc[:20] = 0.0

# ── 3. Daily returns ──────────────────────────────────────────────────────────
ret_df = px_df.pct_change()

# ── 4. Gross P&L (no costs) ───────────────────────────────────────────────────
# Trade at close: positions established at close[t], return captured close[t]→close[t+1]
# Use lagged weights to avoid look-ahead
gross_daily = (weights_df.shift(1) * ret_df).sum(axis=1)

# ── 5. Cost computation ───────────────────────────────────────────────────────
# ADV: 20-day rolling average dollar volume
adv_df = (px_df * vl_df).rolling(20).mean()

# Turnover: |w[t] - w[t-1]| * portfolio_value
# (approximate: portfolio_value ≈ ACCOUNT * cum_return, use ACCOUNT for simplicity)
turnover_df = weights_df.diff().abs()   # weight change per stock per day

# Cost per unit of notional:
# total_bps = SPREAD_BPS + ETA * (order_notional / ADV) * 10000
# order_notional per stock ≈ |Δw| * ACCOUNT
# pov ≈ |Δw| * ACCOUNT / ADV

# Cost as fraction of daily portfolio:
def cost_fraction(dw_row, px_row, adv_row):
    """Daily cost as fraction of portfolio value."""
    total_cost = 0.0
    for t in UNIVERSE:
        dw = dw_row.get(t, 0.0)
        if dw == 0 or pd.isna(dw):
            continue
        notional = abs(dw) * ACCOUNT
        adv_t    = adv_row.get(t, np.nan)
        if pd.isna(adv_t) or adv_t <= 0:
            adv_t = 1e9
        pov      = notional / adv_t
        imp_bps  = ETA * pov * 10_000
        tot_bps  = SPREAD_BPS + imp_bps
        total_cost += notional * tot_bps / 10_000
    return total_cost / ACCOUNT

cost_pct = pd.Series(0.0, index=px_df.index)
rebal_dates = weights_df.index[weights_df.diff().abs().sum(axis=1) > 0.01]
for dt in rebal_dates:
    if dt not in adv_df.index:
        continue
    cost_pct[dt] = cost_fraction(turnover_df.loc[dt].to_dict(),
                                 px_df.loc[dt].to_dict(),
                                 adv_df.loc[dt].to_dict())

# ── 6. Net P&L ────────────────────────────────────────────────────────────────
net_daily = gross_daily - cost_pct

# ── 7. Equity curves ──────────────────────────────────────────────────────────
gross_cum  = (1 + gross_daily).cumprod()
net_cum    = (1 + net_daily).cumprod()

# ── 8. Metrics ────────────────────────────────────────────────────────────────
def metrics(daily, label):
    daily = daily.replace(0, np.nan).dropna()
    n     = len(daily)
    cum   = (1 + daily).cumprod()
    tot   = cum.iloc[-1] - 1
    ann   = (1 + tot) ** (252/n) - 1
    vol   = daily.std() * np.sqrt(252)
    sr    = ann / vol
    dd    = (cum / cum.cummax() - 1).min()
    return {"Label": label, "Total Return": f"{tot:.2%}", "Annual Return": f"{ann:.2%}",
            "Volatility": f"{vol:.2%}", "Sharpe": f"{sr:.2f}", "Max DD": f"{dd:.2%}"}

rows = [metrics(gross_daily, "Gross (no costs)"),
        metrics(net_daily,   "Net (with costs)")]
result_df = pd.DataFrame(rows).set_index("Label")

print("=" * 62)
print("ORDER EXECUTION SIMULATION  (2019-01-01 → 2020-11-08)")
print("Top-5 20-day momentum, monthly rebalance")
print(f"Cost model: {SPREAD_BPS}bps spread + {ETA}*POV impact")
print("=" * 62)
print(result_df.to_string())

# ── 9. Cost analysis ────────────────────────────────────────────────────────
actual_costs = cost_pct[cost_pct > 0]
monthly_turnover = turnover_df.sum(axis=1)  # sum of abs weight changes

print(f"\nRebalance events        : {len(actual_costs)}")
print(f"Avg cost per rebalance  : {actual_costs.mean()*100:.3f}% of portfolio")
print(f"Total cost drag         : {cost_pct.sum()*100:.2f}% over period")
print(f"Annual cost drag        : {cost_pct.sum()*100/2:.2f}%")
print(f"Avg monthly turnover    : {monthly_turnover[monthly_turnover>0].mean():.1%}")
print(f"Max impact seen         : {actual_costs.max()*100:.3f}%")

print("\nRebalance cost breakdown (sample, 5 dates):")
sample_idx = actual_costs.nlargest(5).index
for dt in sorted(sample_idx)[:5]:
    print(f"  {dt.date()}  cost={cost_pct[dt]*100:.3f}%  turnover={monthly_turnover[dt]:.1%}")

# ── 10. Qlib exchange-level execution ────────────────────────────────────────
print("\n--- Testing Qlib Exchange deal price modes ---")
try:
    from qlib.backtest.exchange import Exchange
    exc = Exchange(
        codes=UNIVERSE,
        start_time=START,
        end_time=END,
        freq="day",
        deal_price="close",
        open_cost=SPREAD_BPS/2/10000,
        close_cost=SPREAD_BPS/2/10000,
        min_cost=1,
        limit_threshold=None,
    )
    # Get trade price for AAPL on a specific date
    test_date = pd.Timestamp("2020-01-10")
    order_dir  = 1   # buy
    deal_px    = exc.get_deal_price("AAPL", test_date, test_date, direction=order_dir)
    print(f"  AAPL deal price on 2020-01-10 (buy): {deal_px:.4f}")
    print("  Qlib Exchange: OK")
except Exception as e:
    print(f"  Qlib Exchange test: {e}")
