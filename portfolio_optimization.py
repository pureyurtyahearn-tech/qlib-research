"""
Portfolio optimisation using Qlib data.
- Mean-Variance (Markowitz) frontier
- Maximum Sharpe portfolio
- Risk-Parity portfolio
- Comparison with Equal-Weight
"""
import warnings
warnings.filterwarnings("ignore")

import qlib
import pandas as pd
import numpy as np
from scipy.optimize import minimize

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
from qlib.data import D

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "JPM",  "BAC",  "XOM",  "JNJ",  "PG",
            "NVDA", "V",    "MA",   "HD"]
# Use 2017-2019 as estimation period, 2020 as out-of-sample test
TRAIN_S, TRAIN_E = "2017-01-01", "2019-12-31"
TEST_S,  TEST_E  = "2020-01-01", "2020-11-08"

df_tr = D.features(UNIVERSE, ["$close"], start_time=TRAIN_S, end_time=TRAIN_E, freq="day")
df_te = D.features(UNIVERSE, ["$close"], start_time=TEST_S,  end_time=TEST_E,  freq="day")
df_tr.columns = df_te.columns = ["close"]

def build_ret(df):
    prices = {t: df.xs(t, level="instrument").sort_index()["close"] for t in UNIVERSE}
    return pd.DataFrame(prices).pct_change().dropna()

ret_tr = build_ret(df_tr)
ret_te = build_ret(df_te)
N = len(UNIVERSE)

# In-sample stats
mu  = ret_tr.mean().values * 252       # annualised expected return
S   = ret_tr.cov().values  * 252       # annualised covariance

# ── 1. Equal-Weight ────────────────────────────────────────────────────────────
w_ew = np.ones(N) / N

def port_metrics(w, mu, S):
    r   = w @ mu
    vol = np.sqrt(w @ S @ w)
    sr  = r / vol
    return r, vol, sr

# ── 2. Minimum-Variance portfolio ─────────────────────────────────────────────
def neg_sr(w):
    _, vol, _ = port_metrics(w, mu, S)
    return vol

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds = [(0.0, 0.4)] * N  # max 40% per stock

res_mv = minimize(neg_sr, x0=w_ew, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"ftol": 1e-12, "maxiter": 500})
w_mv = res_mv.x

# ── 3. Maximum Sharpe Ratio ────────────────────────────────────────────────────
rf = 0.02  # 2% risk-free

def neg_sharpe(w):
    r, vol, _ = port_metrics(w, mu, S)
    return -(r - rf) / (vol + 1e-12)

res_msr = minimize(neg_sharpe, x0=w_ew, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 500})
w_msr = res_msr.x

# ── 4. Risk-Parity portfolio ───────────────────────────────────────────────────
def risk_parity_obj(w):
    port_var = w @ S @ w
    mrc      = S @ w                        # marginal risk contribution
    rc       = w * mrc / port_var           # risk contribution (fraction)
    target   = np.ones(N) / N              # equal risk contribution
    return np.sum((rc - target) ** 2)

res_rp = minimize(risk_parity_obj, x0=w_ew, method="SLSQP",
                  bounds=[(0.01, 0.4)] * N, constraints=constraints,
                  options={"ftol": 1e-14, "maxiter": 1000})
w_rp = res_rp.x / res_rp.x.sum()

# ── 5. In-sample vs out-of-sample comparison ─────────────────────────────────
portfolios = {
    "Equal-Weight": w_ew,
    "Min-Variance" : w_mv,
    "Max-Sharpe"   : w_msr,
    "Risk-Parity"  : w_rp,
}

print("=" * 72)
print("PORTFOLIO OPTIMISATION: IN-SAMPLE (2017-2019) vs OOS (2020-Nov)")
print("=" * 72)
print(f"{'Portfolio':<16}  {'IS_Ret':>7} {'IS_Vol':>7} {'IS_SR':>6}  │  {'OOS_Ret':>8} {'OOS_Vol':>7} {'OOS_SR':>6}")
print("-" * 72)

mu_oos = ret_te.mean().values * 252
S_oos  = ret_te.cov().values  * 252

for name, w in portfolios.items():
    is_r, is_v, is_sr = port_metrics(w, mu, S)
    oos_r, oos_v, oos_sr = port_metrics(w, mu_oos, S_oos)
    print(f"{name:<16}  {is_r:>7.2%} {is_v:>7.2%} {is_sr:>6.2f}  │  {oos_r:>8.2%} {oos_v:>7.2%} {oos_sr:>6.2f}")

# ── 6. Weight allocations ──────────────────────────────────────────────────────
print("\n=== Portfolio Weights (%) ===")
wt_df = pd.DataFrame(portfolios, index=UNIVERSE) * 100
print(wt_df.round(1).to_string())

# ── 7. OOS cumulative return comparison ────────────────────────────────────────
print("\n=== OOS Cumulative Return (2020 daily equity curves) ===")
cum_df = {}
for name, w in portfolios.items():
    daily = ret_te @ w
    cum_df[name] = (1 + daily).cumprod()

cum_df = pd.DataFrame(cum_df)
print(f"Final cumulative values on {cum_df.index[-1].date()}:")
print(cum_df.iloc[-1].round(4).to_string())

# ── 8. Max drawdown per portfolio ────────────────────────────────────────────
print("\nMax Drawdown (OOS):")
for col in cum_df.columns:
    dd = (cum_df[col] / cum_df[col].cummax() - 1).min()
    print(f"  {col:<16}: {dd:.2%}")
