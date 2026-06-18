"""
Risk models demo using Qlib data.
- Sample covariance matrix
- PCA factor risk model (3 factors)
- Risk decomposition: factor vs idiosyncratic
"""
import warnings
warnings.filterwarnings("ignore")

import qlib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
from qlib.data import D

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "JPM",  "BAC",  "XOM",  "JNJ",  "PG",
            "NVDA", "V",    "MA",   "HD"]
START, END = "2017-01-01", "2020-11-08"

# ── 1. Fetch returns ──────────────────────────────────────────────────────────
df = D.features(UNIVERSE, ["$close"], start_time=START, end_time=END, freq="day")
df.columns = ["close"]
prices = {t: df.xs(t, level="instrument").sort_index()["close"] for t in UNIVERSE}
px_df  = pd.DataFrame(prices).dropna()
ret_df = px_df.pct_change().dropna()
print(f"Return matrix: {ret_df.shape}  ({ret_df.index[0].date()} → {ret_df.index[-1].date()})")

# ── 2. Sample Covariance (annualised) ────────────────────────────────────────
S = ret_df.cov() * 252
print("\n=== Sample Covariance (annualised) ===")
print(np.sqrt(np.diag(S.values)) * 100)  # individual volatilities %

# Print correlation matrix
corr = ret_df.corr()
print("\n=== Correlation Matrix ===")
print(corr.round(2).to_string())

# ── 3. PCA Factor Risk Model on RAW returns ──────────────────────────────────
# Eigendecompose sample cov matrix directly
N_FACTORS = 3
S_np = S.values
eigvals, eigvecs = np.linalg.eigh(S_np)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# Factor loadings: each column is eigvec * sqrt(eigval)
B = eigvecs[:, :N_FACTORS] * np.sqrt(eigvals[:N_FACTORS])   # (N, k)

# Factor covariance = I (by construction of PCA)
# Idiosyncratic variance = diag(S - B @ B^T)
factor_cov_mat = B @ B.T
idio_var_vec   = np.diag(S_np) - np.diag(factor_cov_mat)
idio_var_vec   = np.clip(idio_var_vec, 0, None)
D_mat          = np.diag(idio_var_vec)

loadings_df = pd.DataFrame(B, index=UNIVERSE,
                            columns=[f"F{i+1}" for i in range(N_FACTORS)])

expl_var = eigvals[:N_FACTORS] / eigvals.sum()
print(f"\n=== PCA Factor Risk Model ({N_FACTORS} factors) ===")
print(f"Explained variance: {expl_var.round(3)}")
print(f"Total explained   : {expl_var.sum():.2%}")
print("\nFactor loadings (normalised):")
print(loadings_df.round(3).to_string())

# ── 4. Risk attribution for equal-weight portfolio ───────────────────────────
total_cov = S_np
w = np.ones(len(UNIVERSE)) / len(UNIVERSE)

port_var_total  = w @ total_cov @ w
port_var_factor = w @ factor_cov_mat @ w
port_var_idio   = w @ D_mat @ w

port_vol_total  = np.sqrt(port_var_total)
port_vol_factor = np.sqrt(port_var_factor)
port_vol_idio   = np.sqrt(port_var_idio)

print("\n=== Equal-Weight Portfolio Risk Decomposition ===")
print(f"Total Annualised Vol   : {port_vol_total:.2%}")
print(f"Factor Risk (3 PCA)    : {port_vol_factor:.2%}  ({port_var_factor/port_var_total:.1%} of total var)")
print(f"Idiosyncratic Risk     : {port_vol_idio:.2%}  ({port_var_idio/port_var_total:.1%} of total var)")

# ── 5. Stock-level risk breakdown ────────────────────────────────────────────
print("\n=== Per-Stock Risk Decomposition ===")
for i, t in enumerate(UNIVERSE):
    s_var    = total_cov[i, i]
    f_pct    = factor_cov_mat[i, i] / s_var * 100
    idio_pct = idio_var_vec[i] / s_var * 100
    print(f"  {t:6s}  total_vol={np.sqrt(s_var)*100:.1f}%  factor={f_pct:.1f}%  idio={idio_pct:.1f}%")
