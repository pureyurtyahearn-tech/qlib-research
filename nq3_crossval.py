"""Cross-validate the NASDAQ Sharadar pull against existing data on overlapping tickers.

CRITICAL METHOD (the fixed ext9 lesson): align each ticker on its OWN dropna overlap window,
THEN pct_change. Never a global reindex (that injects spurious boundary returns and falsely
depressed a correlation to 0.81 earlier this week; the true value was 0.9999).

Two checks:
  (a) vs sep_panel_full (Sharadar SP500 PIT) -- SAME source, must be ~identical (consistency).
  (b) vs daily_pv_pre_sharadar (yfinance SP500 + Kaggle NYSE) -- INDEPENDENT source, the real
      cross-validation. NASDAQ mega-caps (AAPL/MSFT/INTC/CSCO...) live in the old yfinance data.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def per_ticker_corr(new_c, old_c, label):
    both = sorted(set(new_c.columns) & set(old_c.columns))
    corrs, biases = [], []
    for t in both:
        no = new_c[t].dropna(); oo = old_c[t].dropna()
        ov = no.index.intersection(oo.index)
        if len(ov) < 100:
            continue
        ra = no.reindex(ov).pct_change(); rb = oo.reindex(ov).pct_change()
        m = ra.notna() & rb.notna()
        if m.sum() > 100:
            c = ra[m].corr(rb[m])
            if np.isfinite(c):
                corrs.append(c); biases.append((ra[m] - rb[m]).mean())
    corrs = np.array(corrs); biases = np.array(biases)
    print(f"  {label}: {len(corrs)} common tickers")
    if len(corrs):
        print(f"    median return corr {np.median(corrs):.5f}  mean {np.mean(corrs):.5f}  "
              f">=0.99 {100*(corrs>=0.99).mean():.0f}%  >=0.95 {100*(corrs>=0.95).mean():.0f}%")
        print(f"    <0.9 (investigate) {int((corrs<0.9).sum())}  mean daily-return bias {np.mean(biases):+.2e}")
        worst = pd.Series(corrs, index=[t for t in both if t in new_c.columns][:len(corrs)])
    return corrs


def main():
    nq = pd.read_hdf(SH / "sep_nasdaq_panel.h5")["$close"].unstack("ticker").sort_index()
    print(f"NASDAQ panel: {nq.shape[1]} tickers, {nq.index.min().date()}..{nq.index.max().date()}\n")

    # (a) same-source consistency
    print("=== (a) vs Sharadar SP500 PIT panel (SAME source -> expect ~identical) ===")
    sp = pd.read_hdf(SH / "sep_panel_full.h5")["$close"].unstack("ticker").sort_index()
    per_ticker_corr(nq, sp, "NASDAQ vs SP500-Sharadar")

    # (b) independent source
    print("\n=== (b) vs old daily_pv (yfinance+Kaggle, INDEPENDENT source) ===")
    old = pd.read_hdf(SRC / "daily_pv_pre_sharadar.h5")["$close"].unstack("instrument").sort_index()
    corrs = per_ticker_corr(nq, old, "NASDAQ vs yfinance/Kaggle")

    # known split spot-checks on NASDAQ names (adjustment sanity)
    print("\n=== known NASDAQ split events (adjusted series must show NO artificial cliff) ===")
    SPL = [("AAPL", "2020-08-31", 4), ("TSLA", "2020-08-31", 5), ("TSLA", "2022-08-25", 3),
           ("NVDA", "2021-07-20", 4), ("NVDA", "2024-06-10", 10), ("AMZN", "2022-06-06", 20)]
    for tk, d, r in SPL:
        if tk not in nq.columns:
            print(f"  {tk}: not in NASDAQ panel"); continue
        s = nq[tk].pct_change()
        dt = pd.Timestamp(d)
        if dt not in s.index:
            dt = s.index[s.index.get_indexer([dt], method="nearest")][0]
        rv = s.get(dt, np.nan)
        cliff = -(1 - 1 / r)
        ok = abs(rv) < 0.35
        print(f"  {tk:6}{str(dt.date())}  {r}:1  ret {rv:+.4f}  "
              f"{'CLEAN' if ok else 'CHECK (looks like unadjusted cliff!)'}")


if __name__ == "__main__":
    main()
