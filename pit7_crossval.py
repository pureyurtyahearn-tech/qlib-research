"""Cross-validate Sharadar SEP against the yfinance data we already trust.

Same method as the July-12 SP500-vs-Kaggle comparison: for tickers present in BOTH, compare
DAILY RETURNS (not price levels -- the two sources sit on different adjustment bases, so
levels legitimately differ by a constant; returns must agree).

Checks:
  1. correlation of daily returns, per ticker
  2. distribution of return differences + systematic bias (t-test on the mean diff)
  3. KNOWN SPLIT EVENTS -- do they land correctly, i.e. is there NO artificial jump in the
     adjusted return series on the split date? This is the exact failure mode that corrupted
     the Kaggle NYSE data (double-adjustment producing 4x/8x cliffs).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

OUT = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")

# known splits in our window (ticker, date, ratio) -- verified corporate actions
SPLITS = [("NVDA", "2021-07-20", 4), ("NVDA", "2024-06-10", 10), ("AAPL", "2020-08-31", 4),
          ("TSLA", "2020-08-31", 5), ("TSLA", "2022-08-25", 3), ("AMZN", "2022-06-06", 20),
          ("GOOGL", "2022-07-18", 20), ("SHOP", "2022-06-29", 10), ("AVGO", "2024-07-15", 10)]


def main():
    sep = pd.read_hdf(OUT / "sep_panel.h5")
    sep_c = sep["$close"].unstack("ticker").sort_index()
    yf = pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")["$close"].unstack("instrument").sort_index()

    both = sorted(set(sep_c.columns) & set(yf.columns))
    print(f"tickers in BOTH sources: {len(both)}   (SEP {sep_c.shape[1]}, yfinance {yf.shape[1]})")

    idx = sep_c.index.intersection(yf.index)
    a = sep_c.loc[idx, both].pct_change()
    b = yf.loc[idx, both].pct_change()
    print(f"overlapping dates: {len(idx)}   {idx.min().date()} .. {idx.max().date()}\n")

    # ---- 1. per-ticker return correlation ----
    rows = []
    for t in both:
        x, y = a[t], b[t]
        m = x.notna() & y.notna()
        if m.sum() < 250:
            continue
        c = x[m].corr(y[m])
        d = (x[m] - y[m])
        rows.append(dict(ticker=t, n=int(m.sum()), corr=c, mean_diff=d.mean(),
                         std_diff=d.std(), max_abs=d.abs().max()))
    r = pd.DataFrame(rows)
    print("=== 1. daily-return correlation, SEP vs yfinance ===")
    print(f"  tickers compared: {len(r)}")
    print(f"  corr: mean={r.corr_.mean() if 'corr_' in r else r['corr'].mean():.5f}  "
          f"median={r['corr'].median():.5f}  min={r['corr'].min():.4f}")
    for thr in [0.999, 0.99, 0.95]:
        print(f"  corr >= {thr}: {(r['corr'] >= thr).sum():>4}/{len(r)}  ({(r['corr']>=thr).mean():.1%})")
    worst = r.nsmallest(8, "corr")
    print("\n  WORST-AGREEING tickers (investigate these):")
    print(worst[["ticker", "n", "corr", "std_diff", "max_abs"]].to_string(index=False))

    # ---- 2. systematic bias ----
    print("\n=== 2. return-difference distribution + systematic bias ===")
    alld = (a - b).values.ravel()
    alld = alld[np.isfinite(alld)]
    print(f"  n={len(alld):,}  mean={alld.mean():+.3e}  std={alld.std():.3e}")
    print(f"  |diff| percentiles: p50={np.percentile(np.abs(alld),50):.2e}  "
          f"p99={np.percentile(np.abs(alld),99):.2e}  max={np.abs(alld).max():.4f}")
    t, p = stats.ttest_1samp(alld, 0.0)
    print(f"  t-test mean diff = 0:  t={t:+.2f}  p={p:.3f}   "
          f"{'NO systematic bias' if p > 0.01 else 'SYSTEMATIC BIAS PRESENT'}")
    print(f"  share of obs |diff| > 1%: {(np.abs(alld) > 0.01).mean():.4%}")

    # ---- 3. known splits ----
    print("\n=== 3. KNOWN SPLIT EVENTS -- does SEP's adjustment leave an artificial cliff? ===")
    print("  (a correctly-adjusted series shows a NORMAL return on the split date;")
    print("   a mis-adjusted one shows ~1/ratio-1, e.g. -75% for a 4:1)")
    print(f"\n  {'ticker':7}{'split date':12}{'ratio':>6}{'SEP ret':>10}{'yf ret':>10}{'verdict':>12}")
    for tk, d, ratio in SPLITS:
        if tk not in sep_c.columns:
            print(f"  {tk:7}{d:12}{ratio:>6}   not in PIT universe")
            continue
        s = sep_c[tk].pct_change()
        y = yf[tk].pct_change() if tk in yf.columns else None
        dt = pd.Timestamp(d)
        if dt not in s.index:
            near = s.index[s.index.get_indexer([dt], method="nearest")][0]
            dt = near
        sr = s.get(dt, np.nan)
        yr = y.get(dt, np.nan) if y is not None else np.nan
        cliff = -(1 - 1 / ratio)          # what a FAILED adjustment would look like
        ok = abs(sr - cliff) > 0.10 and abs(sr) < 0.35
        print(f"  {tk:7}{str(dt.date()):12}{ratio:>5}:1{sr:>+10.4f}{yr:>+10.4f}"
              f"{'CLEAN' if ok else 'CHECK!':>12}")
    print(f"\n  (a failed 4:1 would print ~-0.7500; a failed 10:1 ~-0.9000)")


if __name__ == "__main__":
    main()
