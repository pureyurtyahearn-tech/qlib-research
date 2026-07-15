"""Classic academic momentum (Jegadeesh & Titman 1993): 12-month lookback, skip most
recent month. Built on the SP500 (yfinance) panel from daily_pv.h5, 2010-2026.

Answers: does a long-horizon, skip-month construction have a DIFFERENT IC decay profile
than the fast technical factors already tested? Controls included:
  mom_12_1 : close[t-21]/close[t-252]-1   <- the academic factor
  mom_12_0 : close[t]   /close[t-252]-1   <- same horizon, NO skip (does the skip matter?)
  mom_6_1  : close[t-21]/close[t-126]-1   <- half horizon, same construction
  rev_1m   : close[t]   /close[t-21] -1   <- 1-month (the academic REVERSAL leg)
  mom_5d   : close[t]   /close[t-5]  -1   <- fast technical (the family already tested)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SRC = Path("git_ignore_folder/factor_implementation_source_data")
OUT = Path("git_ignore_folder/_ac_factors"); OUT.mkdir(exist_ok=True, parents=True)
HORIZONS = [1, 5, 10, 21, 63, 126]


def rank_ic(sig, fwd):
    """mean daily cross-sectional Spearman IC + t-stat of the daily IC series"""
    ics = []
    for t in sig.index:
        if t not in fwd.index:
            continue
        a, b = sig.loc[t], fwd.loc[t]
        m = a.notna() & b.notna()
        if m.sum() > 20:
            ics.append(a[m].corr(b[m], method="spearman"))
    ics = np.array(ics, float)
    ics = ics[~np.isnan(ics)]
    mu = ics.mean()
    # Newey-West-free naive t; IC series is autocorrelated for long horizons so treat
    # t as indicative only (the portfolio test is the real arbiter)
    t_stat = mu / ics.std() * np.sqrt(len(ics)) if ics.std() > 0 else 0.0
    return mu, t_stat, ics


def main():
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    close = comb["$close"].unstack("instrument")[ [c for c in sp if c in comb["$close"].unstack("instrument").columns] ]
    close = close.sort_index()
    close = close.loc[:, close.notna().sum() > 252]
    print(f"SP500 panel: {close.shape[0]} days x {close.shape[1]} instruments   "
          f"{close.index.min().date()} .. {close.index.max().date()}")

    facs = {
        "mom_12_1": close.shift(21) / close.shift(252) - 1,
        "mom_12_0": close / close.shift(252) - 1,
        "mom_6_1":  close.shift(21) / close.shift(126) - 1,
        "rev_1m":   close / close.shift(21) - 1,
        "mom_5d":   close / close.shift(5) - 1,
    }
    # forward return: enter at t+1 close, exit at t+1+h close (causal, no look-ahead)
    fwd = {h: close.shift(-(1 + h)) / close.shift(-1) - 1 for h in HORIZONS}

    # burn-in: only evaluate where the longest factor is defined
    valid = facs["mom_12_1"].notna().sum(axis=1) > 50
    dates = close.index[valid]
    print(f"evaluation window (after 252d burn-in): {dates.min().date()} .. {dates.max().date()}  ({len(dates)} days)\n")
    for k in facs:
        facs[k] = facs[k].loc[dates]
        facs[k].to_hdf(OUT / f"{k}.h5", key="f", complevel=5)
    fwd = {h: v.loc[dates] for h, v in fwd.items()}

    # ---------- noise floor (placebo), per horizon ----------
    rng = np.random.default_rng(0)
    floor = {}
    for h in HORIZONS:
        null = []
        for _ in range(15):
            r = pd.DataFrame(rng.standard_normal(fwd[h].shape), index=fwd[h].index, columns=fwd[h].columns)
            r = r.where(fwd[h].notna())
            null.append(rank_ic(r, fwd[h])[0])
        floor[h] = float(np.std(null))
    print("NOISE FLOOR (15 random signals, RankIC std) by horizon:")
    print("   " + "".join(f"h={h}:{floor[h]:.5f}  " for h in HORIZONS) + "\n")

    # ---------- IC table ----------
    print(f"{'factor':12}" + "".join(f"{'IC h='+str(h):>10}" for h in HORIZONS))
    ic_tab = {}
    for name, w in facs.items():
        row = {}
        for h in HORIZONS:
            mu, ts, _ = rank_ic(w, fwd[h])
            row[h] = mu
        ic_tab[name] = row
        print(f"{name:12}" + "".join(f"{row[h]:>+10.4f}" for h in HORIZONS))

    print(f"\n{'factor':12}" + "".join(f"{'|z| h='+str(h):>10}" for h in HORIZONS) +
          "   (|IC| / noise-floor sigma; >2 = clears placebo)")
    for name in facs:
        print(f"{name:12}" + "".join(f"{abs(ic_tab[name][h])/floor[h]:>10.1f}" for h in HORIZONS))

    print(f"\n=== IC RETENTION  |IC(h)| / |IC(h=1)|   (higher = decays SLOWER) ===")
    print(f"{'factor':12}" + "".join(f"{'h='+str(h):>10}" for h in HORIZONS))
    for name in facs:
        b = abs(ic_tab[name][1])
        print(f"{name:12}" + "".join(f"{abs(ic_tab[name][h])/b:>10.2f}" for h in HORIZONS))

    # ---------- signal persistence -> implied turnover ----------
    print(f"\n=== SIGNAL PERSISTENCE (cross-sectional rank autocorr; higher = less turnover) ===")
    print(f"{'factor':12}{'AC(1)':>9}{'AC(5)':>9}{'AC(21)':>9}{'AC(63)':>9}")
    for name, w in facs.items():
        acs = []
        for k in [1, 5, 21, 63]:
            v = []
            for i in range(k, len(w), 5):
                a, b = w.iloc[i], w.iloc[i - k]
                m = a.notna() & b.notna()
                if m.sum() > 20:
                    v.append(a[m].corr(b[m], method="spearman"))
            acs.append(float(np.nanmean(v)))
        print(f"{name:12}" + "".join(f"{a:>+9.2f}" for a in acs))

    pd.DataFrame(ic_tab).to_csv(OUT / "ic_table.csv")
    print(f"\nsaved factors -> {OUT}/")


if __name__ == "__main__":
    main()
