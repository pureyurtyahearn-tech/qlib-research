"""Clean attribution: how much of the momentum edge was PURE survivorship bias?

The old (+7.4%, t=1.83) and new (+4.2%, t=1.00) results differ in THREE ways at once:
universe, price source, and window. To attribute the drop to survivorship alone, hold
prices (SEP) and window (2010-2026) fixed and vary ONLY the universe:

  A. BACKFILLED  -- the 447 names our old yfinance list had, tradeable on every date
                    (i.e. the survivorship-biased universe, reproduced on clean prices)
  B. TRUE PIT    -- only actual index members on each date

Any difference between A and B is survivorship bias, full stop.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from pit9_rerun import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    panel = pd.read_hdf(SH / "sep_panel.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)

    old = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                     .index.get_level_values("instrument").unique()))
    old_in = [t for t in old if t in close.columns]
    print(f"old backfilled list: {len(old)} names, {len(old_in)} present in SEP panel")

    ret_full = close.pct_change()
    sig_full = (close.shift(21) / close.shift(252) - 1).shift(1)
    S, E = "2010-01-01", "2026-06-29"
    w = (close.index >= S) & (close.index <= E)
    dates = close.index[w]; T = len(dates)
    retv = ret_full.loc[dates].values
    sigv = sig_full.loc[dates].values
    px_ok = np.isfinite(close.loc[dates].values)
    cols = list(close.columns)

    is_old = np.array([c in set(old_in) for c in cols])
    eligs = {
        "A. BACKFILLED (survivor)": px_ok & is_old[None, :],
        "B. TRUE POINT-IN-TIME":    px_ok & memb.loc[dates].values,
    }

    print(f"\nwindow {dates[0].date()}..{dates[-1].date()}  ({T} days), SEP prices for BOTH")
    print(f"{'universe':26}{'names/day':>11}")
    for nm, e in eligs.items():
        print(f"{nm:26}{e.sum(axis=1).mean():>11.0f}")

    print(f"\n{'universe':26}{'K':>4}{'EWbench':>9}{'turn':>7}{'grossEx':>10}{'netEx':>9}"
          f"{'IR':>7}{'t':>7}{'ph>0':>7}")
    out = {}
    for nm, e in eligs.items():
        ew = np.array([np.nanmean(np.where(e[t], retv[t], np.nan)) for t in range(T)])
        for k in [20, 50]:
            nets = np.zeros((21, T)); gros = np.zeros((21, T)); tr = []
            for ph in range(21):
                rb = np.zeros(T, bool); rb[ph::21] = True
                n_, g_, at = simulate(sigv, retv, e, rb, k)
                nets[ph], gros[ph] = n_, g_; tr.append(at)
            net = nets.mean(axis=0); ex = net - ew
            pp = int(sum(ann(nets[p] - ew) > 0 for p in range(21)))
            out[(nm, k)] = (ann(ex), ir(ex), tstat(ex), ann(ew))
            print(f"{nm:26}{k:>4}{ann(ew):>+9.4f}{np.mean(tr):>6.2f}x{ann(gros.mean(axis=0)-ew):>+10.4f}"
                  f"{ann(ex):>+9.4f}{ir(ex):>+7.2f}{tstat(ex):>+7.2f}{pp:>5}/21")

    print("\n=== PURE SURVIVORSHIP EFFECT (same prices, same window, universe only) ===")
    for k in [20, 50]:
        a = out[("A. BACKFILLED (survivor)", k)]
        b = out[("B. TRUE POINT-IN-TIME", k)]
        print(f"  K={k}:  backfilled netEx {a[0]:+.4f} (t {a[2]:+.2f})   ->   "
              f"true PIT {b[0]:+.4f} (t {b[2]:+.2f})")
        print(f"        survivorship was INFLATING the edge by {a[0]-b[0]:+.4f}/yr "
              f"({(a[0]/b[0]-1)*100 if b[0] else float('nan'):+.0f}% overstatement)")
    a = out[("A. BACKFILLED (survivor)", 20)]; b = out[("B. TRUE POINT-IN-TIME", 20)]
    print(f"\n  benchmark itself: backfilled EW {a[3]:+.4f}/yr vs true-PIT EW {b[3]:+.4f}/yr"
          f"  (survivor universe looks {a[3]-b[3]:+.4f}/yr richer even before any strategy)")


if __name__ == "__main__":
    main()
