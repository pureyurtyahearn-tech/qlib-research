"""The payoff: re-run 12-1 momentum on the TRUE point-in-time universe, and sanity-check
the reconstruction against known S&P 500 history.

(b) INDEX SANITY CHECK first -- if a cap-agnostic equal-weight of our reconstructed
    membership does not resemble the real index, nothing downstream is trustworthy. We also
    compare the reconstructed universe's yearly returns to actual S&P 500 total returns.

(a) MOMENTUM: K=20 and K=50, monthly rebalance, all 21 phases, net 5/15bps, benchmarked
    against equal-weight of the SAME point-in-time membership. The question is whether
    t=1.83 (old, backfilled universe) survives on real membership.

Crucially the strategy may ONLY hold a stock on date t if it was in the index on t.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
OPEN_C, CLOSE_C = 0.0005, 0.0015

# actual S&P 500 TOTAL returns (dividends reinvested), calendar year, public figures
SPX_TR = {2010: .1506, 2011: .0211, 2012: .1600, 2013: .3239, 2014: .1369, 2015: .0138,
          2016: .1196, 2017: .2183, 2018: -.0438, 2019: .3149, 2020: .1840, 2021: .2871,
          2022: -.1811, 2023: .2629, 2024: .2502}


def simulate(sig, ret, elig, rebal, k):
    T, N = sig.shape
    hold = np.zeros(N); gross = np.zeros(T); cost = np.zeros(T); turn = 0.0
    for t in range(T):
        r = np.nan_to_num(ret[t])
        if hold.any():
            gross[t] = hold @ r
            hold = hold * (1 + r)
            # force-exit anything that left the index (or stopped pricing)
            hold = np.where(elig[t], hold, 0.0)
            s = hold.sum()
            if s > 0:
                hold /= s
        if rebal[t]:
            s = np.where(elig[t] & np.isfinite(sig[t]), sig[t], np.nan)
            ok = ~np.isnan(s)
            if ok.sum() < k:
                continue
            idx = np.argsort(np.where(ok, s, -np.inf))[-k:]
            tw = np.zeros(N); tw[idx] = 1.0 / k
            dh = tw - hold
            buys = dh[dh > 0].sum(); sells = -dh[dh < 0].sum()
            cost[t] = OPEN_C * buys + CLOSE_C * sells
            turn += buys
            hold = tw
    return gross - cost, gross, turn / T * 252


def ann(x):
    x = x[np.isfinite(x)]; return x.mean() * 252


def ir(x):
    x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.


def tstat(x):
    x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.


def main():
    panel = pd.read_hdf(SH / "sep_panel.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)

    ret_full = close.pct_change()
    sig_full = (close.shift(21) / close.shift(252) - 1).shift(1)   # causal 12-1

    S, E = "2010-01-01", "2026-06-29"
    w = (close.index >= S) & (close.index <= E)
    dates = close.index[w]; T = len(dates)
    retv = ret_full.loc[dates].values
    sigv = sig_full.loc[dates].values
    # eligible = an actual index member on t, WITH a price
    elig = memb.loc[dates].values & np.isfinite(close.loc[dates].values)
    print(f"window {dates[0].date()} .. {dates[-1].date()}  ({T} days)")
    print(f"eligible names/day: mean {elig.sum(axis=1).mean():.0f}  "
          f"min {elig.sum(axis=1).min()}  max {elig.sum(axis=1).max()}")

    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])

    # ---------- (b) INDEX SANITY ----------
    print("\n=== (b) SANITY: reconstructed PIT universe vs actual S&P 500 total return ===")
    print("    (our EW is equal-weight & price-return-on-closeadj [divs reinvested];")
    print("     the real index is CAP-weighted, so a gap is EXPECTED -- we check plausibility,")
    print("     direction, and that no year is wildly wrong.)")
    print(f"\n  {'year':>6}{'PIT equal-wt':>14}{'S&P500 TR':>12}{'diff':>9}")
    rows = []
    for y in sorted(SPX_TR):
        m = np.asarray(dates.year == y)
        if m.sum() < 200:
            continue
        cum = np.nanprod(1 + ew[m]) - 1
        rows.append((y, cum, SPX_TR[y], cum - SPX_TR[y]))
        print(f"  {y:>6}{cum:>+14.4f}{SPX_TR[y]:>+12.4f}{cum-SPX_TR[y]:>+9.4f}")
    r = pd.DataFrame(rows, columns=["y", "pit", "spx", "d"])
    print(f"\n  correlation of yearly returns: {r.pit.corr(r.spx):.4f}")
    print(f"  mean |diff|: {r.d.abs().mean():.4f}   sign agreement: "
          f"{int((np.sign(r.pit)==np.sign(r.spx)).sum())}/{len(r)} years")
    print(f"  cumulative PIT EW {np.nanprod(1+ew)-1:+.2%} vs S&P500 TR "
          f"{np.prod([1+v for v in SPX_TR.values()])-1:+.2%} (2010-2024)")

    # ---------- (a) MOMENTUM ----------
    print("\n=== (a) 12-1 MOMENTUM on the TRUE point-in-time universe ===")
    print("    monthly rebalance, all 21 phases, net 5/15bps, vs EW of same PIT membership")
    print(f"\n{'K':>4}{'turn/yr':>9}{'grossEx':>10}{'netEx':>9}{'netIR':>8}{'t-stat':>8}{'phases>0':>10}{'maxDD':>9}")
    res = {}
    for k in [20, 50]:
        nets = np.zeros((21, T)); gros = np.zeros((21, T)); turns = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            n_, g_, at = simulate(sigv, retv, elig, rb, k)
            nets[ph], gros[ph] = n_, g_; turns.append(at)
        net = nets.mean(axis=0); gross = gros.mean(axis=0)
        ex = net - ew
        pp = int(sum(ann(nets[p] - ew) > 0 for p in range(21)))
        c = np.cumprod(1 + np.nan_to_num(ex)); dd = float((c / np.maximum.accumulate(c) - 1).min())
        res[k] = (ann(ex), ir(ex), tstat(ex))
        print(f"{k:>4}{np.mean(turns):>8.2f}x{ann(gross-ew):>+10.4f}{ann(ex):>+9.4f}"
              f"{ir(ex):>+8.2f}{tstat(ex):>+8.2f}{pp:>7}/21{dd:>+9.3f}")

    print("\n=== THE COMPARISON: old backfilled universe vs true PIT ===")
    print(f"  {'':22}{'netEx':>10}{'IR':>8}{'t':>8}")
    print(f"  {'OLD (backfilled) K=20':22}{'+0.0742':>10}{'+0.46':>8}{'+1.83':>8}")
    print(f"  {'NEW (true PIT)   K=20':22}{res[20][0]:>+10.4f}{res[20][1]:>+8.2f}{res[20][2]:>+8.2f}")
    print(f"  {'OLD (backfilled) K=50':22}{'+0.0246':>10}{'+0.22':>8}{'+0.87':>8}")
    print(f"  {'NEW (true PIT)   K=50':22}{res[50][0]:>+10.4f}{res[50][1]:>+8.2f}{res[50][2]:>+8.2f}")
    print("\n  (old numbers: 2011-2026 on 447 backfilled survivors; new: real membership.)")


if __name__ == "__main__":
    main()
