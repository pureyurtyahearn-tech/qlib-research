"""Re-run K=20 (and K=50) 12-1 momentum on the FULL-history PIT S&P 500 universe.
Uses the PIT-enforcing pandas simulator (force-exit on index removal) reading
sep_panel_full.h5 directly -- memory-safe, no qlib store needed.

Reports the full-history t-stat vs the 2010-2026 result, since today's finding was that
HISTORY was the binding constraint on statistical significance.
START is set from ext3's quality verdict (passed in / edited here after ext3 runs).
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
OPEN_C, CLOSE_C = 0.0005, 0.0015
START = sys.argv[1] if len(sys.argv) > 1 else "1999-06-01"   # overridden by ext3 verdict
END = "2026-06-29"


def simulate(sig, ret, elig, rebal, k):
    T, N = sig.shape
    hold = np.zeros(N); gross = np.zeros(T); cost = np.zeros(T); turn = 0.0
    for t in range(T):
        r = np.nan_to_num(ret[t])
        if hold.any():
            gross[t] = hold @ r
            hold = hold * (1 + r)
            hold = np.where(elig[t], hold, 0.0)      # force-exit on index removal (PIT)
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


def ann(x): x = x[np.isfinite(x)]; return x.mean() * 252
def ir(x): x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.
def tstat(x): x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.


def main():
    panel = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)

    ret_full = close.pct_change()
    sig_full = (close.shift(21) / close.shift(252) - 1).shift(1)     # causal 12-1
    w = (close.index >= START) & (close.index <= END)
    dates = close.index[w]; T = len(dates)
    retv = ret_full.loc[dates].values
    sigv = sig_full.loc[dates].values
    elig = memb.loc[dates].values & np.isfinite(close.loc[dates].values)
    print(f"window {dates[0].date()}..{dates[-1].date()}  ({T} days, {T/252:.1f}y)")
    print(f"eligible members/day: mean {elig.sum(axis=1).mean():.0f}  min {elig.sum(axis=1).min()}")

    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
    print(f"EW own-universe benchmark: {ann(ew):+.4f}/yr\n")

    print(f"{'K':>4}{'turn/yr':>9}{'grossEx':>10}{'netEx':>9}{'netIR':>8}{'t-stat':>8}{'phases>0':>10}")
    res = {}
    for k in [20, 50]:
        nets = np.zeros((21, T)); gros = np.zeros((21, T)); tr = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            n_, g_, at = simulate(sigv, retv, elig, rb, k)
            nets[ph], gros[ph] = n_, g_; tr.append(at)
        net = nets.mean(axis=0); ex = net - ew
        pp = int(sum(ann(nets[p] - ew) > 0 for p in range(21)))
        res[k] = (ann(ex), ir(ex), tstat(ex))
        print(f"{k:>4}{np.mean(tr):>8.2f}x{ann(gros.mean(axis=0)-ew):>+10.4f}"
              f"{ann(ex):>+9.4f}{ir(ex):>+8.2f}{tstat(ex):>+8.2f}{pp:>7}/21")

    # decade breakdown -- is the edge stable across eras or concentrated?
    print(f"\n=== K=20 net excess vs EW by era ===")
    nets20 = np.zeros((21, T))
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        nets20[ph], _, _ = simulate(sigv, retv, elig, rb, 20)
    n20 = nets20.mean(axis=0); ex20 = n20 - ew
    for lo, hi in [(1999, 2005), (2005, 2010), (2010, 2016), (2016, 2021), (2021, 2027)]:
        m = (dates.year >= lo) & (dates.year < hi)
        if m.sum() < 100: continue
        print(f"  {lo}-{hi-1}: netEx {ann(ex20[m]):+.4f}  IR {ir(ex20[m]):+.2f}  t {tstat(ex20[m]):+.2f}")

    print(f"\n=== t-stat vs history (today's question: does more history move it?) ===")
    print(f"  2010-2026 (16y, PIT):  K=20 t=+1.00   K=50 t=+0.48")
    print(f"  {START[:4]}-2026 ({T/252:.0f}y, PIT):  K=20 t={res[20][2]:+.2f}   K=50 t={res[50][2]:+.2f}")


if __name__ == "__main__":
    main()
