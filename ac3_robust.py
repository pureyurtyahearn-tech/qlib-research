"""Stress the 12-1 momentum result before believing it.

The headline (net +2.46%/yr vs EW own universe, 21/21 phases) has two red flags:
  (a) net IR is only +0.22 -> is the excess even statistically distinguishable from zero?
  (b) 2026 shows +41.9% annualized excess on a PARTIAL year -> is the whole result that?
Also: does it survive dropping the best year, splitting the sample, and changing K?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from ac2_portfolio import simulate, ann, ir, maxdd, SRC, FDIR


def tstat(x):
    x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(len(x))


def main():
    import qlib
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "us_data"), region="us")

    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sig12 = pd.read_hdf(FDIR / "mom_12_1.h5")
    close = comb["$close"].unstack("instrument").sort_index()[sig12.columns].loc[sig12.index]
    ret_w = close.pct_change()
    dates = ret_w.index; T = len(dates)
    retv = ret_w.values
    ewv = ret_w.mean(axis=1).values
    rsp = D.features(["RSP"], ["$close/Ref($close,1)-1"], start_time=str(dates.min().date()),
                     end_time=str(dates.max().date())).iloc[:, 0]
    rsp.index = rsp.index.get_level_values("datetime")
    rspv = rsp.reindex(dates).values
    sv = sig12.shift(1).values

    # phase-averaged net series (the thing we are testing)
    allnet = np.zeros((21, T))
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        allnet[ph], _, _, _ = simulate(sv, retv, rb)
    net = allnet.mean(axis=0)
    ex_ew = net - ewv
    ex_rsp = net - rspv
    yrs = T / 252

    print("=== (a) IS THE EXCESS STATISTICALLY REAL? (monthly, phase-averaged, net of cost) ===")
    for lbl, ex in [("vs EW own universe", ex_ew), ("vs RSP", ex_rsp)]:
        e = ex[np.isfinite(ex)]
        te = e.std() * np.sqrt(252)
        print(f"  {lbl:20} net {ann(e):+.4f}/yr   trackErr {te:.4f}   IR {ir(e):+.2f}   "
              f"t-stat {tstat(e):+.2f}   (n={yrs:.1f}y)")
    print(f"\n  A t-stat of ~2.0 is the usual bar. Over {yrs:.1f} years an IR of {ir(ex_ew):+.2f} "
          f"gives t={ir(ex_ew)*np.sqrt(yrs):+.2f}.")

    print("\n=== (b) IS IT ALL 2026? (2026 is a PARTIAL year: "
          f"{int((dates.year==2026).sum())} trading days) ===")
    for lbl, msk in [("full sample", np.ones(T, bool)),
                     ("excl. 2026 (partial)", dates.year < 2026),
                     ("excl. 2026 + 2024 (2 best)", (dates.year < 2026) & (dates.year != 2024))]:
        e = ex_ew[msk]
        print(f"  {lbl:28} net vs EW {ann(e):+.4f}/yr   IR {ir(e):+.2f}   t {tstat(e):+.2f}")

    print("\n=== drop-one-year jackknife (net excess vs EW, rest of sample) ===")
    res = []
    for y in sorted(set(dates.year)):
        m = dates.year != y
        res.append((y, ann(ex_ew[m])))
    res_s = sorted(res, key=lambda r: r[1])
    print("  most influential years (excess of sample WITHOUT that year):")
    for y, v in res_s[:3]:
        print(f"    drop {y}: {v:+.4f}   <- removing it HURTS most (year was a big contributor)")
    for y, v in res_s[-2:]:
        print(f"    drop {y}: {v:+.4f}   <- removing it HELPS (year was a drag)")
    vals = np.array([v for _, v in res])
    print(f"  jackknife range: {vals.min():+.4f} .. {vals.max():+.4f}   "
          f"(all positive: {bool((vals>0).all())})")

    print("\n=== sample split ===")
    half = T // 2
    for lbl, msk in [("first half " + f"{dates[0].date()}..{dates[half-1].date()}", np.arange(T) < half),
                     ("second half " + f"{dates[half].date()}..{dates[-1].date()}", np.arange(T) >= half)]:
        e = ex_ew[msk]
        print(f"  {lbl:42} net vs EW {ann(e):+.4f}/yr  IR {ir(e):+.2f}  t {tstat(e):+.2f}")

    print("\n=== portfolio-size sensitivity (monthly, phase-avg, net vs EW) ===")
    print(f"{'K':>5}{'turn/yr':>9}{'grossEx':>10}{'netEx':>9}{'IR':>7}{'t':>7}")
    for k in [20, 30, 50, 75, 100, 150]:
        nn = np.zeros((21, T)); gg = np.zeros((21, T)); tt = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            nn[ph], gg[ph], _, at = simulate(sv, retv, rb, k=k)
            tt.append(at)
        n_ = nn.mean(axis=0); g_ = gg.mean(axis=0)
        e = n_ - ewv
        print(f"{k:>5}{np.mean(tt):>8.2f}x{ann(g_-ewv):>+10.4f}{ann(e):>+9.4f}{ir(e):>+7.2f}{tstat(e):>+7.2f}")

    print("\n=== cost sensitivity (K=50, monthly, phase-avg, net vs EW) ===")
    print(f"{'round-trip bps':>16}{'netEx_EW':>10}{'IR':>7}")
    import ac2_portfolio as ac2
    base_o, base_c = ac2.OPEN_C, ac2.CLOSE_C
    for rt in [0, 10, 20, 40, 60, 87, 100]:
        ac2.OPEN_C = rt / 2 / 1e4; ac2.CLOSE_C = rt / 2 / 1e4
        nn = np.zeros((21, T))
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            nn[ph], _, _, _ = simulate(sv, retv, rb)
        e = nn.mean(axis=0) - ewv
        print(f"{rt:>16}{ann(e):>+10.4f}{ir(e):>+7.2f}")
    ac2.OPEN_C, ac2.CLOSE_C = base_o, base_c


if __name__ == "__main__":
    main()
