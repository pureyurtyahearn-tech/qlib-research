"""Evaluate the 7 fundamental factors on the PIT S&P 500 universe, same rigor as momentum:
IC vs placebo noise floor, TURNOVER (the headline metric), net-of-cost vs EW-own-universe.

The whole thesis: quarterly fundamentals -> naturally low turnover -> survive costs where the
fast-decaying technical factors did not. This script tests that thesis with actual numbers.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from ext6_momentum_full import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
START, END = "1999-06-01", "2026-06-29"
# economic-prior signs: +1 = high value good (long), -1 = high value bad (short)
FACTORS = {"$ey": +1, "$fcfy": +1, "$roe": +1, "$rgrow": +1, "$pe": -1, "$pb": -1, "$de": -1}


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        a, b, m = sig[t], fwd[t], mask[t]
        g = m & np.isfinite(a) & np.isfinite(b)
        if g.sum() > 30:
            ra = pd.Series(a[g]).rank(); rb = pd.Series(b[g]).rank()
            out.append(np.corrcoef(ra, rb)[0, 1])
    out = np.array(out); return out[np.isfinite(out)]


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    fund = pd.read_hdf(SH / "fundamentals_daily.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = px["$close"].unstack("ticker").sort_index()
    w = (close.index >= START) & (close.index <= END)
    dates = close.index[w]; T = len(dates)
    cols = close.columns
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(columns=cols, fill_value=False)
    ret = close.pct_change()
    retv = ret.loc[dates].values
    elig = memb.loc[dates].values & np.isfinite(close.loc[dates].values)
    fwd21 = (close.shift(-22) / close.shift(-1) - 1).loc[dates].values
    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
    print(f"window {dates[0].date()}..{dates[-1].date()} ({T/252:.1f}y), members/day {elig.sum(axis=1).mean():.0f}")
    print(f"EW own-universe benchmark: {ann(ew):+.4f}/yr\n")

    # noise floor for IC
    rng = np.random.default_rng(0)
    null = [rank_ic(rng.standard_normal((T, len(cols))), fwd21, elig).mean() for _ in range(15)]
    nsd = float(np.std(null))
    print(f"IC noise floor (placebo sd): {nsd:.5f}\n")

    def wide(fac):
        s = fund[fac].unstack("instrument")
        return s.reindex(index=dates, columns=cols)

    print(f"{'factor':8}{'sign':>5}{'RankIC':>9}{'|z|':>6}{'AC(21)':>8}{'turn/yr':>9}"
          f"{'netEx_EW':>10}{'netIR':>8}{'t':>7}")
    sig_store = {}
    for fac, sgn in FACTORS.items():
        raw = wide(fac).values
        sig = sgn * raw
        sig_store[fac] = sig
        ic = rank_ic(raw, fwd21, elig)     # IC of RAW factor (sign shows if prior is right)
        z = ic.mean() / nsd
        # signal persistence: cross-sectional rank autocorr at 21d
        acs = []
        for i in range(21, T, 10):
            a, b = sig[i], sig[i - 21]
            m = np.isfinite(a) & np.isfinite(b) & elig[i] & elig[i - 21]
            if m.sum() > 30:
                acs.append(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))
        ac21 = np.nanmean(acs)
        # portfolio: long-only Top-50, monthly, net 5/15bps, all 21 phases
        s_causal = pd.DataFrame(sig, index=dates, columns=cols).shift(1).values
        nets = np.zeros((21, T)); tr = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            n_, _, at = simulate(s_causal, retv, elig, rb, 50)
            nets[ph] = n_; tr.append(at)
        ex = nets.mean(axis=0) - ew
        print(f"{fac:8}{sgn:>+5}{ic.mean():>+9.4f}{abs(z):>6.1f}{ac21:>+8.2f}{np.mean(tr):>8.2f}x"
              f"{ann(ex):>+10.4f}{ir(ex):>+8.2f}{tstat(ex):>+7.2f}")

    # ---- composite value/quality: z-score average of signed factors ----
    print()
    zs = []
    for fac, sgn in FACTORS.items():
        v = sig_store[fac]
        mu = np.nanmean(np.where(elig, v, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(elig, v, np.nan), axis=1, keepdims=True)
        zs.append(np.where(sd > 0, (v - mu) / sd, np.nan))
    comp = np.nanmean(np.stack(zs), axis=0)
    ic = rank_ic(comp, fwd21, elig); z = ic.mean() / nsd
    s_causal = pd.DataFrame(comp, index=dates, columns=cols).shift(1).values
    for K in [20, 50]:
        nets = np.zeros((21, T)); tr = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            n_, _, at = simulate(s_causal, retv, elig, rb, K)
            nets[ph] = n_; tr.append(at)
        ex = nets.mean(axis=0) - ew
        pp = int(sum(ann(nets[p] - ew) > 0 for p in range(21)))
        tag = "COMPOSITE" if K == 50 else "COMPOSITE"
        print(f"{tag} K={K:<3} RankIC {ic.mean():+.4f} (|z| {abs(z):.1f})  turn/yr {np.mean(tr):.2f}x  "
              f"netEx_EW {ann(ex):+.4f}  IR {ir(ex):+.2f}  t {tstat(ex):+.2f}  ph>0 {pp}/21")

    print("\n=== TURNOVER HEADLINE (annualized one-side, monthly rebalance) ===")
    print("  for comparison this week: 12-1 momentum ~4.4x/yr, fast-technical ~10x/yr,")
    print("  daily-rebalanced long-short ~180-755x/yr. Lower = cheaper to trade.")


if __name__ == "__main__":
    main()
