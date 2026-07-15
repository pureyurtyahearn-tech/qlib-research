"""THE ONE-SHOT HOLDOUT TEST. Pre-registered: factor=$fcfy (FCF yield), sign FIXED from
development (<=2023), evaluated ONCE on the untouched holdout 2024-01-01..2026-06-29.

Nothing here is fit on the holdout. The sign and the entire construction are locked from the
development period. We simply apply the factor and measure. Whatever it shows, it stands.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from ext6_momentum_full import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
DEV_END = "2023-12-31"
HOLD_START, HOLD_END = "2024-01-01", "2026-06-29"


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        g = mask[t] & np.isfinite(sig[t]) & np.isfinite(fwd[t])
        if g.sum() > 30:
            out.append(np.corrcoef(pd.Series(sig[t][g]).rank(), pd.Series(fwd[t][g]).rank())[0, 1])
    return np.array([x for x in out if np.isfinite(x)])


def build(close, mat, fund, s, e):
    w = (close.index >= s) & (close.index <= e)
    dates = close.index[w]; T = len(dates); cols = close.columns
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(columns=cols, fill_value=False)
    retv = close.pct_change().loc[dates].values
    elig = memb.loc[dates].values & np.isfinite(close.loc[dates].values)
    fwd = (close.shift(-22) / close.shift(-1) - 1).loc[dates].values
    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
    fac = fund["$fcfy"].unstack("instrument").reindex(index=dates, columns=cols).values.astype(float)
    return dates, T, cols, retv, elig, fwd, ew, fac


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    fund = pd.read_hdf(SH / "fundamentals_daily.h5")
    close = px["$close"].unstack("ticker").sort_index()

    # --- lock the sign on DEVELOPMENT only ---
    d = build(close, mat, fund, "1999-06-01", DEV_END)
    dates, T, cols, retv, elig, fwd, ew, fac = d
    ic_dev = rank_ic(fac, fwd, elig).mean()
    sign = 1.0 if ic_dev >= 0 else -1.0
    print(f"DEV (<=2023): $fcfy raw RankIC {ic_dev:+.4f} -> LOCKED sign {int(sign):+d} "
          f"(high FCF yield {'LONG' if sign>0 else 'SHORT'})\n")

    # --- apply, untouched, to the holdout ---
    dts, T, cols, retv, elig, fwd, ew, fac = build(close, mat, fund, HOLD_START, HOLD_END)
    print(f"=== HOLDOUT {dts[0].date()} .. {dts[-1].date()}  ({T} days, {T/252:.1f}y) ===")
    print(f"  EW-PIT benchmark: {ann(ew):+.4f}/yr,  members/day {elig.sum(axis=1).mean():.0f}")

    # placebo noise floor ON THE HOLDOUT
    rng = np.random.default_rng(0)
    nsd = float(np.std([rank_ic(rng.standard_normal((T, len(cols))), fwd, elig).mean() for _ in range(15)]))
    ic = rank_ic(fac, fwd, elig)
    print(f"  RankIC {ic.mean():+.5f}  (placebo sd {nsd:.5f}, |z| {abs(ic.mean())/nsd:.1f}, "
          f"daily-IC t {ic.mean()/ic.std()*np.sqrt(len(ic)):+.1f})")

    sig = pd.DataFrame(sign * fac, index=dts, columns=cols).shift(1).values
    print(f"\n  {'':4}{'turn/yr':>9}{'grossEx':>10}{'netEx':>9}{'netIR':>8}{'t':>7}")
    exs = []; turns = []
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        net, gross, turn = simulate(sig, retv, elig, rb, 50)
        exs.append(ann(net - ew)); turns.append(turn)
    exs = np.array(exs)
    # representative phase-0 detail
    rb = np.zeros(T, bool); rb[::21] = True
    net, gross, turn = simulate(sig, retv, elig, rb, 50)
    print(f"  {'ph0':4}{turn:>8.2f}x{ann(gross-ew):>+10.4f}{ann(net-ew):>+9.4f}"
          f"{ir(net-ew):>+8.2f}{tstat(net-ew):>+7.2f}")
    print(f"\n  across all 21 rebalance phases: mean netEx {exs.mean():+.4f}/yr  "
          f"range {exs.min():+.4f}..{exs.max():+.4f}  phases>0 {100*(exs>0).mean():.0f}%")
    print(f"  turnover {np.mean(turns):.2f}x/yr")

    print("\n=== HOLDOUT VERDICT (compare to DEV expectation: netEx ~+2.8%/yr, IC z~33, 100% phases) ===")
    dev_like = "CONFIRMS" if (exs.mean() > 0 and (exs > 0).mean() > 0.7 and abs(ic.mean())/nsd > 2) else "does NOT confirm"
    print(f"  Holdout {dev_like} the development result.")


if __name__ == "__main__":
    main()
