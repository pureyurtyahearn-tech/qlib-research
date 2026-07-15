"""Two pre-flight checks before extending the qlib store with NYSE-only names.

(1) POINT-IN-TIME? If the Kaggle NYSE set is survivorship-backfilled (like our SP500
    universe turned out to be), every name runs to the data end. If it is point-in-time,
    names should DIE mid-sample (delist/bankrupt/acquired) and be BORN mid-sample (IPO).
    Distinguishing evidence = the distribution of per-instrument last-dates, and whether
    the names that die look like failures (price collapsing into the exit).

(2) LIQUIDITY, not price. Rank by median dollar volume (close * volume). Dollar volume is
    split-invariant in a way price is not: a split halves price AND doubles share volume,
    so their product is unchanged. That is exactly why a price cutoff wrongly deleted
    NVDA/AMD/AVGO (tiny split-adjusted early prices) while dollar volume will not.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    close = comb["$close"].unstack("instrument").sort_index()
    vol = comb["$volume"].unstack("instrument").sort_index()
    nyse = sorted(set(close.columns) - set(sp))
    cal = close.index
    print(f"calendar {cal.min().date()} .. {cal.max().date()}  ({len(cal)} days)")
    print(f"SP500={len(sp)}  NYSE-only={len(nyse)}\n")

    # ---------- (1) point-in-time ----------
    rows = []
    for t in nyse:
        s = close[t].dropna()
        if len(s) == 0:
            continue
        rows.append((t, s.index[0], s.index[-1], len(s)))
    fl = pd.DataFrame(rows, columns=["t", "first", "last", "n"]).set_index("t")
    print("=== (1) POINT-IN-TIME MEMBERSHIP CHECK (NYSE-only names) ===")
    print(f"  names with data: {len(fl)}")
    last_max = fl["last"].max()
    print(f"  NYSE data actually ends: {last_max.date()}   (SP500 runs to {close[sp].dropna(how='all').index[-1].date()})")

    # exit distribution: how many names stop well before the NYSE data end?
    print("\n  --- EXIT (last date) distribution ---")
    for cut, lbl in [(30, "within 30d of NYSE end (survived)"),
                     (90, "30-90d before end"),
                     (252, "90d-1y before end"),
                     (10**9, "MORE than 1y before end (died mid-sample)")]:
        pass
    gap = (last_max - fl["last"]).dt.days
    buckets = [("survived to end (<30d gap)", gap < 30),
               ("exited 30d-1y before end", (gap >= 30) & (gap < 365)),
               ("exited 1-2y before end", (gap >= 365) & (gap < 730)),
               ("exited >2y before end", gap >= 730)]
    for lbl, m in buckets:
        print(f"    {lbl:34} {int(m.sum()):>5}  ({m.mean():>5.1%})")

    print("\n  --- ENTRY (first date) distribution ---")
    first_min = fl["first"].min()
    g0 = (fl["first"] - first_min).dt.days
    for lbl, m in [("present from data start (<30d)", g0 < 30),
                   ("appeared 30d-1y in", (g0 >= 30) & (g0 < 365)),
                   ("appeared 1-2y in", (g0 >= 365) & (g0 < 730)),
                   ("appeared >2y in (late listing/IPO)", g0 >= 730)]:
        print(f"    {lbl:34} {int(m.sum()):>5}  ({m.mean():>5.1%})")

    # do the exiters look like FAILURES? (price path into the exit)
    died = fl[gap >= 365].index
    if len(died):
        perf = []
        for t in died:
            s = close[t].dropna()
            if len(s) > 60:
                perf.append(s.iloc[-1] / s.iloc[0] - 1)
        perf = np.array(perf)
        surv = []
        for t in fl[gap < 30].index:
            s = close[t].dropna()
            if len(s) > 60:
                surv.append(s.iloc[-1] / s.iloc[0] - 1)
        surv = np.array(surv)
        print(f"\n  --- do mid-sample exiters look like failures? ---")
        print(f"    exiters (n={len(perf)}):  total return over their life  median {np.median(perf):+.3f}  mean {perf.mean():+.3f}")
        print(f"    survivors (n={len(surv)}): total return over their life  median {np.median(surv):+.3f}  mean {surv.mean():+.3f}")
        print(f"    exiters ending below -50%: {(perf < -0.5).mean():.1%}   survivors: {(surv < -0.5).mean():.1%}")

    # ---------- (2) liquidity ----------
    print("\n=== (2) LIQUIDITY: median dollar volume (close * volume) ===")
    dv = (close * vol)
    med_dv = dv.median(axis=0)
    print("  sanity — dollar volume is split-invariant (price/2 * vol*2 = same):")
    for t in ["NVDA", "AMD", "AVGO", "NFLX"]:
        if t in close.columns:
            p0 = close[t].dropna().iloc[0]
            print(f"    {t:5} first split-adj price ${p0:>8.2f}  median $vol ${med_dv[t]/1e6:>10.1f}M"
                  f"   <- a $2 PRICE filter would have deleted this; $vol does not")
    for grp, lbl in [(sp, "SP500"), (nyse, "NYSE-only")]:
        m = med_dv[[c for c in grp if c in med_dv.index]].dropna()
        qs = m.quantile([.01, .05, .10, .25, .50, .75, .90])
        print(f"\n  {lbl} median-$vol percentiles ($M):")
        print("    " + "  ".join(f"p{int(q*100)}={v/1e6:.2f}" for q, v in qs.items()))

    print("\n  --- candidate thresholds on NYSE-only ---")
    mn = med_dv[nyse].dropna()
    msp = med_dv[sp].dropna()
    print(f"  {'threshold':>14}{'NYSE kept':>11}{'% kept':>9}{'SP500 kept':>12}")
    for thr in [0, 1e5, 5e5, 1e6, 5e6, 1e7, 2e7]:
        print(f"  ${thr/1e6:>12.2f}M{int((mn >= thr).sum()):>11}{(mn >= thr).mean():>8.1%}"
              f"{int((msp >= thr).sum()):>10}/{len(msp)}")

    # how much of the NYSE set is genuinely untradeable junk?
    print("\n  --- what the low end looks like (bottom decile by $vol) ---")
    lo = mn.nsmallest(int(len(mn) * .10))
    lopx = close[lo.index].median()
    print(f"    n={len(lo)}   median $vol ${lo.median()/1e6:.3f}M   median price ${lopx.median():.2f}")
    print(f"    share with median price < $1: {(lopx < 1).mean():.1%}")
    print(f"    median # days with data: {int(fl.loc[[t for t in lo.index if t in fl.index], 'n'].median())}")

    med_dv.to_frame("med_dollar_vol").to_csv(SRC / "median_dollar_volume.csv")
    print(f"\n  wrote {SRC/'median_dollar_volume.csv'}")


if __name__ == "__main__":
    main()
