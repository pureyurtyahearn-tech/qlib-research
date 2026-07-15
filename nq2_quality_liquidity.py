"""Verify NASDAQ data quality year-by-year and apply the dollar-volume liquidity filter.
Memory-conscious: operates on the long (25M-row) frame, avoids full unstack where possible.

Eligibility = listed & trading (has a price bar). Coverage/NaN measured within each name's
active span. Liquidity store gate = median daily $vol >= $1M (split-invariant; never price).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
STORE_GATE = 1e6


def main():
    p = pd.read_hdf(SH / "sep_nasdaq_panel.h5")
    p = p.reset_index()
    p["year"] = p["date"].dt.year
    print(f"panel: {len(p):,} rows, {p.ticker.nunique():,} tickers, "
          f"{p.date.min().date()}..{p.date.max().date()}\n")

    # ---- liquidity: median dollar volume per ticker (split-invariant) ----
    p["dv"] = p["$close"] * p["$volume"]
    med_dv = p.groupby("ticker")["dv"].median()
    keep = med_dv[med_dv >= STORE_GATE].index
    print(f"=== LIQUIDITY FILTER (median $vol >= ${STORE_GATE/1e6:.0f}M store gate) ===")
    print(f"  tickers: {p.ticker.nunique():,} -> kept {len(keep):,} "
          f"({len(keep)/p.ticker.nunique():.0%}); dropped {p.ticker.nunique()-len(keep):,} illiquid")
    q = med_dv.quantile([.1, .25, .5, .75, .9])
    print(f"  median-$vol percentiles ($M): " + "  ".join(f"p{int(k*100)}={v/1e6:.2f}" for k, v in q.items()))
    kept_dv = med_dv[keep]
    print(f"  kept universe median-$vol ($M): p10={kept_dv.quantile(.1)/1e6:.1f} "
          f"p50={kept_dv.median()/1e6:.1f} p90={kept_dv.quantile(.9)/1e6:.1f}")
    biggest = med_dv.nlargest(8).index.tolist()
    print(f"  most liquid NASDAQ names: {', '.join(biggest)}")
    pd.Series(sorted(keep)).to_csv(SH / "nasdaq_liquid_universe.csv", index=False, header=["ticker"])
    print(f"  saved nasdaq_liquid_universe.csv ({len(keep)} names)\n")

    # restrict quality analysis to the kept (liquid) universe -- that's what we'll trade
    pk = p[p.ticker.isin(set(keep))].copy()
    pk = pk.sort_values(["ticker", "date"])
    pk["ret"] = pk.groupby("ticker")["$close"].pct_change()

    print("=== QUALITY BY YEAR (liquid universe only) ===")
    print(f"{'year':>6}{'names':>8}{'rows':>10}{'zeroVol%':>10}{'flat%':>8}{'px<$1%':>9}{'medPx':>8}")
    rows = []
    for y in range(1998, 2027):
        s = pk[pk.year == y]
        if len(s) < 500:
            continue
        n = s.ticker.nunique()
        zv = (s["$volume"] == 0).mean()
        flat = (s["ret"].abs() < 1e-12).mean()
        sub1 = (s["$close"] < 1).mean()
        medpx = s["$close"].median()
        rows.append((y, n, zv, flat, sub1))
        print(f"{y:>6}{n:>8}{len(s):>10,}{zv:>9.2%}{flat:>8.2%}{sub1:>8.2%}{medpx:>8.2f}")

    qd = pd.DataFrame(rows, columns=["year", "n", "zv", "flat", "sub1"])
    base = qd[qd.year >= 2015]
    print(f"\n  2015+ baseline: zeroVol {base.zv.mean():.2%}  flat {base.flat.mean():.2%}  px<$1 {base.sub1.mean():.2%}")
    print(f"  years with zeroVol>3% or flat>8% or px<$1>5% (materially worse):")
    flagged = qd[(qd.zv > 0.03) | (qd.flat > 0.08) | (qd.sub1 > 0.05)]
    if len(flagged):
        for _, r in flagged.iterrows():
            print(f"    {int(r.year)}: zeroVol {r.zv:.2%} flat {r.flat:.2%} px<$1 {r.sub1:.2%}")
    else:
        print("    none -- quality consistent across all years")

    # ---- interior NaN within active spans (liquid universe) ----
    print("\n=== COVERAGE / NaN (liquid universe, within active spans) ===")
    span = nan = 0
    cw = pk.pivot_table(index="date", columns="ticker", values="$close")
    for t in cw.columns:
        s = cw[t]; fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None: continue
        seg = s.loc[fv:lv]; span += len(seg); nan += seg.isna().sum()
    print(f"  interior NaN rate (gaps inside active span): {nan/span:.4%}")
    print(f"  active names/day: median {int(cw.notna().sum(axis=1).median())}, "
          f"1999: {int(cw.loc['1999'].notna().sum(axis=1).median()) if '1999' in cw.index.strftime('%Y') else 0}, "
          f"2020: {int(cw.loc['2020'].notna().sum(axis=1).median())}")


if __name__ == "__main__":
    main()
