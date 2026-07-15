"""Phase 1 verdict: how far back are PRICES clean enough to trust? Picks the start date.

For each year, among the ACTUAL index members that year, measure:
  - price coverage: % of members with a price on a typical day (must be ~100%)
  - NaN/gap rate within a member's active span
  - zero-volume day rate (a data-quality red flag: no trading recorded)
  - flat-price rate (close==prev close exactly: stale/illiquid prints)
  - median price level and count of implausibly small prices
A year is "usable" only if coverage ~100% AND the quality flags are comparable to the
2010+ era we already trust.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")


def main():
    panel = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    vol = panel["$volume"].unstack("ticker").sort_index()
    # align membership to the price trading calendar
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)
    print(f"price panel {close.index.min().date()}..{close.index.max().date()}  "
          f"{close.shape[0]} days x {close.shape[1]} tickers\n")

    print(f"{'year':>6}{'members':>9}{'coverage':>10}{'NaN%':>8}{'zeroVol%':>10}"
          f"{'flat%':>8}{'medPx':>8}{'px<$1':>7}")
    rows = []
    for y in range(1998, 2027):
        dm = memb.index.year == y
        if dm.sum() < 40:
            continue
        M = memb.loc[dm]                        # (days, tickers) membership this year
        C = close.loc[dm]
        V = vol.loc[dm]
        member_cols = M.any(axis=0)
        M = M.loc[:, member_cols]; C = C.loc[:, member_cols]; V = V.loc[:, member_cols]
        # cells that SHOULD have a price (member on that day)
        should = M.values
        has = np.isfinite(C.values) & should
        coverage = has.sum() / should.sum()
        nan_rate = 1 - coverage
        # among priced member-cells:
        priced = np.isfinite(C.values) & should
        zerovol = ((V.values == 0) & priced).sum() / priced.sum()
        chg = C.pct_change().values
        flat = ((chg == 0) & priced[1:].astype(bool) if False else (np.abs(chg) < 1e-12) & priced).sum() / priced.sum()
        medpx = np.nanmedian(np.where(priced, C.values, np.nan))
        sub1 = ((C.values < 1) & priced).sum() / priced.sum()
        avg_members = should.sum(axis=1).mean() if should.ndim > 1 else should.mean()
        rows.append((y, coverage, nan_rate, zerovol, flat, sub1))
        print(f"{y:>6}{int(M.any(axis=0).sum()):>9}{coverage:>9.1%}{nan_rate:>8.2%}"
              f"{zerovol:>9.2%}{flat:>8.2%}{medpx:>8.2f}{sub1:>7.2%}")

    q = pd.DataFrame(rows, columns=["year", "cov", "nanr", "zvol", "flat", "sub1"])
    base = q[q.year >= 2015]
    print(f"\n=== 2015+ baseline (the era we trust) ===")
    print(f"  coverage {base['cov'].mean():.3%}   zeroVol {base['zvol'].mean():.3%}   "
          f"flat {base['flat'].mean():.3%}   px<$1 {base['sub1'].mean():.3%}")
    # flag years materially worse than baseline
    print(f"\n=== years materially worse than 2015+ baseline ===")
    for _, r in q.iterrows():
        flags = []
        if r["cov"] < 0.999: flags.append(f"coverage {r['cov']:.2%}")
        if r["zvol"] > base["zvol"].mean() * 3 + 0.005: flags.append(f"zeroVol {r['zvol']:.2%}")
        if r["flat"] > base["flat"].mean() * 3 + 0.005: flags.append(f"flat {r['flat']:.2%}")
        if flags:
            print(f"  {int(r.year)}: " + ", ".join(flags))
    print("  (no lines above = every year back to 1998 matches the modern-era quality bar)")


if __name__ == "__main__":
    main()
