"""Phase 1 probe (local, no API): how far back is MEMBERSHIP data clean?
Uses the already-downloaded SHARADAR/SP500 table. Prices are probed separately (ext2).

We must not assume 1998 is usable just because snapshots exist. Check:
  - snapshot spacing over the FULL history (are they really quarterly back to 1998, or
    sparse/irregular early?)
  - membership size per snapshot (should sit at ~500; big deviations = thin early data)
  - add/remove event density by year (a plausible ~20-25/yr; near-zero early = incomplete)
  - how many unique tickers were EVER members for candidate start years
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")


def main():
    df = pd.read_csv(SH / "sp500_raw.csv", parse_dates=["date"])
    hist = df[df.action == "historical"]
    snaps = sorted(hist.date.unique())
    print(f"historical snapshots: {len(snaps)}   {pd.Timestamp(snaps[0]).date()} .. {pd.Timestamp(snaps[-1]).date()}")

    sz = hist.groupby("date").ticker.nunique()
    sp = pd.Series(snaps)
    gap = sp.diff().dt.days
    print("\n=== snapshot spacing + size, by 3-year era (is early data as good as recent?) ===")
    print(f"  {'era':13}{'#snaps':>8}{'gap_med':>9}{'gap_max':>9}{'size_min':>10}{'size_med':>10}")
    for lo in range(1998, 2027, 3):
        hi = lo + 3
        m = (sp.dt.year >= lo) & (sp.dt.year < hi)
        if not m.any():
            continue
        ds = sp[m]
        gg = ds.diff().dt.days.dropna()
        szs = sz.loc[ds.values]
        print(f"  {lo}-{hi-1:<8}{int(m.sum()):>8}{gg.median() if len(gg) else float('nan'):>9.0f}"
              f"{gg.max() if len(gg) else float('nan'):>9.0f}{szs.min():>10}{int(szs.median()):>10}")

    ev = df[df.action.isin(["added", "removed"])].copy()
    ev["year"] = ev.date.dt.year
    print("\n=== add/remove events per year (near-zero early = incomplete change log) ===")
    piv = ev.pivot_table(index="year", columns="action", values="ticker", aggfunc="count").fillna(0).astype(int)
    for y in range(1998, 2027):
        if y in piv.index:
            print(f"  {y}: added {piv.loc[y].get('added',0):>3}  removed {piv.loc[y].get('removed',0):>3}")

    # ever-member ticker count for candidate start years (drives the price pull size)
    print("\n=== unique tickers EVER in the index, by candidate start year -> 2026 ===")
    for start in [1998, 2000, 2003, 2005, 2010]:
        names = set()
        # union of all snapshot members with date >= start, plus names removed after start
        names |= set(hist[hist.date >= f"{start}-01-01"].ticker)
        names |= set(ev[(ev.date >= f"{start}-01-01")].ticker)
        print(f"  from {start}: ~{len(names)} unique tickers")

    # earliest snapshot membership sample (are these real large caps or junk?)
    first = pd.Timestamp(snaps[0])
    fm = sorted(hist[hist.date == first].ticker)
    print(f"\n  first snapshot {first.date()} sample members: {fm[:12]}")


if __name__ == "__main__":
    main()
