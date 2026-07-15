"""Does the PIT universe actually differ from what we have? Show the missing names.

Our current SP500 universe was built by taking TODAY's constituents and backfilling prices
to 2010. If that is survivorship-biased, then names that were IN the index during 2010-2026
but got REMOVED (bankruptcy, acquisition, index demotion) should be ABSENT from our data.
This quantifies exactly who is missing and how much index-time they account for.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    mat = pd.read_hdf(OUT / "sp500_pit_membership.h5")
    raw = pd.read_csv(OUT / "sp500_raw.csv", parse_dates=["date"])
    ours = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                      .index.get_level_values("instrument").unique()))
    close = pd.read_hdf(SRC / "daily_pv.h5")["$close"].unstack("instrument")
    ours_withdata = sorted([t for t in ours if close[t].notna().sum() > 0])

    ever = sorted(mat.columns[mat.any(axis=0)])
    today = sorted(mat.columns[mat.iloc[-1]])
    print(f"PIT: tickers EVER in the S&P 500, 2010-2026 : {len(ever)}")
    print(f"PIT: tickers in the index on the last day    : {len(today)}")
    print(f"OURS: SP500 universe in daily_pv.h5          : {len(ours)}  ({len(ours_withdata)} with price data)")

    missing = [t for t in ever if t not in set(ours)]
    extra = [t for t in ours if t not in set(ever)]
    print(f"\n>>> IN THE INDEX AT SOME POINT BUT MISSING FROM OUR DATA: {len(missing)}")
    print(f">>> in our data but never in the index 2010-2026        : {len(extra)}"
          f"   {extra[:8] if extra else ''}")

    # index-time accounted for by the missing names
    tot_days = int(mat.values.sum())
    miss_days = int(mat[missing].values.sum())
    print(f"\nSURVIVORSHIP HOLE (share of index membership-days we cannot see):")
    print(f"  total ticker-days of index membership 2010-2026 : {tot_days:,}")
    print(f"  of which held by names we DON'T have            : {miss_days:,}  ({miss_days/tot_days:.1%})")
    n_miss_by_year = {}
    for y in range(2010, 2027):
        m = mat[mat.index.year == y]
        if len(m) == 0:
            continue
        avg_missing = m[missing].sum(axis=1).mean()
        n_miss_by_year[y] = avg_missing
    print(f"\n  average # index members per day that we are MISSING, by year:")
    print("   " + "  ".join(f"{y}:{v:.0f}" for y, v in n_miss_by_year.items()))

    # why did they leave? use the removal events
    rem = raw[(raw.action == "removed") & (raw.date >= "2010-01-01")]
    rem = rem.sort_values("date").drop_duplicates("ticker", keep="last")
    rmap = rem.set_index("ticker")
    print(f"\n=== the {len(missing)} missing names: when and why they left ===")
    rows = []
    for t in missing:
        if t in rmap.index:
            r = rmap.loc[t]
            rows.append((t, str(r["name"])[:28], pd.Timestamp(r["date"]).date(),
                         str(r["contraname"])[:26]))
        else:
            # left before 2010? no -- they're in `ever` for 2010+. So: still in index but
            # renamed/reticker, or removed with no event captured
            last = mat.index[mat[t]][-1]
            rows.append((t, "(no removal event)", pd.Timestamp(last).date(), ""))
    dd = pd.DataFrame(rows, columns=["ticker", "name", "left", "replaced_by/acquirer"])
    dd = dd.sort_values("left")
    print(f"\n  EARLIEST 15 departures:")
    print(dd.head(15).to_string(index=False))
    print(f"\n  MOST RECENT 15 departures:")
    print(dd.tail(15).to_string(index=False))

    # the ones that matter most: big names, and outright FAILURES
    print(f"\n=== names that were in the index for the LONGEST and are missing ===")
    dur = mat[missing].sum(axis=0).sort_values(ascending=False)
    for t in dur.head(20).index:
        nm = dd[dd.ticker == t].iloc[0]
        yrs = dur[t] / 252
        print(f"  {t:6} {nm['name']:30} in index {yrs:5.1f}y, left {nm['left']}"
              f"  -> {nm['replaced_by/acquirer']}")

    dd.to_csv(OUT / "missing_constituents.csv", index=False)
    print(f"\nsaved -> {OUT/'missing_constituents.csv'}")

    # sanity: do these missing names exist in the WIDER daily_pv (NYSE-only) set?
    have_any = [t for t in missing if t in close.columns]
    print(f"\n  of the {len(missing)} missing, present anywhere in daily_pv (incl NYSE set): {len(have_any)}")
    print(f"  -> we would still need to SOURCE PRICES for {len(missing)-len(have_any)} of them")


if __name__ == "__main__":
    main()
