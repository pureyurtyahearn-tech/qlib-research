"""What does the qlib store ACTUALLY contain for the NYSE-only names?
The store has 8994 instruments (the original qlib US bundle) but the bundle's data
mostly stops ~2020-11-10; only SP500 + SPY + RSP were extended to 2026. Before writing
anything, find out: are our NYSE names present, and over what dates?"""
import warnings; warnings.filterwarnings("ignore")
import struct
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data"
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def bin_span(inst, field="close"):
    """read a qlib .day.bin: float32 [start_index, v0, v1, ...]"""
    p = STORE / "features" / inst.lower() / f"{field}.day.bin"
    if not p.exists():
        return None
    a = np.fromfile(p, dtype="<f4")
    if len(a) < 2:
        return None
    return int(a[0]), len(a) - 1


def main():
    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    print(f"calendar: {len(cal)} days  {cal[0]} .. {cal[-1]}")

    # all.txt end-date distribution -> what the exchange thinks is tradeable
    ends = Counter()
    for l in open(STORE / "instruments" / "all.txt"):
        parts = l.rstrip("\n").split("\t")
        if len(parts) >= 3:
            ends[parts[2]] += 1
    print("\n=== all.txt END-DATE distribution (top 6) ===")
    for d, n in ends.most_common(6):
        print(f"  {d}   {n:>5} instruments")

    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    close = comb["$close"].unstack("instrument")
    nyse = sorted(set(close.columns) - set(sp))

    present = [t for t in nyse if (STORE / "features" / t.lower()).exists()]
    missing = [t for t in nyse if t not in present]
    print(f"\n=== our {len(nyse)} NYSE-only names vs the store ===")
    print(f"  have a features/ dir in the store: {len(present)}")
    print(f"  MISSING from the store entirely:   {len(missing)}   e.g. {missing[:8]}")

    # for the ones present, what date range does their bin cover?
    spans = []
    for t in present[:600]:
        s = bin_span(t)
        if s:
            i0, n = s
            spans.append((t, cal[i0], cal[min(i0 + n - 1, len(cal) - 1)]))
    df = pd.DataFrame(spans, columns=["t", "first", "last"])
    print(f"\n  of {len(df)} sampled present names, their bin LAST date:")
    for d, n in Counter(df["last"]).most_common(5):
        print(f"    {d}  {n:>5}")

    # do any have data in our NYSE window (2019-2024)?
    need_last = "2024-01-08"
    ok = int((df["last"] >= need_last).sum())
    print(f"\n  present names whose store bin reaches {need_last}: {ok}/{len(df)}")

    # cross-check: does the store agree with daily_pv for a name that IS extended?
    print("\n=== does the store's SP500 data match daily_pv.h5? (spot check) ===")
    for t in ["AAPL", "MSFT"]:
        s = bin_span(t)
        if s:
            i0, n = s
            print(f"  {t}: store bin {cal[i0]} .. {cal[min(i0+n-1, len(cal)-1)]}  ({n} obs)")
            dp = close[t].dropna() if t in close.columns else None
            if dp is not None:
                print(f"        daily_pv  {dp.index[0].date()} .. {dp.index[-1].date()}  ({len(dp)} obs)")


if __name__ == "__main__":
    main()
