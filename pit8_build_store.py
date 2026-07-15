"""Rebuild the qlib store with a TRUE point-in-time SP500 universe.

The fix: a stock is tradeable on date t ONLY if it was an actual S&P 500 member on t.

Two separate mechanisms, and both are needed:
  1. PRICE BINS cover each ticker's full price history (so a 252-day momentum lookback can
     see a name's data BEFORE it entered the index -- that is legitimate, the prices were
     public at the time).
  2. The INSTRUMENT LIST (`sp500pit.txt`) encodes membership SPANS. qlib's listing file
     supports multiple rows per ticker, so a name that joined in 2012 and left in 2019 gets
     a row 2012->2019 and is simply not in the universe outside it. Names with several index
     stints get several rows. THIS is what makes the backtest survivorship-free.

We write to a SEPARATE store dir (us_data_pit) rather than mutating us_data: the old store
is what every previous result was computed on, and clobbering it would destroy the ability
to compare old vs new.
"""
import warnings; warnings.filterwarnings("ignore")
import shutil
import numpy as np, pandas as pd
from pathlib import Path

OLD = Path.home() / ".qlib" / "qlib_data" / "us_data"
NEW = Path.home() / ".qlib" / "qlib_data" / "us_data_pit"
SH = Path("git_ignore_folder/sharadar")
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]


def spans(series):
    """bool Series indexed by date -> list of (start,end) membership spans"""
    v = series.values
    idx = series.index
    out = []
    i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j + 1 < len(v) and v[j + 1]:
                j += 1
            out.append((idx[i], idx[j]))
            i = j + 1
        else:
            i += 1
    return out


def main():
    panel = pd.read_hdf(SH / "sep_panel.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    print(f"SEP panel: {close.shape[0]} dates x {close.shape[1]} tickers")

    # calendar = union of trading dates in the price panel
    cal = [d.strftime("%Y-%m-%d") for d in close.index]
    cal_idx = {d: i for i, d in enumerate(cal)}
    NEW.mkdir(parents=True, exist_ok=True)
    (NEW / "calendars").mkdir(exist_ok=True)
    (NEW / "instruments").mkdir(exist_ok=True)
    (NEW / "features").mkdir(exist_ok=True)
    (NEW / "calendars" / "day.txt").write_text("\n".join(cal) + "\n")
    print(f"calendar: {len(cal)} days  {cal[0]} .. {cal[-1]}")

    wide = {c.strip("$"): panel[c].unstack("ticker").sort_index().reindex(
        index=close.index, columns=close.columns) for c in ["$open", "$high", "$low", "$close", "$volume"]}

    # ---- write bins (full price history per ticker) ----
    n = 0
    listed = {}
    for t in close.columns:
        c = wide["close"][t]
        ok = np.where(c.notna().values)[0]
        if len(ok) < 30:
            continue
        i0, i1 = int(ok[0]), int(ok[-1])
        m = i1 - i0 + 1
        fd = NEW / "features" / t.lower()
        fd.mkdir(parents=True, exist_ok=True)
        cl = wide["close"][t].values[i0:i1 + 1].astype("<f4")
        for f in FEATURES:
            if f == "factor":
                arr = np.ones(m, dtype="<f4")
            elif f == "change":
                arr = np.full(m, np.nan, dtype="<f4")
                arr[1:] = cl[1:] / cl[:-1] - 1.0
            else:
                arr = wide[f][t].values[i0:i1 + 1].astype("<f4")
            buf = np.empty(m + 1, dtype="<f4")
            buf[0] = np.float32(i0)
            buf[1:] = arr
            buf.tofile(fd / f"{f}.day.bin")
        listed[t] = (cal[i0], cal[i1])
        n += 1
    print(f"wrote bins for {n} tickers")

    # ---- all.txt: full price-history spans (needed so factors can look back pre-entry) ----
    (NEW / "instruments" / "all.txt").write_text(
        "\n".join(f"{t}\t{s}\t{e}" for t, (s, e) in sorted(listed.items())) + "\n")

    # ---- sp500pit.txt: MEMBERSHIP spans -- the survivorship fix ----
    trading = pd.DatetimeIndex(close.index)
    memb = mat.reindex(index=trading, method="ffill").fillna(False)
    rows = []
    multi = 0
    for t in memb.columns:
        if t not in listed:
            continue
        sp = spans(memb[t])
        if len(sp) > 1:
            multi += 1
        for s, e in sp:
            rows.append(f"{t}\t{s.strftime('%Y-%m-%d')}\t{e.strftime('%Y-%m-%d')}")
    (NEW / "instruments" / "sp500pit.txt").write_text("\n".join(sorted(rows)) + "\n")
    print(f"sp500pit.txt: {len(rows)} membership spans over "
          f"{len(set(r.split(chr(9))[0] for r in rows))} tickers "
          f"({multi} tickers had MULTIPLE index stints)")

    miss = [t for t in memb.columns[memb.any()] if t not in listed]
    print(f"\nPIT members with NO price bins: {len(miss)}  {miss[:10]}")

    # membership count over time -- must sit at ~500
    cnt = memb.sum(axis=1)
    print(f"\nmembership count on trading days: min={cnt.min()} median={int(cnt.median())} max={cnt.max()}")
    for y in [2010, 2014, 2018, 2022, 2026]:
        s = cnt[cnt.index.year == y]
        if len(s):
            print(f"  {y}: mean {s.mean():.0f}")

    # how many members have prices, by year -- THE survivorship check
    print("\ncoverage: % of actual index members with price data, by year")
    for y in range(2010, 2027):
        m = memb[memb.index.year == y]
        if not len(m):
            continue
        tot = m.sum(axis=1).mean()
        have = m[[c for c in m.columns if c in listed]].sum(axis=1).mean()
        px = wide["close"].reindex(m.index)
        withpx = (m & px[m.columns].notna()).sum(axis=1).mean()
        print(f"  {y}: members {tot:5.0f}   with price data {withpx:5.0f}  ({withpx/tot:6.1%})")


if __name__ == "__main__":
    main()
