"""Extend the qlib binary store (us_data_pit_full) with the Sharadar-native NYSE-only
universe (1,168 tickers, sep_nyse_panel.h5). Writes real price bins + an
instruments/nyse.txt listing with each name's ACTUAL trading span (first..last price
date), and adds them to all.txt so the backtest exchange knows exactly when each is
tradeable. No index-membership spans -- eligibility IS the trading span here, same
pattern as nq4_extend_store.py's NASDAQ extension (no liquidation-bar padding: there is
no membership matrix for this universe to outlive its price data, so nothing to pad).

sep_nyse_panel.h5 is already the final liquidity-filtered, SP500-deduplicated universe
(nx3_select.py's $1M store gate, minus ever_members_full.csv) -- no further filtering
needed here, unlike nq4 which had to intersect the raw NASDAQ panel against a separate
liquid-universe CSV.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]


def main():
    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    cal_i = {d: i for i, d in enumerate(cal)}
    panel = pd.read_hdf(SH / "sep_nyse_panel.h5")
    wide = {c.strip("$"): panel[c].unstack("ticker").sort_index()
            for c in ["$open", "$high", "$low", "$close", "$volume"]}
    close = wide["close"]

    # calendar alignment check (both Sharadar SEP -> should match exactly)
    pdates = [d.strftime("%Y-%m-%d") for d in close.index]
    off_cal = [d for d in pdates if d not in cal_i]
    print(f"NYSE panel dates not in store calendar: {len(off_cal)}"
          + (f"  e.g. {off_cal[:3]}" if off_cal else "  (calendars match)"))
    pos = np.array([cal_i.get(d, -1) for d in pdates])

    existing = {p.name for p in (STORE / "features").iterdir()}      # lowercase dirs
    n_new = n_skip = 0
    listing = {}
    for t in close.columns:
        c = close[t]
        ok = np.where(c.notna().values)[0]
        if len(ok) < 30:
            continue
        i0, i1 = int(ok[0]), int(ok[-1])
        s_idx, e_idx = int(pos[i0]), int(pos[i1])
        if s_idx < 0 or e_idx < 0:
            continue
        listing[t] = (cal[s_idx], cal[e_idx])
        if t.lower() in existing:            # already in store (NASDAQ pull overlap, identical data)
            n_skip += 1
            continue
        n = e_idx - s_idx + 1
        grid = {f: np.full(n, np.nan, dtype="<f4") for f in FEATURES}
        rows = np.arange(i0, i1 + 1)
        slot = pos[rows] - s_idx
        good = slot >= 0
        for f in ["open", "high", "low", "close", "volume"]:
            grid[f][slot[good]] = wide[f][t].values[rows][good].astype("<f4")
        grid["factor"][:] = 1.0
        cl = grid["close"]
        ch = np.full(n, np.nan, dtype="<f4"); ch[1:] = cl[1:] / cl[:-1] - 1.0
        grid["change"] = ch
        fd = STORE / "features" / t.lower(); fd.mkdir(parents=True, exist_ok=True)
        for f in FEATURES:
            buf = np.empty(n + 1, dtype="<f4"); buf[0] = np.float32(s_idx); buf[1:] = grid[f]
            buf.tofile(fd / f"{f}.day.bin")
        n_new += 1
    print(f"wrote bins: {n_new} new NYSE names, {n_skip} already in store (NASDAQ pull overlap, identical)")

    # instruments/nyse.txt = trading spans
    rows = [f"{t}\t{s}\t{e}" for t, (s, e) in sorted(listing.items())]
    (STORE / "instruments" / "nyse.txt").write_text("\n".join(rows) + "\n")
    print(f"instruments/nyse.txt: {len(rows)} names")

    # add NYSE names to all.txt (skip those already present)
    ap = STORE / "instruments" / "all.txt"
    lines = [l.rstrip("\n") for l in ap.read_text().splitlines() if l.strip()]
    have = {l.split("\t")[0] for l in lines}
    added = 0
    for t, (s, e) in sorted(listing.items()):
        if t not in have:
            lines.append(f"{t}\t{s}\t{e}"); added += 1
    ap.write_text("\n".join(lines) + "\n")
    print(f"all.txt: +{added} NYSE names -> {len(lines)} total instruments")

    # coverage sanity: active NYSE names per year
    print("\nactive NYSE names/year (from nyse.txt spans):")
    ls = pd.DataFrame([(t, pd.Timestamp(s), pd.Timestamp(e)) for t, (s, e) in listing.items()],
                      columns=["t", "s", "e"])
    for y in [2000, 2005, 2010, 2015, 2020, 2025]:
        d = pd.Timestamp(f"{y}-06-30")
        print(f"  {y}: {int(((ls.s <= d) & (ls.e >= d)).sum())}")


if __name__ == "__main__":
    main()
