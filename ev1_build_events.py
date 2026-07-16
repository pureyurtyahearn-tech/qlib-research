"""MILESTONE 1a — build the earnings-event panel across SP500 PIT + NASDAQ-only universes.

Event detection (validated by the 25-name probe): the earnings move LEADS the SF1 10-Q
filing date (datekey) by ~5 days (press-release/8-K day). So for each quarterly filing,
the announcement day is DETECTED as the max |close-to-close return| trading day inside
[datekey-12, datekey+2]. Volume ratio on that day is recorded as a detection diagnostic
(real earnings days have volume spikes).

Per event: detected day, signed move, |move|, overnight gap (the un-manageable part for
covered calls), volume ratio, trailing baseline vol (60d std ending 5d before the event,
causal), normalized move = |move| / baseline.

Output: git_ignore_folder/sharadar/earnings_events.h5  (one row per (ticker, event)).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
WIN_PRE, WIN_POST = 12, 2         # calendar days around datekey to search
MIN_EVENT_SPACING = 60            # days; dedup amended/duplicate filings


def build(universe, px_file, sf1_file, only=None):
    px = pd.read_hdf(SH / px_file)
    close = px["$close"].unstack("ticker").sort_index()
    opn = px["$open"].unstack("ticker").sort_index()
    vol = px["$volume"].unstack("ticker").sort_index()
    f = pd.read_hdf(SH / sf1_file)
    f["datekey"] = pd.to_datetime(f["datekey"])
    tickers = sorted(set(f.ticker) & set(close.columns))
    if only is not None:
        tickers = [t for t in tickers if t in only]
    rows = []
    for n, t in enumerate(tickers):
        cs = close[t].dropna()
        if len(cs) < 300:
            continue
        ret = cs.pct_change()
        gap = (opn[t].reindex(cs.index) / cs.shift(1) - 1)
        vr = (vol[t].reindex(cs.index) /
              vol[t].reindex(cs.index).rolling(60, min_periods=20).median())
        base = ret.rolling(60, min_periods=30).std().shift(5)     # causal baseline vol
        dks = np.sort(f.loc[f.ticker == t, "datekey"].dropna().unique())
        kept = []
        for dk in dks:                                            # dedup close filings
            if not kept or (dk - kept[-1]) / np.timedelta64(1, "D") >= MIN_EVENT_SPACING:
                kept.append(dk)
        for dk in kept:
            dk = pd.Timestamp(dk)
            w = ret.loc[dk - pd.Timedelta(days=WIN_PRE): dk + pd.Timedelta(days=WIN_POST)].dropna()
            if len(w) < 4:
                continue
            eday = w.abs().idxmax()
            b = base.get(eday, np.nan)
            rows.append((universe, t, dk, eday, (eday - dk).days,
                         float(w[eday]), float(abs(w[eday])),
                         float(gap.get(eday, np.nan)), float(vr.get(eday, np.nan)),
                         float(b) if np.isfinite(b) else np.nan))
        if (n + 1) % 500 == 0:
            print(f"  [{universe}] {n+1}/{len(tickers)} tickers, {len(rows)} events", flush=True)
    print(f"  [{universe}] done: {len(tickers)} tickers -> {len(rows)} events", flush=True)
    return rows


def main():
    nas_only = set(pd.read_csv(SH / "nasdaq_only_tickers.csv")["ticker"].astype(str))
    rows = build("SP500", "sep_panel_full.h5", "sf1_art_raw.h5")
    rows += build("NASDAQ", "sep_nasdaq_panel.h5", "sf1_nasdaq_raw.h5", only=nas_only)
    ev = pd.DataFrame(rows, columns=["universe", "ticker", "datekey", "event_day", "offset",
                                     "move", "absmove", "gap", "volratio", "base_vol"])
    ev["norm"] = ev["absmove"] / ev["base_vol"]
    ev = ev.sort_values(["ticker", "event_day"]).reset_index(drop=True)
    ev.to_hdf(SH / "earnings_events.h5", key="ev", complevel=5)

    print(f"\n=== EVENT PANEL ===")
    print(f"events {len(ev):,}  tickers {ev.ticker.nunique():,}  "
          f"{ev.event_day.min().date()}..{ev.event_day.max().date()}")
    for u, g in ev.groupby("universe"):
        print(f"  {u}: {len(g):,} events, {g.ticker.nunique():,} names, "
              f"median |move| {g.absmove.median():.2%}, median norm {g.norm.median():.1f}sigma")
    print(f"\ndetection diagnostics:")
    print(f"  median volume ratio on detected day: {ev.volratio.median():.2f}x "
          f"(frac >1.5x: {(ev.volratio>1.5).mean():.0%})  -- real earnings days spike volume")
    print(f"  offset (event day - datekey): median {ev.offset.median():.0f}d, "
          f"p10..p90 {ev.offset.quantile(.1):.0f}..{ev.offset.quantile(.9):.0f}d")
    print(f"  |move| percentiles: p50 {ev.absmove.quantile(.5):.2%}  "
          f"p75 {ev.absmove.quantile(.75):.2%}  p90 {ev.absmove.quantile(.9):.2%}")
    print(f"  gap share of move: median |gap|/|move| "
          f"{(ev.gap.abs()/ev.absmove).replace([np.inf],np.nan).median():.0%}")


if __name__ == "__main__":
    main()
