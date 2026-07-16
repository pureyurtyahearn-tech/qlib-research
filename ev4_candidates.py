"""MILESTONE 2 candidate screen — names worth pulling straddles for.

Filters (covered-call premium-harvest shortlist):
  - median earnings |move| 4-8% (interesting but not binary),
  - p90 <= 18% (excludes biotech-binary tails),
  - >= 20 events of history (stable profile),
  - still active + reporting (last event recent),
  - options-grade: trailing 60d median dollar volume >= $25M/day AND last price >= $15,
  - PREDICTED next earnings inside [today+14d, today+42d] (2-6 weeks) — predicted as
    last detected event + the name's own median inter-event gap (~91d cadence).
    Verify the actual date in IBKR when pulling the chain.

Output: candidates table + straddle_candidates.csv (feeds straddle_quotes.csv workflow).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
TODAY = pd.Timestamp("2026-07-16")
WIN_LO, WIN_HI = TODAY + pd.Timedelta(days=14), TODAY + pd.Timedelta(days=42)


def main():
    prof = pd.read_csv(SH / "earnings_move_profiles.csv", parse_dates=["last_event"])
    ev = pd.read_hdf(SH / "earnings_events.h5")

    # predicted next earnings from each name's own cadence
    gaps = (ev.sort_values(["ticker", "event_day"])
              .groupby("ticker")["event_day"].apply(lambda s: s.diff().dt.days.tail(8).median()))
    prof = prof.set_index("ticker")
    prof["gap"] = gaps
    prof["next_est"] = prof["last_event"] + pd.to_timedelta(prof["gap"], unit="D")

    # profile filters
    c = prof[(prof.med_move >= 0.04) & (prof.med_move <= 0.08)
             & (prof.p90 <= 0.18) & (prof.n >= 20)
             & (prof.last_event >= "2026-02-01")
             & (prof.next_est >= WIN_LO) & (prof.next_est <= WIN_HI)].copy()
    print(f"profile+window filter: {len(c)} names (from {len(prof)})")

    # liquidity/price from the panels (trailing 60d, options-grade)
    liq = {}
    for pf in ["sep_panel_full.h5", "sep_nasdaq_panel.h5"]:
        px = pd.read_hdf(SH / pf)
        cl = px["$close"].unstack("ticker"); vo = px["$volume"].unstack("ticker")
        for t in c.index.intersection(cl.columns):
            s = cl[t].dropna().tail(60)
            if len(s) < 40:
                continue
            dv = (s * vo[t].reindex(s.index)).median()
            liq.setdefault(t, (float(s.iloc[-1]), float(dv)))
        del px, cl, vo
    c["price"] = [liq.get(t, (np.nan, np.nan))[0] for t in c.index]
    c["dvol"] = [liq.get(t, (np.nan, np.nan))[1] for t in c.index]
    c = c[(c.price >= 15) & (c.dvol >= 25e6)]
    print(f"+ options-grade liquidity (px>=$15, $vol>=$25M/d): {len(c)} names\n")

    c = c.sort_values("trail4", ascending=False)
    print(f"{'tkr':>6} {'univ':>7} {'next_est':>11} {'med':>6} {'p90':>6} {'trail4':>7} "
          f"{'P>5%':>5} {'P>10%':>6} {'px':>8} {'$vol/d':>8} {'n':>4}")
    for t, r in c.iterrows():
        print(f"{t:>6} {r.universe:>7} {str(r.next_est.date()):>11} {r.med_move:>6.1%} "
              f"{r.p90:>6.1%} {r.trail4:>7.1%} {r.p_gt5:>5.0%} {r.p_gt10:>6.0%} "
              f"{r.price:>8.2f} {r.dvol/1e6:>7.0f}M {int(r.n):>4}")
    out = c[["universe", "next_est", "med_move", "p75", "p90", "trail4",
             "p_gt5", "p_gt10", "price", "dvol", "n"]]
    out.to_csv(SH / "straddle_candidates.csv")
    print(f"\nsaved straddle_candidates.csv ({len(out)} names) -- pull pre/post-earnings "
          f"straddles for these in IBKR, append to straddle_quotes.csv, run ev3")


if __name__ == "__main__":
    main()
