"""Choose the NYSE universe to load into the qlib store. Dry-run selection report.

Two separate gates, deliberately:
  STORE GATE (what data to load at all): median $vol >= $1M, >=252 obs, common stock.
    Generous on purpose -- loading a name does not mean trading it.
  TRADING SCREEN (applied causally at each rebalance, inside the backtest):
    trailing 63-day median $vol >= $5M.
    A full-sample liquidity criterion would be a look-ahead (we'd be using 2024 liquidity
    to pick 2020 holdings). Since we are here BECAUSE of a survivorship look-ahead, we are
    not going to introduce a liquidity one. The store gate only decides what is loaded.

$5M justification: at K=20 equal weight, a $10M book holds $500k per name. The standard
participation limit is <=10% of ADV, so a tradeable name needs ADV >= $500k/0.10 = $5M.
"""
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np, pandas as pd
from pathlib import Path

SRC = Path("git_ignore_folder/factor_implementation_source_data")
STORE_GATE = 1e6
TRADE_GATE = 5e6
JUNK = re.compile(r"(\.(W|WS|U|S|T|RT|P)$)|(-[A-Z]$)", re.I)   # warrants/units/preferreds/classes


def main():
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    close = comb["$close"].unstack("instrument").sort_index()
    vol = comb["$volume"].unstack("instrument").sort_index()
    nyse = sorted(set(close.columns) - set(sp))
    dv = close * vol
    med = dv.median(axis=0)
    nobs = close.notna().sum(axis=0)

    print(f"NYSE-only candidates: {len(nyse)}")
    junk = [t for t in nyse if JUNK.search(t)]
    print(f"\n=== non-common-stock ticker patterns (warrants/units/preferreds) ===")
    print(f"  matched: {len(junk)}   e.g. {junk[:10]}")
    jm = med[junk].dropna()
    print(f"  their median $vol: p50=${jm.median()/1e6:.3f}M   share already under the "
          f"$1M store gate: {(jm < STORE_GATE).mean():.1%}")
    print("  -> excluded explicitly anyway: momentum on a warrant is meaningless.")

    step = {}
    keep = [t for t in nyse if not JUNK.search(t)]
    step["after ticker-pattern exclusion"] = len(keep)
    keep = [t for t in keep if nobs.get(t, 0) >= 252]
    step["after >=252 obs (need 12m lookback)"] = len(keep)
    keep_store = [t for t in keep if med.get(t, 0) >= STORE_GATE]
    step[f"after median $vol >= ${STORE_GATE/1e6:.0f}M (STORE GATE)"] = len(keep_store)
    keep_trade = [t for t in keep_store if med.get(t, 0) >= TRADE_GATE]
    step[f"  (of which median $vol >= ${TRADE_GATE/1e6:.0f}M)"] = len(keep_trade)

    print(f"\n=== selection funnel ===")
    print(f"  {'start (NYSE-only)':44}{len(nyse):>6}")
    for k, v in step.items():
        print(f"  {k:44}{v:>6}")

    print(f"\n=== what the STORE GATE keeps ({len(keep_store)} names) ===")
    m = med[keep_store]
    print(f"  median $vol: p10=${m.quantile(.10)/1e6:.1f}M  p50=${m.median()/1e6:.1f}M  "
          f"p90=${m.quantile(.90)/1e6:.1f}M  max=${m.max()/1e6:.0f}M")
    px = close[keep_store].median()
    print(f"  median price: p10=${px.quantile(.10):.2f}  p50=${px.median():.2f}")
    print(f"  biggest by $vol: {', '.join(m.nlargest(8).index)}")
    print(f"  smallest kept:   {', '.join(m.nsmallest(8).index)}")

    print(f"\n=== resulting universe ===")
    print(f"  SP500          {len(sp):>5}")
    print(f"  + NYSE (store) {len(keep_store):>5}")
    print(f"  = total        {len(sp)+len(keep_store):>5}   "
          f"({(len(sp)+len(keep_store))/len(sp):.1f}x the SP500-only breadth)")
    print(f"\n  BUT: NYSE data spans only 2019-01..2024-01-08, so the COMBINED-universe test")
    print(f"       window is ~2019-2024 (minus 252d momentum burn-in) = ~4 years,")
    print(f"       vs 15.5 years for SP500-only. More breadth, much less history.")

    pd.Series(sorted(keep_store)).to_csv(SRC / "nyse_store_universe.csv", index=False, header=["ticker"])
    print(f"\nwrote {SRC/'nyse_store_universe.csv'}")


if __name__ == "__main__":
    main()
