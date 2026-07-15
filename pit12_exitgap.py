"""Can a forced exit actually EXECUTE? Feasibility check for the qlib fix.

To sell a removed name, the exchange must still have a price for it on the day we act.
The strategy only learns a name has left the index on the day AFTER its last membership day.
So the question is: on (last_membership_day + 1), does a price still exist?

If yes for (nearly) all names -> a clean strategy-level fix works: force-sell on exit.
If no  -> some names delist and de-index simultaneously and must be written off at the last
          close, which needs exchange/position surgery rather than an order.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")


def main():
    panel = pd.read_hdf(SH / "sep_panel.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)
    memb = memb.loc[memb.index >= "2010-01-04"]
    cl = close.reindex(memb.index)

    gap = []
    for t in memb.columns:
        m = memb[t].values
        if not m.any():
            continue
        # every day where the name WAS a member yesterday and is NOT today = an exit
        ex = np.where(m[:-1] & ~m[1:])[0] + 1     # index of first non-member day
        for i in ex:
            priced_today = np.isfinite(cl[t].values[i])
            # how many more days does a price exist after the exit?
            fut = np.isfinite(cl[t].values[i:])
            n_after = int(fut.sum())
            gap.append((t, memb.index[i].date(), priced_today, n_after))

    g = pd.DataFrame(gap, columns=["ticker", "first_nonmember_day", "priced_that_day", "priced_days_after"])
    print(f"index exits in 2010-2026: {len(g)}")
    print(f"  priced on the first non-member day (sellable immediately): "
          f"{g.priced_that_day.sum()}/{len(g)}  ({g.priced_that_day.mean():.1%})")
    stuck = g[~g.priced_that_day]
    print(f"  NOT priced -> cannot be sold with a normal order: {len(stuck)}")
    if len(stuck):
        print("\n  these need a write-off at last close (or an earlier exit rule):")
        print(stuck.head(15).to_string(index=False))
    print(f"\n  distribution of priced_days_after (how long we still CAN trade them):")
    print(f"    0 days: {(g.priced_days_after==0).sum()}   1-5: {g.priced_days_after.between(1,5).sum()}"
          f"   6-21: {g.priced_days_after.between(6,21).sum()}   >21: {(g.priced_days_after>21).sum()}")


if __name__ == "__main__":
    main()
