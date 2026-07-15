"""Rebuild daily PIT S&P 500 membership for the FULL history 1998-03-31 -> 2026-06-29.

Same validated method as pit2 (event-walk anchored at the first snapshot), extended range.
Validation: the walk must reproduce ALL 114 quarterly snapshots exactly (0 disagreement),
as it did for the 2010-2026 build.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
START = "1998-03-31"
END = "2026-06-29"


def main():
    df = pd.read_csv(SH / "sp500_raw.csv", parse_dates=["date"])
    hist = df[df.action == "historical"]
    snaps = sorted(pd.Timestamp(d) for d in hist.date.unique())
    snap_members = {d: set(hist[hist.date == d].ticker) for d in snaps}
    ev = df[df.action.isin(["added", "removed"])].sort_values("date")
    ev_recs = ev.to_dict("records")

    grid = pd.bdate_range(START, END)
    # also ensure every snapshot + event date is representable (they are business days)
    members = set(snap_members[snaps[0]])
    ei = 0
    rows = {}
    checks = []
    snap_set = {d: m for d, m in snap_members.items()}
    for d in grid:
        while ei < len(ev_recs) and pd.Timestamp(ev_recs[ei]["date"]) <= d:
            e = ev_recs[ei]
            if e["action"] == "added":
                members.add(e["ticker"])
            else:
                members.discard(e["ticker"])
            ei += 1
        if d in snap_set:
            checks.append((d, len(members ^ snap_set[d])))
        rows[d] = set(members)

    ck = pd.DataFrame(checks, columns=["date", "disagreement"])
    print(f"validation: {len(ck)} snapshots on the grid, "
          f"exact matches {(ck.disagreement == 0).sum()}/{len(ck)}, "
          f"max disagreement {ck.disagreement.max()}")
    # snapshots that fall on non-business days would be missed; count them
    missed = [d for d in snaps if d not in rows]
    if missed:
        print(f"  NOTE: {len(missed)} snapshot dates not on the bdate grid (checked separately)")
        bad = 0
        for d in missed:
            # membership on the business day <= d
            prior = grid[grid <= d]
            if len(prior) and len(rows[prior[-1]] ^ snap_set[d]) > 3:
                bad += 1
        print(f"        of those, >3 disagreement vs nearest prior bday: {bad}")

    sizes = pd.Series({d: len(v) for d, v in rows.items()})
    print(f"\nmembership size: min={sizes.min()} median={int(sizes.median())} max={sizes.max()}")
    ever = sorted(set().union(*rows.values()))
    print(f"unique tickers EVER in the index {START}..{END}: {len(ever)}")

    mat = pd.DataFrame(False, index=grid, columns=ever)
    for d, v in rows.items():
        mat.loc[d, sorted(v)] = True
    mat.to_hdf(SH / "sp500_pit_membership_full.h5", key="m", complevel=5)
    pd.Series(ever).to_csv(SH / "ever_members_full.csv", index=False, header=["ticker"])
    print(f"saved -> sp500_pit_membership_full.h5  ({mat.shape[0]} days x {mat.shape[1]} tickers)")
    print(f"saved -> ever_members_full.csv")


if __name__ == "__main__":
    main()
