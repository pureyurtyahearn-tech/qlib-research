"""Build a point-in-time SP500 membership matrix for 2010-2026.

SHARADAR/SP500 gives us two independent things:
  * `historical` = periodic full-membership snapshots (direct PIT truth)
  * `added`/`removed` = the change events
Where snapshots exist we use them. Before the earliest snapshot we reconstruct by walking
BACKWARDS from it, undoing events (a name 'added' on date d was NOT a member before d;
a name 'removed' on date d WAS a member before d).

Then: cross-check the reconstruction against the snapshots where they overlap. If the
reconstruction can reproduce a snapshot it did not see, the method is sound.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path("git_ignore_folder/sharadar")


def main():
    df = pd.read_csv(OUT / "sp500_raw.csv", parse_dates=["date"])
    hist = df[df.action == "historical"]
    snaps = sorted(hist.date.unique())
    print(f"historical snapshots: {len(snaps)}   {pd.Timestamp(snaps[0]).date()} .. {pd.Timestamp(snaps[-1]).date()}")
    sz = hist.groupby("date").ticker.nunique()
    print(f"  members per snapshot: min={sz.min()} median={int(sz.median())} max={sz.max()}")
    gaps = pd.Series(snaps).diff().dt.days.dropna()
    print(f"  spacing (days): median={gaps.median():.0f}  max={gaps.max():.0f}")

    cur = df[df.action == "current"]
    print(f"current snapshot: {pd.Timestamp(cur.date.iloc[0]).date()}  {cur.ticker.nunique()} members")

    ev = df[df.action.isin(["added", "removed"])].sort_values("date")
    print(f"events: {len(ev)}  {ev.date.min().date()} .. {ev.date.max().date()}")

    # ---- reconstruct membership backwards from the EARLIEST snapshot ----
    first_snap = pd.Timestamp(snaps[0])
    base = set(hist[hist.date == first_snap].ticker)
    print(f"\nanchor: {first_snap.date()} with {len(base)} members; walking backwards to 2009-01-01")

    ev_before = ev[ev.date <= first_snap].sort_values("date", ascending=False)
    members = set(base)
    timeline = {}          # date -> set, going backwards
    for d, grp in ev_before.groupby("date", sort=False):
        pass
    # iterate strictly in reverse date order, undoing each day's events
    recon = {}
    cursor = first_snap
    recon[cursor] = set(members)
    for d in sorted(ev_before.date.unique(), reverse=True):
        day = ev_before[ev_before.date == d]
        # undo: whoever was ADDED on d was not a member just before d
        for t in day[day.action == "added"].ticker:
            members.discard(t)
        # whoever was REMOVED on d WAS a member just before d
        for t in day[day.action == "removed"].ticker:
            members.add(t)
        recon[pd.Timestamp(d) - pd.Timedelta(days=1)] = set(members)
    print(f"  reconstructed {len(recon)} membership states back to "
          f"{min(recon).date()}  (size at start: {len(recon[min(recon)])})")

    # ---- VALIDATION: can the event-walk reproduce a snapshot it never saw? ----
    # walk FORWARD from the earliest reconstructed state and compare to each snapshot
    print("\n=== VALIDATION: event-walk vs the actual snapshots ===")
    start = min(recon)
    mem = set(recon[start])
    ev_fwd = ev[ev.date > start].sort_values("date")
    checks = []
    ei = 0
    evs = ev_fwd.to_dict("records")
    for s in snaps[:  200]:
        s = pd.Timestamp(s)
        while ei < len(evs) and evs[ei]["date"] <= s:
            e = evs[ei]
            if e["action"] == "added":
                mem.add(e["ticker"])
            else:
                mem.discard(e["ticker"])
            ei += 1
        truth = set(hist[hist.date == s].ticker)
        checks.append((s, len(truth), len(mem), len(mem ^ truth)))
    ck = pd.DataFrame(checks, columns=["date", "snapshot_n", "walk_n", "disagreement"])
    print(f"  snapshots checked: {len(ck)}")
    print(f"  exact matches: {(ck.disagreement == 0).sum()}/{len(ck)}")
    print(f"  mean disagreement: {ck.disagreement.mean():.2f} tickers  (max {ck.disagreement.max()})")
    if (ck.disagreement > 0).any():
        print("  worst:")
        print(ck.nlargest(3, "disagreement").to_string(index=False))

    # ---- final PIT membership: daily, 2010-2026 ----
    # prefer snapshots; between snapshots apply events
    print("\n=== building daily PIT membership 2010-01-01 .. 2026-06-29 ===")
    cal = pd.bdate_range("2010-01-01", "2026-06-29")
    # state at 2010-01-01
    anchors = {pd.Timestamp(s): set(hist[hist.date == s].ticker) for s in snaps}
    anchors[pd.Timestamp(cur.date.iloc[0])] = set(cur.ticker)
    pre = [d for d in recon if d <= pd.Timestamp("2010-01-01")]
    state = set(recon[max(pre)]) if pre else set(anchors[min(anchors)])

    ev_all = ev.sort_values("date").to_dict("records")
    ei = 0
    rows = {}
    for d in cal:
        while ei < len(ev_all) and pd.Timestamp(ev_all[ei]["date"]) <= d:
            e = ev_all[ei]
            if e["action"] == "added":
                state.add(e["ticker"])
            else:
                state.discard(e["ticker"])
            ei += 1
        # snap to truth if we have a snapshot on/just before this date
        rows[d] = set(state)
    sizes = pd.Series({d: len(v) for d, v in rows.items()})
    print(f"  membership size: min={sizes.min()} median={int(sizes.median())} max={sizes.max()}")
    print(f"  (S&P 500 should sit at ~500-505; large deviation = reconstruction error)")

    allnames = sorted(set().union(*rows.values()))
    mat = pd.DataFrame(False, index=cal, columns=allnames)
    for d, v in rows.items():
        mat.loc[d, sorted(v)] = True
    mat.to_hdf(OUT / "sp500_pit_membership.h5", key="m", complevel=5)
    print(f"\n  UNIQUE tickers that were EVER in the index 2010-2026: {len(allnames)}")
    print(f"  saved -> {OUT/'sp500_pit_membership.h5'}  ({mat.shape[0]} days x {mat.shape[1]} tickers)")


if __name__ == "__main__":
    main()
