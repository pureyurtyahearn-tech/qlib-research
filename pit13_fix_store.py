"""Fix part 1/2: give the exchange something to sell.

95 of 329 index exits are same-day delistings (acquisitions: ABMD, AET, AGN, ATVI, BCR...).
The stock stops trading on the very day it leaves the index, so on the first non-member day
there is no price -- qlib's is_stock_tradable() returns False, the SELL is skipped, and the
position is frozen forever. That is the ghost-position bug.

Fix: append ONE flat bar (open=high=low=close = last close, volume = last volume) on the
first non-member day for any ticker whose price history ends while it is still a member.
The forced sell then executes at the last traded price.

Why this is not look-ahead: the bar is a synthetic liquidation print, not information. It
carries a 0% return, sits on a day the name is NOT an index member (so the strategy can
never BUY it), and simply lets the backtest realise proceeds at the last close -- which is
exactly what happens economically when a deal closes, and exactly what our pandas simulator
already does.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit"
SH = Path("git_ignore_folder/sharadar")
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]


def read_bin(p):
    a = np.fromfile(p, dtype="<f4")
    return int(a[0]), a[1:]


def main():
    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    cal_i = {d: i for i, d in enumerate(cal)}
    panel = pd.read_hdf(SH / "sep_panel.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)

    # NOTE: the price can stop BEFORE membership ends (AET: last price 2018-11-28, still a
    # member to 2018-11-30). One bar is not enough -- we must pad flat all the way THROUGH
    # the first non-member day, which is the day the forced sell fires.
    todo = []
    for t in memb.columns:
        m = memb[t]
        if not m.any():
            continue
        px = close[t]
        last_px = px.last_valid_index()
        if last_px is None:
            continue
        if not bool(m.loc[last_px]):
            continue                       # already de-indexed while still priced: fine
        last_memb = m[m].index[-1]
        nxt = memb.index[memb.index > last_memb]
        if len(nxt) == 0:
            continue                       # still a member at the end of the data
        first_non = nxt[0]                 # the day the forced sell must execute
        i0 = cal_i.get(last_px.strftime("%Y-%m-%d"))
        i1 = cal_i.get(first_non.strftime("%Y-%m-%d"))
        if i0 is None or i1 is None or i1 <= i0:
            continue
        todo.append((t, i0, i1))

    print(f"tickers whose price ends WHILE still an index member: {len(todo)}")
    print(f"  e.g. {[t for t, _, _ in todo[:10]]}")
    pads = [i1 - i0 for _, i0, i1 in todo]
    print(f"  bars needed to reach the first non-member day: "
          f"min={min(pads)} median={int(np.median(pads))} max={max(pads)}")

    listing = {}
    n = nbar = 0
    for t, i0, i1 in todo:
        fd = STORE / "features" / t.lower()
        if not fd.exists():
            continue
        s0, cl = read_bin(fd / "close.day.bin")
        have_to = s0 + len(cl) - 1              # last calendar index currently in the bins
        if have_to >= i1:                       # idempotent: already padded
            listing[t] = cal[have_to]
            continue
        k = i1 - have_to                        # bars to append, through the first non-member day
        lastc = float(cl[-1])
        _, vol = read_bin(fd / "volume.day.bin")
        lastv = float(vol[-1]) if len(vol) and np.isfinite(vol[-1]) else 0.0
        add = {"open": lastc, "high": lastc, "low": lastc, "close": lastc,
               "volume": lastv, "factor": 1.0, "change": 0.0}
        for f in FEATURES:
            p = fd / f"{f}.day.bin"
            if p.exists():
                np.full(k, add[f], dtype="<f4").tofile(open(p, "ab"))
        listing[t] = cal[i1]
        n += 1
        nbar += k
    print(f"  padded {n} tickers with {nbar} liquidation bars total")

    # all.txt must cover the new final day, or the exchange will not see it
    ap = STORE / "instruments" / "all.txt"
    lines = [l.rstrip("\n") for l in ap.read_text().splitlines() if l.strip()]
    upd = 0
    for i, ln in enumerate(lines):
        p = ln.split("\t")
        if p[0] in listing:
            p[2] = listing[p[0]]
            lines[i] = "\t".join(p)
            upd += 1
    ap.write_text("\n".join(lines) + "\n")
    print(f"  all.txt end-dates updated for {upd} tickers")

    # verify: a known same-day delisting is now priced on its first non-member day
    for t in ["ATVI", "AET", "ABMD"]:
        fd = STORE / "features" / t.lower()
        if not fd.exists():
            continue
        s0, cl = read_bin(fd / "close.day.bin")
        end = cal[s0 + len(cl) - 1]
        m = memb[t]
        lastm = m[m].index[-1].strftime("%Y-%m-%d")
        print(f"  {t:6} last membership {lastm}   bins now end {end}   "
              f"{'OK (sellable)' if end > lastm else 'STILL STUCK'}")


if __name__ == "__main__":
    main()
