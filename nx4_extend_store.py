"""Write the filtered NYSE universe into the qlib binary store.

IMPORTANT — why we OVERWRITE rather than append: the store already holds bundle-vintage
bins for ~1703 of these tickers, ending 2020-11-10, from a DIFFERENT adjustment lineage
than daily_pv.h5 (which is our split-fixed + winsorized data). Splicing two adjustment
lineages at a boundary is precisely the class of bug that produced the fake 4x/8x cliffs
earlier. So each selected NYSE name's bins are rewritten end-to-end from daily_pv.h5,
giving one consistent source. Affected dirs are backed up first.

Bin format (verified against extend_benchmark.py): float32 LE, [start_idx, v0, v1, ...]
where start_idx indexes calendars/day.txt. factor=1.0 (prices already adjusted; qlib
region='us' sets trade_unit=None so factor is not used for lot rounding).
"""
import warnings; warnings.filterwarnings("ignore")
import shutil, sys
import numpy as np, pandas as pd
from pathlib import Path

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data"
SRC = Path("git_ignore_folder/factor_implementation_source_data")
BACKUP = Path("git_ignore_folder/_store_backup_nyse")
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]
DRY = "--dry" in sys.argv


def main():
    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    cal_idx = {d: i for i, d in enumerate(cal)}

    tick = pd.read_csv(SRC / "nyse_store_universe.csv")["ticker"].tolist()
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    wide = {c.strip("$"): comb[c].unstack("instrument").sort_index() for c in
            ["$open", "$high", "$low", "$close", "$volume"]}
    dates = [d.strftime("%Y-%m-%d") for d in wide["close"].index]

    missing_cal = [d for d in dates if d not in cal_idx]
    print(f"daily_pv dates not in qlib calendar: {len(missing_cal)}"
          + (f"  e.g. {missing_cal[:5]}" if missing_cal else "  (clean)"))
    pos = np.array([cal_idx.get(d, -1) for d in dates])

    print(f"writing {len(tick)} NYSE names   (dry-run={DRY})")
    if not DRY:
        BACKUP.mkdir(parents=True, exist_ok=True)
        n_bak = 0
        for t in tick:
            src = STORE / "features" / t.lower()
            dst = BACKUP / t.lower()
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst); n_bak += 1
        print(f"  backed up {n_bak} existing feature dirs -> {BACKUP}")
        ap = STORE / "instruments" / "all.txt"
        if not (STORE / "instruments" / "all.txt.prenyse").exists():
            shutil.copy(ap, STORE / "instruments" / "all.txt.prenyse")
            print("  backed up all.txt -> all.txt.prenyse")

    listing = {}
    written = 0
    for t in tick:
        c = wide["close"][t]
        ok = np.where(c.notna().values)[0]
        if len(ok) < 252:
            continue
        i0, i1 = ok[0], ok[-1]
        sidx, eidx = pos[i0], pos[i1]
        if sidx < 0 or eidx < 0:
            continue
        n = eidx - sidx + 1
        # place daily_pv values onto the calendar grid (NaN where the name has no bar)
        grid = {f: np.full(n, np.nan, dtype="<f4") for f in FEATURES}
        rows = np.arange(i0, i1 + 1)
        slots = pos[rows] - sidx
        good = slots >= 0
        for f in ["open", "high", "low", "close", "volume"]:
            grid[f][slots[good]] = wide[f][t].values[rows][good].astype("<f4")
        grid["factor"][:] = 1.0
        cl = grid["close"]
        ch = np.full(n, np.nan, dtype="<f4")
        ch[1:] = cl[1:] / cl[:-1] - 1.0
        grid["change"] = ch

        listing[t] = (cal[sidx], cal[eidx])
        if not DRY:
            fd = STORE / "features" / t.lower()
            fd.mkdir(parents=True, exist_ok=True)
            for f in FEATURES:
                arr = np.empty(n + 1, dtype="<f4")
                arr[0] = np.float32(sidx)
                arr[1:] = grid[f]
                arr.tofile(fd / f"{f}.day.bin")
        written += 1

    print(f"  wrote bins for {written} names")
    ends = pd.Series([v[1] for v in listing.values()]).value_counts()
    print(f"  their listing end-dates (top 3): {dict(ends.head(3))}")

    if DRY:
        print("dry run: no files written."); return

    # ---- instruments/all.txt: add/refresh the NYSE names ----
    ap = STORE / "instruments" / "all.txt"
    lines = [l.rstrip("\n") for l in ap.read_text().splitlines() if l.strip()]
    idx = {}
    for i, ln in enumerate(lines):
        idx[ln.split("\t")[0].strip().upper()] = i
    added = updated = 0
    for t, (s, e) in listing.items():
        row = f"{t}\t{s}\t{e}"
        if t.upper() in idx:
            lines[idx[t.upper()]] = row; updated += 1
        else:
            lines.append(row); added += 1
    ap.write_text("\n".join(lines) + "\n")
    print(f"  all.txt: {updated} updated, {added} added -> {len(lines)} lines")

    # ---- a dedicated instrument list for the combined universe ----
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    spmap = {}
    for ln in lines:
        p = ln.split("\t")
        spmap[p[0].strip().upper()] = (p[1], p[2])
    out = []
    for t in sp:
        if t.upper() in spmap:
            out.append(f"{t}\t{spmap[t.upper()][0]}\t{spmap[t.upper()][1]}")
    for t, (s, e) in sorted(listing.items()):
        out.append(f"{t}\t{s}\t{e}")
    (STORE / "instruments" / "combo.txt").write_text("\n".join(out) + "\n")
    print(f"  wrote instruments/combo.txt  ({len(out)} names = {len(sp)} SP500 + {len(listing)} NYSE)")


if __name__ == "__main__":
    main()
