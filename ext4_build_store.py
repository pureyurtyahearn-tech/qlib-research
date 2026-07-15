"""Build the FULL-history PIT qlib store from Sharadar SEP (single source of truth).
Integrates pit8 (bins + membership spans) and pit13 (liquidation bars for same-day
delistings) into one pass. New store: ~/.qlib/qlib_data/us_data_pit_full
(us_data_pit is left intact until this is verified).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

NEW = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]


def spans(series):
    v = series.values; idx = series.index; out = []; i = 0
    while i < len(v):
        if v[i]:
            j = i
            while j + 1 < len(v) and v[j + 1]:
                j += 1
            out.append((idx[i], idx[j])); i = j + 1
        else:
            i += 1
    return out


def main():
    panel = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = panel["$close"].unstack("ticker").sort_index()
    wide = {c.strip("$"): panel[c].unstack("ticker").sort_index().reindex(
        index=close.index, columns=close.columns) for c in ["$open", "$high", "$low", "$close", "$volume"]}
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(
        columns=close.columns, fill_value=False)

    cal = [d.strftime("%Y-%m-%d") for d in close.index]
    cal_i = {d: i for i, d in enumerate(cal)}
    for sub in ["calendars", "instruments", "features"]:
        (NEW / sub).mkdir(parents=True, exist_ok=True)
    (NEW / "calendars" / "day.txt").write_text("\n".join(cal) + "\n")
    print(f"calendar {cal[0]}..{cal[-1]}  ({len(cal)} days), {close.shape[1]} tickers")

    listed = {}
    npad = 0
    for t in close.columns:
        c = wide["close"][t]
        ok = np.where(c.notna().values)[0]
        if len(ok) < 30:
            continue
        i0, i1 = int(ok[0]), int(ok[-1])
        # pit13: if price ends while still a member, pad flat through the first non-member day
        last_px = close.index[i1]
        mt = memb[t]
        if bool(mt.loc[last_px]) and mt.any():
            last_memb = mt[mt].index[-1]
            nxt = memb.index[memb.index > last_memb]
            if len(nxt):
                i_ext = cal_i.get(nxt[0].strftime("%Y-%m-%d"))
                if i_ext and i_ext > i1:
                    i1 = i_ext; npad += 1
        m = i1 - i0 + 1
        cl = wide["close"][t].values[i0:i1 + 1].astype("<f4")
        # forward-fill the padded tail (flat liquidation)
        if np.isnan(cl[-1]):
            lastv = cl[np.isfinite(cl)][-1]
            cl = pd.Series(cl).ffill().values.astype("<f4")
        fd = NEW / "features" / t.lower()
        fd.mkdir(parents=True, exist_ok=True)
        for f in FEATURES:
            if f == "factor":
                arr = np.ones(m, dtype="<f4")
            elif f == "change":
                arr = np.full(m, np.nan, dtype="<f4"); arr[1:] = cl[1:] / cl[:-1] - 1.0
            elif f == "close":
                arr = cl
            else:
                a = wide[f][t].values[i0:i1 + 1].astype("<f4")
                a = pd.Series(a).ffill().values.astype("<f4")   # flat-fill padded tail
                arr = a
            buf = np.empty(m + 1, dtype="<f4"); buf[0] = np.float32(i0); buf[1:] = arr
            buf.tofile(fd / f"{f}.day.bin")
        listed[t] = (cal[i0], cal[i1])
    print(f"wrote bins for {len(listed)} tickers ({npad} padded with liquidation tails)")

    (NEW / "instruments" / "all.txt").write_text(
        "\n".join(f"{t}\t{s}\t{e}" for t, (s, e) in sorted(listed.items())) + "\n")

    rows = []; multi = 0
    for t in memb.columns:
        if t not in listed:
            continue
        sp = spans(memb[t])
        if len(sp) > 1:
            multi += 1
        for s, e in sp:
            rows.append(f"{t}\t{s.strftime('%Y-%m-%d')}\t{e.strftime('%Y-%m-%d')}")
    (NEW / "instruments" / "sp500pit.txt").write_text("\n".join(sorted(rows)) + "\n")
    print(f"sp500pit.txt: {len(rows)} spans / {len(set(r.split(chr(9))[0] for r in rows))} tickers "
          f"({multi} multi-stint)")

    print("\ncoverage: % of actual index members with price data, by year")
    for y in range(1998, 2027):
        mm = memb[memb.index.year == y]
        if not len(mm):
            continue
        tot = mm.sum(axis=1).mean()
        px = wide["close"].reindex(mm.index)
        withpx = (mm & px[mm.columns].notna()).sum(axis=1).mean()
        print(f"  {y}: members {tot:5.0f}   with price {withpx:5.0f}  ({withpx/tot:6.1%})")


if __name__ == "__main__":
    main()
