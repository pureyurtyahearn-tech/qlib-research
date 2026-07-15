"""Write an equal-weight PIT-S&P500 index into us_data_pit_full as a benchmark instrument
(SP500EW). This is the correct matched benchmark for factor evaluation (equal-weight of the
same point-in-time universe) -- SPY/RSP aren't in Sharadar SEP, and EW-own-universe is what
we've benchmarked against all week anyway.

Index level_t = level_{t-1} * (1 + mean daily return of members priced on t). Written as a
flat OHLC = level, so qlib's benchmark return calc reproduces the EW return exactly.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]
NAME = "SP500EW"


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = px["$close"].unstack("ticker").sort_index()
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(columns=close.columns, fill_value=False)
    ret = close.pct_change()
    elig = memb & close.notna()
    ewret = ret.where(elig).mean(axis=1)                 # daily EW return of members
    ewret.iloc[0] = 0.0
    level = 100.0 * (1 + ewret.fillna(0)).cumprod()

    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    cal_i = {d: i for i, d in enumerate(cal)}
    idx = np.array([cal_i.get(d.strftime("%Y-%m-%d"), -1) for d in level.index])
    ok = idx >= 0
    level = level[ok]; idx = idx[ok]; ewret = ewret[ok]
    i0 = int(idx[0])
    lv = level.values.astype("<f4")
    chg = np.full(len(lv), np.nan, dtype="<f4"); chg[1:] = lv[1:] / lv[:-1] - 1.0

    fd = STORE / "features" / NAME.lower(); fd.mkdir(parents=True, exist_ok=True)
    vals = {"open": lv, "close": lv, "high": lv, "low": lv,
            "volume": np.zeros(len(lv), "<f4"), "factor": np.ones(len(lv), "<f4"), "change": chg}
    for f in FEATURES:
        buf = np.empty(len(lv) + 1, dtype="<f4"); buf[0] = np.float32(i0); buf[1:] = vals[f]
        buf.tofile(fd / f"{f}.day.bin")
    print(f"wrote {NAME}: {len(lv)} bars, {cal[i0]}..{cal[idx[-1]]}, level {lv[0]:.1f}->{lv[-1]:.1f}")

    ap = STORE / "instruments" / "all.txt"
    lines = [l.rstrip("\n") for l in ap.read_text().splitlines() if l.strip()]
    lines = [l for l in lines if l.split("\t")[0] != NAME]
    lines.append(f"{NAME}\t{cal[i0]}\t{cal[idx[-1]]}")
    ap.write_text("\n".join(lines) + "\n")
    # also add to sp500pit.txt so it resolves under that market if needed (benchmark lookup)
    print(f"added {NAME} to all.txt ({len(lines)} lines)")
    print(f"sanity: EW index total return {lv[-1]/lv[0]-1:+.1%} over {len(lv)/252:.1f}y "
          f"= {((lv[-1]/lv[0])**(252/len(lv))-1):+.2%}/yr")


if __name__ == "__main__":
    main()
