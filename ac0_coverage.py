"""Data coverage audit before building 12-1 momentum.
How far back does usable SP500 + RSP data actually go?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    sp = [l.split("\t")[0].strip() for l in
          open(Path.home()/".qlib"/"qlib_data"/"us_data"/"instruments"/"sp500.txt")]
    sp = sorted(set(sp))
    print(f"sp500.txt tickers: {len(sp)}")

    px = D.features(sp, ["$close"], start_time="1999-12-31", end_time="2026-06-29").iloc[:, 0]
    px.index = px.index.set_names(["instrument", "datetime"])
    w = px.unstack("instrument").sort_index()
    print(f"panel: {w.shape[0]} days x {w.shape[1]} instruments\n")

    print(f"{'year':>6}{'n_with_data':>13}{'median_px':>11}")
    for y in range(2000, 2027):
        sub = w.loc[str(y)] if str(y) in w.index.strftime("%Y") else None
        rows = w[w.index.year == y]
        if len(rows) == 0:
            continue
        n = int(rows.notna().any(axis=0).sum())
        med = float(np.nanmedian(rows.values))
        print(f"{y:>6}{n:>13}{med:>11.2f}")

    for b in ["RSP", "SPY"]:
        try:
            s = D.features([b], ["$close"], start_time="1999-12-31", end_time="2026-06-29").iloc[:, 0]
            s = s.dropna()
            d = s.index.get_level_values("datetime")
            print(f"\n{b}: {len(s)} obs   {d.min().date()} .. {d.max().date()}")
        except Exception as e:
            print(f"\n{b}: ERROR {e}")

if __name__ == "__main__":
    main()
