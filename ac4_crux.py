"""The crux: alpha rises sharply as K shrinks (K=20 -> +7.4%/yr, t=1.83), but the full-sample
result leans hard on 2024 + a 122-day partial 2026. Does the concentration effect survive
removing them? If K=20 collapses without 2026, the 'top of the book carries the signal'
story is really just 'recent months carry the signal'."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from ac2_portfolio import simulate, ann, ir, SRC, FDIR
from ac3_robust import tstat


def main():
    import qlib
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "us_data"), region="us")
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sig12 = pd.read_hdf(FDIR / "mom_12_1.h5")
    close = comb["$close"].unstack("instrument").sort_index()[sig12.columns].loc[sig12.index]
    ret_w = close.pct_change()
    dates = ret_w.index; T = len(dates)
    retv = ret_w.values; ewv = ret_w.mean(axis=1).values
    sv = sig12.shift(1).values

    windows = {
        "full 2011-2026":      np.ones(T, bool),
        "excl 2026 (partial)": (dates.year < 2026),
        "excl 2024 & 2026":    ((dates.year < 2026) & (dates.year != 2024)),
        "2011-2018 (1st half)": (dates.year <= 2018),
        "2019-2025 (2nd half)": ((dates.year >= 2019) & (dates.year < 2026)),
    }
    print("net excess vs EW own universe, monthly rebal, phase-averaged, 20bps round trip")
    print(f"{'window':22}" + "".join(f"{'K='+str(k):>16}" for k in [20, 50, 100]))
    cache = {}
    for k in [20, 50, 100]:
        nn = np.zeros((21, T))
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            nn[ph], _, _, _ = simulate(sv, retv, rb, k=k)
        cache[k] = nn.mean(axis=0) - ewv
    for wn, m in windows.items():
        cells = []
        for k in [20, 50, 100]:
            e = cache[k][m]
            cells.append(f"{ann(e):+.4f}(t{tstat(e):+.2f})")
        print(f"{wn:22}" + "".join(f"{c:>16}" for c in cells))

    # how concentrated is the 2026 gain? top contributors in the K=20 book
    print("\n=== 2026 sanity: is the +64% net a few names or broad? (K=20, phase 0) ===")
    m26 = np.asarray(dates.year == 2026)
    rb = np.zeros(T, bool); rb[0::21] = True
    idx26 = np.where(m26)[0]
    sig26 = sig12.shift(1)
    picks = {}
    for t in idx26:
        if not rb[t]:
            continue
        row = sig26.iloc[t].dropna()
        for nm in row.nlargest(20).index:
            picks[nm] = picks.get(nm, 0) + 1
    held = sorted(picks, key=picks.get, reverse=True)
    r26 = (close.loc[dates[m26]].iloc[-1] / close.loc[dates[m26]].iloc[0] - 1)
    print(f"  names held in 2026 K=20 book: {len(held)}")
    top = r26[held].sort_values(ascending=False)
    print(f"  their 2026 returns: median {top.median():+.3f}  mean {top.mean():+.3f}")
    print(f"  best 5: {', '.join(f'{i} {v:+.2f}' for i, v in top.head(5).items())}")
    print(f"  worst 5: {', '.join(f'{i} {v:+.2f}' for i, v in top.tail(5).items())}")
    print(f"  EW universe 2026 return: {(close.loc[dates[m26]].iloc[-1]/close.loc[dates[m26]].iloc[0]-1).median():+.3f} (median name)")
    print(f"  share of held names beating universe median: "
          f"{(top > r26.median()).mean():.0%}")


if __name__ == "__main__":
    main()
