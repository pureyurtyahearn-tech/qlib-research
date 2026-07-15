"""Did the fix work? Re-run the exact verification that found 16,232 ghost position-days,
now with PITTopkDropoutStrategy. Ghosts must be ZERO, and delisted names must still be
genuinely held and then exited on removal (not simply avoided).

Runs BOTH strategies over the same window so the difference is attributable.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from qlib.contrib.strategy import TopkDropoutStrategy
from pit_strategy import PITTopkDropoutStrategy

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit"
SH = Path("git_ignore_folder/sharadar")
S, E = "2016-01-04", "2021-12-31"


def ghosts(pos, mat):
    g = []
    for k in sorted(pos.keys()):
        d = pd.Timestamp(k)
        i = mat.index.searchsorted(d, side="right") - 1
        if i < 0:
            continue
        row = mat.iloc[i]
        for nm in pos[k].position:
            if nm in ("cash", "now_account_value"):
                continue
            if nm in row.index and not row[nm]:
                g.append((d.date(), nm))
    return g


def main():
    import qlib
    # kernels=1: force qlib's data loader to run SERIALLY in-process. The default joblib
    # parallel loader spawns worker processes (Windows spawn) that each duplicate the panel
    # in memory -- with ~6GB free that OOMs even for a few hundred names. Serial loading is
    # a little slower but has a flat, bounded memory footprint.
    qlib.init(provider_uri=str(STORE), region="us", kernels=1)
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")

    # MEMORY FIX: the old code called D.features on the ENTIRE sp500pit universe in one
    # shot -- qlib's parallel loader materialised a ~3M-row series and unstacked it to wide,
    # which OOM'd the machine (and crashed VS Code). The qlib store was built FROM
    # sep_panel.h5 (identical closeadj prices), so we compute the 12-1 momentum signal
    # straight from that on-disk panel: one already-materialised frame, no parallel workers,
    # no giant unstack. Restrict to the eval window up front so the wide frame stays small.
    panel = pd.read_hdf(SH / "sep_panel.h5")           # MultiIndex (date, ticker)
    cl = panel["$close"].unstack("ticker").sort_index()
    cl = cl.loc["2015-01-02":E]                        # 252d lookback before S=2016-01-04
    mom = (cl.shift(21) / cl.shift(252) - 1).loc[S:E]
    del panel
    sig = mom.stack().sort_index()
    sig.index = sig.index.set_names(["datetime", "instrument"])
    codes = sorted(sig.index.get_level_values("instrument").unique())
    print(f"signal: {len(sig):,} obs, {len(codes)} instruments (built from sep_panel.h5, no D.features)")

    from qlib.backtest import backtest
    ex = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
          "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    # pass explicit codes so the exchange loads only the ~600 names in the signal,
    # not the full 822-name market universe
    xk = {"freq": "day", "limit_threshold": None, "deal_price": "close",
          "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 0, "codes": codes}

    runs = {
        "BROKEN (stock qlib)": TopkDropoutStrategy(signal=sig, topk=50, n_drop=5),
        "FIXED (PIT strategy)": PITTopkDropoutStrategy(signal=sig, topk=50, n_drop=5,
                                                       membership=mat),
    }
    dead = {"CELG", "RHT", "ETFC", "AGN", "NBL", "M", "JWN", "HOG", "KSS", "HRB", "VIAB",
            "TSS", "APC1", "RTN", "STI1", "ESRX", "SCG", "GT", "MAT", "FLR", "PCG", "ATVI",
            "AET", "ABMD", "BCR", "ANDV", "ALTR1", "BHI", "BMC"}

    out = {}
    for name, strat in runs.items():
        pm, ind = backtest(start_time=S, end_time=E, strategy=strat, executor=ex,
                           benchmark="AAPL", account=1e7, exchange_kwargs=xk)
        rep, pos = pm["1day"]
        net = (1 + (rep["return"] - rep["cost"])).prod() - 1
        g = ghosts(pos, mat)
        heldset, held_dead = set(), {}
        for k in sorted(pos.keys()):
            for nm in pos[k].position:
                if nm in ("cash", "now_account_value"):
                    continue
                heldset.add(nm)
                if nm in dead:
                    held_dead.setdefault(nm, []).append(pd.Timestamp(k))
        out[name] = dict(net=net, ghosts=len(g), held=len(heldset), dead=held_dead,
                         turn=rep["turnover"].mean(), strat=strat, g=g)
        print(f"\n=== {name} ===")
        print(f"  net return        : {net:+.4f}")
        print(f"  mean turnover     : {out[name]['turn']:.4f}")
        print(f"  distinct names held: {len(heldset)}")
        print(f"  delisted names held: {len(held_dead)}  {sorted(held_dead)[:8]}")
        print(f"  >>> GHOST POSITION-DAYS: {len(g)}")
        if g:
            worst = pd.Series([x[1] for x in g]).value_counts().head(5)
            print(f"      worst offenders: {dict(worst)}")

    b, f = out["BROKEN (stock qlib)"], out["FIXED (PIT strategy)"]
    print("\n" + "=" * 62)
    print(f"  ghost position-days : {b['ghosts']:>7}  ->  {f['ghosts']:>7}")
    print(f"  reported net return : {b['net']:>+7.4f}  ->  {f['net']:>+7.4f}"
          f"   (fabricated: {b['net']-f['net']:+.4f})")
    print("=" * 62)

    fs = getattr(f["strat"], "forced_sales", [])
    sold = [x for x in fs if x[2] == "SOLD"]
    stuck = [x for x in fs if x[2] == "UNSELLABLE"]
    print(f"\n  forced exits executed: {len(sold)}   UNSELLABLE (should be 0): {len(stuck)}")
    print(f"  sample forced exits (name left the index -> position closed):")
    for d, c, _ in sold[:8]:
        print(f"    {d.date()}  {c}")
    if stuck:
        print(f"  !! still unsellable: {stuck[:6]}")
    if f["ghosts"] == 0 and not stuck:
        print("\n  VERDICT: point-in-time membership is now ENFORCED. No ghosts, no stuck positions.")
    else:
        print("\n  VERDICT: NOT fully fixed -- see above.")


if __name__ == "__main__":
    main()
