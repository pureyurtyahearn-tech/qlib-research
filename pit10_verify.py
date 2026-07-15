"""Tradeability proof for the PIT store -- same standard as the NYSE extension.

Data existing != tradeable. The specific things that must be TRUE, and that would be FALSE
under the old backfilled universe:
  1. the backtest actually trades (turnover > 0, orders fill)
  2. it HOLDS names that later delisted (CELG, RHT, ETFC...) -- proving the survivorship
     names are real, tradeable positions and not decoration
  3. it DROPS a name when it leaves the index / delists -- no ghost positions
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit"
SH = Path("git_ignore_folder/sharadar")


def main():
    import qlib
    qlib.init(provider_uri=str(STORE), region="us")

    inst = D.instruments(market="sp500pit")
    for y in ["2011", "2015", "2019", "2025"]:
        n = D.list_instruments(inst, start_time=f"{y}-01-01", end_time=f"{y}-12-31", as_list=True)
        print(f"  sp500pit members resolvable in {y}: {len(n)}")

    # the survivorship names must be in the universe in their era, and gone after
    print("\n=== do delisted names appear in the universe WHEN THEY WERE MEMBERS? ===")
    for t, era, gone in [("CELG", "2018-01-01", "2021-01-01"), ("RHT", "2018-01-01", "2021-01-01"),
                         ("ETFC", "2018-01-01", "2022-01-01"), ("EKDKQ", "2010-06-01", "2015-01-01")]:
        a = D.list_instruments(inst, start_time=era, end_time=era, as_list=True)
        b = D.list_instruments(inst, start_time=gone, end_time=gone, as_list=True)
        print(f"  {t:6} member on {era}: {t in a:<5}   still a member on {gone}: {t in b}")

    S, E = "2016-01-04", "2021-12-31"
    px = D.features(D.instruments("sp500pit"), ["$close"], start_time="2015-01-02", end_time=E).iloc[:, 0]
    px.index = px.index.set_names(["instrument", "datetime"])
    cl = px.unstack("instrument").sort_index()
    mom = (cl.shift(21) / cl.shift(252) - 1).loc[S:E]
    sig = mom.stack().sort_index()
    sig.index = sig.index.set_names(["datetime", "instrument"])
    print(f"\nsignal: {len(sig):,} obs, {sig.index.get_level_values('instrument').nunique()} instruments")

    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    strat = TopkDropoutStrategy(signal=sig, topk=50, n_drop=5)
    ex = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
          "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    xk = {"freq": "day", "limit_threshold": None, "deal_price": "close",
          "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 0}
    pm, ind = backtest(start_time=S, end_time=E, strategy=strat, executor=ex,
                       benchmark="AAPL", account=1e7, exchange_kwargs=xk)
    rep, pos = pm["1day"]
    net = (1 + (rep["return"] - rep["cost"])).prod() - 1
    print(f"\n=== live backtest {S} .. {E} ===")
    print(f"  days={len(rep)}  mean turnover={rep['turnover'].mean():.4f}  "
          f"days trading={(rep['turnover']>0).sum()}")
    print(f"  total NET return={net:+.4f}   cost paid={rep['cost'].sum():.4f}")
    idc = ind["1day"]; idc = idc[0] if isinstance(idc, tuple) else idc
    if hasattr(idc, "columns") and "ffr" in idc.columns:
        print(f"  fill ratio={idc['ffr'].mean():.3f}")

    # did we EVER hold a name that is now delisted?
    dead = {"CELG", "RHT", "ETFC", "AGN", "NBL", "M", "JWN", "HOG", "KSS", "HRB", "VIAB",
            "TSS", "APC1", "RTN", "STI1", "ESRX", "SCG", "GT", "MAT", "FLR", "PCG"}
    everheld, held_dead = set(), {}
    for k in sorted(pos.keys()):
        p = pos[k].position
        for nm in p:
            if nm in ("cash", "now_account_value"):
                continue
            everheld.add(nm)
            if nm in dead:
                held_dead.setdefault(nm, []).append(pd.Timestamp(k))
    print(f"\n  distinct names held over the run: {len(everheld)}")
    print(f"  >>> DELISTED (survivorship-hole) names actually HELD: {len(held_dead)}")
    for nm, ds in sorted(held_dead.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"      {nm:6} held {len(ds):>4} days   {ds[0].date()} .. {ds[-1].date()}")
    if not held_dead:
        print("      NONE -- the PIT universe is NOT working.")

    # ghost check: is any position still open after the name left the index?
    mat = pd.read_hdf(SH / "sp500_pit_membership.h5")
    ghosts = []
    for k in sorted(pos.keys()):
        d = pd.Timestamp(k)
        if d not in mat.index:
            continue
        row = mat.loc[d]
        for nm in pos[k].position:
            if nm in ("cash", "now_account_value"):
                continue
            if nm in row.index and not row[nm]:
                ghosts.append((d.date(), nm))
    print(f"\n  GHOST POSITIONS (held while NOT an index member): {len(ghosts)}"
          + (f"  e.g. {ghosts[:5]}" if ghosts else "   <- clean"))


if __name__ == "__main__":
    main()
