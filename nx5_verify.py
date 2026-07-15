"""Prove the extended store is (a) faithful to daily_pv.h5 and (b) actually TRADEABLE.

(b) is the real test. A signal containing ONLY NYSE names is fed to a real qlib backtest
(TopkDropoutStrategy + SimulatorExecutor). If those names were not tradeable the book
would sit in cash and return ~0 with zero turnover -- exactly the failure we hit before
when all.txt listed SP500 constituents as delisted in 2020.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data"
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    import qlib
    qlib.init(provider_uri=str(STORE), region="us")

    # ---------- (a) round-trip integrity ----------
    print("=== (a) store round-trip vs daily_pv.h5 ===")
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    close_dp = comb["$close"].unstack("instrument")
    tick = pd.read_csv(SRC / "nyse_store_universe.csv")["ticker"].tolist()
    probe = ["BABA", "SHOP", "UBER", "PLTR", "NIO"]
    probe = [t for t in probe if t in tick]
    got = D.features(probe, ["$close", "$volume"], start_time="2021-01-04", end_time="2023-12-29")
    got.index = got.index.set_names(["instrument", "datetime"])
    cw = got["$close"].unstack("instrument")
    for t in probe:
        a = cw[t].dropna()
        b = close_dp[t].reindex(a.index).astype(float)
        d = (a.astype(float) - b).abs().max()
        print(f"  {t:6} store obs={len(a):>5}  {a.index[0].date()}..{a.index[-1].date()}  "
              f"max|store-daily_pv| = {d:.6f}")

    inst = D.instruments(market="combo")
    names = D.list_instruments(inst, start_time="2021-01-04", end_time="2023-12-29", as_list=True)
    print(f"\n  combo universe resolves to {len(names)} instruments in 2021-2023")

    # ---------- (b) can the backtest TRADE them? ----------
    print("\n=== (b) LIVE BACKTEST on an NYSE-ONLY signal (proves tradeability) ===")
    S, E = "2020-02-03", "2024-01-05"
    nyse_only = [t for t in tick]
    px = D.features(nyse_only, ["$close"], start_time="2019-01-02", end_time=E).iloc[:, 0]
    px.index = px.index.set_names(["instrument", "datetime"])
    cl = px.unstack("instrument").sort_index()
    mom = cl.shift(21) / cl.shift(252) - 1          # 12-1 momentum, NYSE names only
    mom = mom.loc[S:E]
    sig = mom.stack().sort_index()
    sig.index = sig.index.set_names(["datetime", "instrument"])
    print(f"  signal: {len(sig):,} obs, {sig.index.get_level_values('instrument').nunique()} "
          f"instruments (ALL NYSE-only, zero SP500 names)")

    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy

    strategy = TopkDropoutStrategy(signal=sig, topk=20, n_drop=2)
    executor = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    exchange = {"freq": "day", "limit_threshold": None, "deal_price": "close",
                "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 0,
                "codes": nyse_only}
    pm, ind = backtest(start_time=S, end_time=E, strategy=strategy, executor=executor,
                       benchmark="SPY", account=1e7, exchange_kwargs=exchange)
    rep, pos = pm["1day"]
    print(f"\n  report columns: {list(rep.columns)}")
    tot_ret = (1 + (rep['return'] - rep['cost'])).prod() - 1
    print(f"  days simulated      : {len(rep)}")
    print(f"  mean daily turnover : {rep['turnover'].mean():.4f}   (0 == never traded)")
    print(f"  days with turnover>0: {(rep['turnover'] > 0).sum()}")
    print(f"  total NET return    : {tot_ret:+.4f}")
    print(f"  total cost paid     : {rep['cost'].sum():.4f}")
    idc = ind["1day"]
    idc = idc[0] if isinstance(idc, tuple) else idc
    if hasattr(idc, "columns") and "ffr" in idc.columns:
        print(f"  fill ratio (ffr)    : {idc['ffr'].mean():.3f}   (1.0 == orders fully filled)")

    # what was actually held?
    klast = sorted(pos.keys())[-1]
    last = pos[klast]
    held = [k for k in last.position.keys() if k not in ("cash", "now_account_value")]
    print(f"\n  positions on {pd.Timestamp(klast).date()}: {len(held)} names")
    print(f"    {', '.join(held[:20])}")
    sp = set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5").index.get_level_values("instrument").unique())
    print(f"    of these, in SP500: {len([h for h in held if h in sp])}   "
          f"NYSE-only: {len([h for h in held if h not in sp])}")

    ok = (rep["turnover"].mean() > 0) and (len(held) > 0)
    print(f"\n  VERDICT: NYSE names are {'TRADEABLE' if ok else 'NOT tradeable'} in the backtest.")


if __name__ == "__main__":
    main()
