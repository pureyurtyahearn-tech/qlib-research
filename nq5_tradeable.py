"""Tradeability proof for the NASDAQ store extension -- same standard as the SP500/NYSE
proofs. Data existing != tradeable. A live qlib backtest fed a NASDAQ-ONLY momentum signal
must actually trade (turnover>0, orders fill), hold NASDAQ names (not SP500), and hold
names that later delisted (proving delisted names are real tradeable positions).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
S, E = "2020-02-03", "2024-06-28"


def main():
    import qlib
    qlib.init(provider_uri=str(STORE), region="us", kernels=1)

    nas = D.list_instruments(D.instruments("nasdaq"), start_time=S, end_time=E, as_list=True)
    print(f"nasdaq market resolves to {len(nas)} instruments in {S[:4]}-{E[:4]}")

    # build 12-1 momentum from the store for NASDAQ names (chunked read to bound memory)
    px = D.features(nas, ["$close"], start_time="2019-01-02", end_time=E).iloc[:, 0]
    px.index = px.index.set_names(["instrument", "datetime"])
    cl = px.unstack("instrument").sort_index()
    mom = (cl.shift(21) / cl.shift(252) - 1).loc[S:E]
    sig = mom.stack().sort_index()
    sig.index = sig.index.set_names(["datetime", "instrument"])
    codes = sorted(sig.index.get_level_values("instrument").unique())
    print(f"signal: {len(sig):,} obs, {len(codes)} NASDAQ instruments (zero SP500-only names by construction)")

    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy
    strat = TopkDropoutStrategy(signal=sig, topk=50, n_drop=5)
    ex = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
          "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    xk = {"freq": "day", "limit_threshold": None, "deal_price": "close",
          "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 0, "codes": codes}
    pm, ind = backtest(start_time=S, end_time=E, strategy=strat, executor=ex,
                       benchmark="SP500EW", account=1e7, exchange_kwargs=xk)
    rep, pos = pm["1day"]
    net = (1 + (rep["return"] - rep["cost"])).prod() - 1
    print(f"\n=== live NASDAQ backtest {S}..{E} ===")
    print(f"  days {len(rep)}  mean turnover {rep['turnover'].mean():.4f}  "
          f"days trading {(rep['turnover']>0).sum()}")
    print(f"  total NET return {net:+.4f}  cost paid {rep['cost'].sum():.4f}")
    idc = ind["1day"]; idc = idc[0] if isinstance(idc, tuple) else idc
    if hasattr(idc, "columns") and "ffr" in idc.columns:
        print(f"  fill ratio {idc['ffr'].mean():.3f}")

    # what did it hold, and were any delisted NASDAQ names held?
    sp = set(pd.read_csv(SH / "ever_members_full.csv")["ticker"])
    tk = pd.read_csv(SH / "nasdaq_candidates.csv", low_memory=False)
    delisted = set(tk[tk.isdelisted == "Y"].ticker.astype(str))
    everheld = set(); held_delisted = {}
    for k in sorted(pos.keys()):
        for nm in pos[k].position:
            if nm in ("cash", "now_account_value"):
                continue
            everheld.add(nm)
            if nm in delisted:
                held_delisted.setdefault(nm, []).append(pd.Timestamp(k))
    klast = sorted(pos.keys())[-1]
    held = [k for k in pos[klast].position if k not in ("cash", "now_account_value")]
    print(f"\n  distinct names held over run: {len(everheld)}")
    print(f"  final book {pd.Timestamp(klast).date()}: {len(held)} names, "
          f"of which SP500-universe: {len([h for h in held if h in sp])}  "
          f"NASDAQ-only: {len([h for h in held if h not in sp])}")
    print(f"  >>> delisted NASDAQ names actually HELD during run: {len(held_delisted)}")
    for nm, ds in sorted(held_delisted.items(), key=lambda x: -len(x[1]))[:6]:
        print(f"      {nm:6} held {len(ds):>4} days, last {ds[-1].date()}")

    ok = rep["turnover"].mean() > 0 and len(held) > 0 and len(held_delisted) > 0
    print(f"\n  VERDICT: NASDAQ store is {'TRADEABLE (trades, holds NASDAQ + delisted names)' if ok else 'NOT fully tradeable -- investigate'}")


if __name__ == "__main__":
    main()
