"""Long-short (top50-bottom50) with realistic transaction costs.
Turnover computed from actual daily position weights (with intraday drift),
netted across a one-way cost ladder + short-borrow."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

REAL_PRED = r"git_ignore_folder/RD-Agent_workspace/12b0eb7178a0495a980618bf8bac9a54/mlruns/657215356007540889/871f0a480c7547f5bd6dd434c863d6c7/artifacts/pred.pkl"
S, E, K = "2023-01-01", "2026-06-16", 50
BORROW_ANNUAL = 0.0035   # ~35 bps/yr easy-to-borrow SP500 short fee

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    pred = pd.read_pickle(REAL_PRED); pred = pred.iloc[:,0] if pred.ndim>1 else pred
    insts = list(pred.index.get_level_values("instrument").unique())
    fwd = D.features(insts, ["Ref($close,-2)/Ref($close,-1)-1"], start_time=S, end_time=E).iloc[:,0]
    fwd.index = fwd.index.set_names(["instrument","datetime"])
    fwd = fwd.reorder_levels(["datetime","instrument"]).sort_index()

    sig = pred.unstack("instrument").sort_index()
    fw  = fwd.unstack("instrument").reindex(sig.index)

    # build weight matrix: +1/K on top-K, -1/K on bottom-K by signal each day
    W = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    for dt in sig.index:
        row = sig.loc[dt].dropna()
        if len(row) < 2*K: continue
        order = row.sort_values()
        W.loc[dt, order.index[:K]]  = -1.0/K   # short lowest
        W.loc[dt, order.index[-K:]] = +1.0/K   # long highest

    gross = (W * fw).sum(axis=1)                       # gross daily LS return
    # turnover with intraday drift: compare today's target to yesterday's drifted book
    Wdrift = (W.shift(1) * (1 + fw.shift(1))).fillna(0.0)
    turnover = (W - Wdrift).abs().sum(axis=1)          # gross notional traded (units of gross book)
    turnover.iloc[0] = W.iloc[0].abs().sum()           # initial ramp

    def stats(r):
        r = r.dropna()
        return ann(r), r.mean()/r.std()*np.sqrt(252), (( (1+r).cumprod()/(1+r).cumprod().cummax())-1).min()
    def ann(r):
        r=r.dropna(); return (1+r).prod()**(252/len(r))-1

    print(f"representative pred (IC~0.005), K={K} per side, {len(sig)} days")
    print(f"avg 2-sided daily turnover = {turnover.mean():.3f} of gross book  "
          f"(~{turnover.mean()*252:.1f}x annualized)")
    gA,gS,gD = stats(gross)
    print(f"\n  GROSS               annRet={gA:+.4f}  Sharpe={gS:+.2f}  maxDD={gD:+.4f}")
    print(f"\n  {'one-way cost':>14} {'annRet':>9} {'Sharpe':>8} {'maxDD':>9}")
    for bps in [5,10,20]:
        c = bps/1e4
        net = gross - c*turnover - BORROW_ANNUAL/252   # borrow on gross short notional (=1)
        nA,nS,nD = stats(net)
        print(f"  {bps:>10} bps  {nA:>+9.4f} {nS:>+8.2f} {nD:>+9.4f}")
    # breakeven cost where Sharpe->0
    lo,hi=0,200
    for _ in range(40):
        mid=(lo+hi)/2; net=gross-(mid/1e4)*turnover-BORROW_ANNUAL/252
        s=net.dropna().mean()
        if s>0: lo=mid
        else: hi=mid
    print(f"\n  breakeven one-way cost (net mean return -> 0): ~{lo:.1f} bps")

if __name__ == "__main__":
    main()
