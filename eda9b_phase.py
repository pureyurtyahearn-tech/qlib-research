"""Robustness: is the monthly (21d) net-positive result real or a rebalance-phase artifact?
Test monthly across 7 starting offsets, averaged over all 30 predictions."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from eda9_turnover import simulate, topk, metrics, S, E

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    preds = [pd.read_pickle(p.strip()) for p in open("pred_paths.txt")]
    preds = [p.iloc[:,0] if p.ndim>1 else p for p in preds]
    allinsts = sorted(set().union(*[set(p.index.get_level_values("instrument")) for p in preds]))
    ret_all = D.features(allinsts, ["$close/Ref($close,1)-1"], start_time=S, end_time=E).iloc[:,0]
    ret_all.index = ret_all.index.set_names(["instrument","datetime"])
    ret_w = ret_all.unstack("instrument").sort_index()
    rsp = D.features(["RSP"], ["$close/Ref($close,1)-1"], start_time=S, end_time=E).iloc[:,0]
    rsp.index = rsp.index.get_level_values("datetime"); rsp = rsp.reindex(ret_w.index)

    sigs = [p.unstack("instrument").reindex(index=ret_w.index, columns=ret_w.columns).shift(1).values for p in preds]
    ret = ret_w.values; rspv = rsp.values; T = len(ret_w)
    print(f"monthly (21d) net excess vs RSP, by rebalance phase (mean over 30 preds):")
    print(f"  {'phase':>6}{'netEx':>9}{'netIR_avg':>11}{'%loops>0':>10}")
    allnet=[]
    for ph in [0,3,6,9,12,15,18]:
        rb = np.zeros(T, bool); rb[ph::21] = True
        ne=[]; ir=[]
        for sig in sigs:
            net,_,_ = simulate(sig, ret, rb, topk)
            a,i,_ = metrics(net, rspv); ne.append(a); ir.append(i)
        ne=np.array(ne); allnet.append(ne.mean())
        print(f"  {ph:>6}{ne.mean():>+9.4f}{np.mean(ir):>+11.2f}{100*(ne>0).mean():>9.0f}%")
    allnet=np.array(allnet)
    print(f"\n  across phases: mean netEx={allnet.mean():+.4f}  std={allnet.std():.4f}  min={allnet.min():+.4f}  max={allnet.max():+.4f}")
    print(f"  phases with positive mean net excess: {int((allnet>0).sum())}/7")

if __name__ == "__main__":
    main()
