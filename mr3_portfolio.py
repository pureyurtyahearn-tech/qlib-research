"""MR-only vs momentum-only vs blended: long-only Top-50, net of 5/15bps, vs RSP.
Signs fitted on 2023 (in-sample), portfolio evaluated 2024-01..2026-06 (out-of-sample)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from eda9_turnover import simulate, topk

FIT_S, FIT_E = "2023-01-01", "2023-12-31"
EV_S,  EV_E  = "2024-01-01", "2026-06-16"
FDIR = Path("git_ignore_folder/_mr_factors")
MEANREV = ["composite_reversal_zscore","reversal_20d","price_zscore_20d","rsi_14d","rsi_7d",
           "williams_r_10d","stochastic_k_14d","z_ema_dev_5d",
           "volatility_normalized_reversal_1d","ts_percentile_rank_5d_return_20d",
           "price_channel_position_20d"]
MOMENTUM = ["vw_momentum_5d","vol_adj_momentum_10d","price_trend_slope_10d","obv_momentum_20d",
            "vpt_momentum_10d","PriceToHigh20","VolumeWeightedMom10"]

def ann(r): r=r[~np.isnan(r)]; return r.mean()*252
def sharpe(r): r=r[~np.isnan(r)]; return r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    sp = [l.split("\t")[0].strip() for l in open(Path.home()/".qlib"/"qlib_data"/"us_data"/"instruments"/"sp500.txt") if "2099-12-31" in l]

    ret_all = D.features(sp, ["$close/Ref($close,1)-1"], start_time=FIT_S, end_time=EV_E).iloc[:,0]
    ret_all.index = ret_all.index.set_names(["instrument","datetime"])
    ret_w = ret_all.unstack("instrument").sort_index()
    fwd = D.features(sp, ["Ref($close,-2)/Ref($close,-1)-1"], start_time=FIT_S, end_time=FIT_E).iloc[:,0]
    fwd.index = fwd.index.set_names(["instrument","datetime"])
    fwd_w = fwd.unstack("instrument").sort_index()
    rsp = D.features(["RSP"], ["$close/Ref($close,1)-1"], start_time=FIT_S, end_time=EV_E).iloc[:,0]
    rsp.index = rsp.index.get_level_values("datetime"); rsp = rsp.reindex(ret_w.index)

    def load(n):
        p=FDIR/f"{n}.h5"
        if not p.exists(): return None
        w = pd.read_hdf(p).iloc[:,0].unstack("instrument")
        return w.reindex(index=ret_w.index, columns=ret_w.columns)

    # ---- fit signs on 2023 only ----
    fitmask = (ret_w.index >= FIT_S) & (ret_w.index <= FIT_E)
    def fit_sign(w):
        ics=[]
        for t in ret_w.index[fitmask]:
            if t not in fwd_w.index: continue
            a,b = w.loc[t], fwd_w.loc[t]
            m=a.notna()&b.notna()
            if m.sum()>20: ics.append(a[m].corr(b[m], method="spearman"))
        return -1.0 if np.nanmean(ics) < 0 else 1.0

    def composite(names):
        zs=[]; signs={}
        for n in names:
            w=load(n)
            if w is None: continue
            s=fit_sign(w); signs[n]=s
            z=(w.sub(w.mean(axis=1),axis=0)).div(w.std(axis=1),axis=0)  # cross-sectional z
            zs.append(z*s)
        return pd.concat(zs).groupby(level=0).mean(), signs

    fams = {}
    fams["MEANREV"], sg_mr = composite(MEANREV)
    fams["MOMENTUM"], sg_mo = composite(MOMENTUM)
    fams["BLENDED"], _ = composite(MEANREV+MOMENTUM)
    print("signs fitted on 2023 (-1 => factor works as REVERSAL):")
    print("  MEANREV :", {k:int(v) for k,v in sg_mr.items()})
    print("  MOMENTUM:", {k:int(v) for k,v in sg_mo.items()})

    # ---- evaluate OOS 2024-2026 ----
    evm = (ret_w.index >= EV_S) & (ret_w.index <= EV_E)
    ev_idx = ret_w.index[evm]
    ret_ev = ret_w.loc[ev_idx].values
    rsp_ev = rsp.loc[ev_idx].values
    T = len(ev_idx)
    print(f"\nOOS evaluation {ev_idx[0].date()}..{ev_idx[-1].date()}  ({T} days)   RSP ann={rsp_ev[np.isfinite(rsp_ev)].mean()*252:+.4f}")
    print(f"\n{'family':10}{'rebal':>7}{'turn/yr':>9}{'grossEx':>9}{'netEx':>9}{'netIR':>8}{'netSharpe':>11}")
    for fam, sig in fams.items():
        s_ev = sig.reindex(index=ev_idx, columns=ret_w.columns).shift(1).values
        for f in [1,3,5,10,21]:
            rb = np.zeros(T, bool); rb[::f] = True
            net, gross, at = simulate(s_ev, ret_ev, rb, topk)
            gx = ann(gross - rsp_ev); nx = ann(net - rsp_ev)
            ir = (net-rsp_ev); ir = np.nanmean(ir)/np.nanstd(ir)*np.sqrt(252)
            print(f"{fam:10}{f:>6}d{at:>8.0f}x{gx:>+9.3f}{nx:>+9.3f}{ir:>+8.2f}{sharpe(net):>+11.2f}")
        print()

if __name__ == "__main__":
    main()
