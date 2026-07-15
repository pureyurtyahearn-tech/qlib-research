"""Turnover-controlled long-only Top-50 variants, net of 5/15bps costs, vs RSP.
Event-driven numpy simulator (causal: holdings on day t set by signal <= t-1).
Run across all 30 loop predictions; report mean net excess vs RSP + turnover."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

S, E, K = "2023-01-01", "2026-06-16", 50
OPEN_C, CLOSE_C = 0.0005, 0.0015

def simulate(sig, ret, rebal, select):
    """sig,ret: (T,N) arrays; rebal: (T,) bool; select(sig_row, hold_w)->idx array. Returns net daily ret."""
    T, N = sig.shape
    hold = np.zeros(N); gross = np.zeros(T); cost = np.zeros(T); turn = 0.0; nreb = 0
    for t in range(T):
        r = np.nan_to_num(ret[t])
        if hold.any():
            gross[t] = hold @ r
            hold = hold * (1 + r)
            s = hold.sum();  hold = hold / s if s > 0 else hold
        if rebal[t]:
            idx = select(sig[t], hold)
            tw = np.zeros(N); tw[idx] = 1.0 / len(idx)
            dh = tw - hold
            buys = dh[dh > 0].sum(); sells = -dh[dh < 0].sum()
            cost[t] = OPEN_C * buys + CLOSE_C * sells
            turn += buys; nreb += 1
            hold = tw
    net = gross - cost
    ann_turn = turn / T * 252    # one-side annualized turnover
    return net, gross, ann_turn

def topk(sig_row, hold_w):
    s = np.where(np.isnan(sig_row), -np.inf, sig_row)
    return np.argsort(s)[-K:]

def hysteresis(band):
    def f(sig_row, hold_w):
        s = np.where(np.isnan(sig_row), -np.inf, sig_row)
        order = np.argsort(s)[::-1]           # best -> worst
        rank = np.empty_like(order); rank[order] = np.arange(len(order))
        held = np.where(hold_w > 0)[0]
        keep = [i for i in held if rank[i] < band and s[i] > -np.inf]
        if len(keep) >= K: keep = sorted(keep, key=lambda i: rank[i])[:K]
        need = K - len(keep)
        keepset = set(keep)
        add = [i for i in order if i not in keepset][:need]
        return np.array(keep + add)
    return f

def every(n):
    def mask(T): m = np.zeros(T, bool); m[::n] = True; return m
    return mask

def metrics(net, rsp):
    ex = (net - rsp)
    ex = ex[~np.isnan(ex)]
    ann = ex.mean() * 252
    ir = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    cum = np.cumprod(1 + ex); dd = (cum / np.maximum.accumulate(cum) - 1).min()
    return ann, ir, dd

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    preds = [pd.read_pickle(p.strip()) for p in open("pred_paths.txt")]
    preds = [p.iloc[:,0] if p.ndim>1 else p for p in preds]
    allinsts = sorted(set().union(*[set(p.index.get_level_values("instrument")) for p in preds]))
    # daily returns realized on day t (close_t/close_{t-1}-1) and RSP
    ret_all = D.features(allinsts, ["$close/Ref($close,1)-1"], start_time=S, end_time=E).iloc[:,0]
    ret_all.index = ret_all.index.set_names(["instrument","datetime"])
    ret_w = ret_all.unstack("instrument").sort_index()
    rsp = D.features(["RSP"], ["$close/Ref($close,1)-1"], start_time=S, end_time=E).iloc[:,0]
    rsp.index = rsp.index.get_level_values("datetime"); rsp = rsp.reindex(ret_w.index)

    variants = {
        "daily top50 (ref)":      ("daily", topk),
        "weekly (5d) top50":      ("5d",    topk),
        "biweekly (10d) top50":   ("10d",   topk),
        "monthly (21d) top50":    ("21d",   topk),
        "hysteresis band 50/75":  ("daily", hysteresis(75)),
        "hysteresis band 50/100": ("daily", hysteresis(100)),
        "hysteresis 50/100 weekly":("5d",   hysteresis(100)),
    }
    freq_map = {"daily":1, "5d":5, "10d":10, "21d":21}

    print(f"RSP ann (arith) = {rsp.mean()*252:+.4f}   (window {S}..{E}, {len(ret_w)} days)\n")
    print(f"{'variant':26}{'turn/yr':>9}{'grossEx':>9}{'netEx':>9}{'netIR':>8}{'netDD':>8}{'%>0':>6}")
    for name,(freq,sel) in variants.items():
        gross_ex=[]; net_ex=[]; net_ir=[]; net_dd=[]; turns=[]
        for p in preds:
            sw = p.unstack("instrument").reindex(index=ret_w.index, columns=ret_w.columns)
            # shift signal by 1 day: holdings on day t use signal from t-1 (causal execution)
            sig = sw.shift(1).values
            ret = ret_w.values
            rb = np.zeros(len(ret_w), bool); rb[::freq_map[freq]] = True
            net, gross, at = simulate(sig, ret, rb, sel)
            gA,_,_ = metrics(gross, rsp.values)
            nA,nIR,nDD = metrics(net, rsp.values)
            gross_ex.append(gA); net_ex.append(nA); net_ir.append(nIR); net_dd.append(nDD); turns.append(at)
        ne=np.array(net_ex)
        print(f"{name:26}{np.mean(turns):>8.0f}x{np.mean(gross_ex):>+9.3f}{ne.mean():>+9.3f}{np.mean(net_ir):>+8.2f}{np.mean(net_dd):>+8.3f}{100*(ne>0).mean():>5.0f}%")

if __name__ == "__main__":
    main()
