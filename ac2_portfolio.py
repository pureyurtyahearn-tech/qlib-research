"""12-1 momentum portfolio: monthly rebalance, long-only Top-50 + long-short, net of
5/15bps, vs RSP *and* vs equal-weight-own-universe (the survivorship-matched benchmark).

Survivorship note: the 447 names are current SP500 constituents backfilled to 2010, so
beating RSP is NOT sufficient -- a survivor universe beats a real index for free. The
EW-own-universe benchmark neutralizes that: it asks whether momentum picks better than
average FROM THE SAME NAMES. That is the honest alpha test.

Phase robustness is mandatory: a prior monthly result looked positive and turned out to
be a rebalance-timing artifact, so every one of the 21 monthly phases is evaluated.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

SRC = Path("git_ignore_folder/factor_implementation_source_data")
FDIR = Path("git_ignore_folder/_ac_factors")
OPEN_C, CLOSE_C = 0.0005, 0.0015   # 5bps buy / 15bps sell -- identical to prior tests
K = 50


def simulate(sig, ret, rebal, k=K, short=False):
    """Long-only top-k (or dollar-neutral top-k/bottom-k). Causal: sig already shifted.
    Returns (net, gross, cost, ann_turnover_one_side)."""
    T, N = sig.shape
    hold = np.zeros(N)
    gross = np.zeros(T); cost = np.zeros(T); turn = 0.0
    for t in range(T):
        r = np.nan_to_num(ret[t])
        if hold.any():
            gross[t] = hold @ r
            hold = hold * (1 + r)
            g = np.abs(hold).sum()
            if g > 0:
                hold = hold / g if not short else hold / (g / 2.0)  # keep leverage stable
        if rebal[t]:
            s = sig[t].copy()
            ok = ~np.isnan(s)
            if ok.sum() < 2 * k:
                continue
            order = np.argsort(np.where(ok, s, -np.inf))
            tw = np.zeros(N)
            tw[order[-k:]] = 1.0 / k
            if short:
                lo = np.argsort(np.where(ok, s, np.inf))[:k]
                tw[lo] = -1.0 / k
            dh = tw - hold
            buys = dh[dh > 0].sum(); sells = -dh[dh < 0].sum()
            cost[t] = OPEN_C * buys + CLOSE_C * sells
            turn += buys
            hold = tw
    return gross - cost, gross, cost, turn / T * 252


def ann(x):
    x = x[np.isfinite(x)]
    return x.mean() * 252


def ir(x):
    x = x[np.isfinite(x)]
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


def maxdd(x):
    x = np.nan_to_num(x)
    c = np.cumprod(1 + x)
    return float((c / np.maximum.accumulate(c) - 1).min())


def main():
    import qlib
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "us_data"), region="us")

    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sig12 = pd.read_hdf(FDIR / "mom_12_1.h5")
    sig5 = pd.read_hdf(FDIR / "mom_5d.h5")
    close = comb["$close"].unstack("instrument").sort_index()
    close = close[sig12.columns].loc[sig12.index]
    ret_w = close.pct_change()

    # benchmarks
    rsp = D.features(["RSP"], ["$close/Ref($close,1)-1"], start_time=str(ret_w.index.min().date()),
                     end_time=str(ret_w.index.max().date())).iloc[:, 0]
    rsp.index = rsp.index.get_level_values("datetime")
    rsp = rsp.reindex(ret_w.index)
    ew = ret_w.mean(axis=1)          # equal-weight of OUR universe (survivorship-matched)

    dates = ret_w.index
    T = len(dates)
    retv = ret_w.values
    rspv = rsp.values
    ewv = ew.values

    print(f"window {dates.min().date()} .. {dates.max().date()}   {T} days, {ret_w.shape[1]} names")
    print(f"  RSP                    ann = {ann(rspv):+.4f}")
    print(f"  EW own universe        ann = {ann(ewv):+.4f}   <- survivorship-matched benchmark")
    print(f"  survivorship premium (EW - RSP) = {ann(ewv) - ann(rspv):+.4f}/yr\n")

    sigs = {"mom_12_1": sig12.shift(1).values, "mom_5d": sig5.reindex(index=dates, columns=ret_w.columns).shift(1).values}

    # ---------- monthly, ALL 21 phases ----------
    for nm, sv in sigs.items():
        print(f"=== {nm}: long-only Top-50, monthly (21d) rebalance, ALL 21 phases ===")
        rows = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            net, gross, cost, at = simulate(sv, retv, rb)
            rows.append(dict(ph=ph, turn=at, gross_rsp=ann(gross - rspv), net_rsp=ann(net - rspv),
                             gross_ew=ann(gross - ewv), net_ew=ann(net - ewv),
                             ir_ew=ir(net - ewv), net_abs=ann(net), sh=ir(net), dd=maxdd(net)))
        d = pd.DataFrame(rows)
        print(f"  {'':4}{'turn/yr':>9}{'grossEx_EW':>12}{'netEx_EW':>10}{'netIR_EW':>10}{'netEx_RSP':>11}")
        print(f"  {'mean':4}{d.turn.mean():>8.2f}x{d.gross_ew.mean():>+12.4f}{d.net_ew.mean():>+10.4f}"
              f"{d.ir_ew.mean():>+10.2f}{d.net_rsp.mean():>+11.4f}")
        print(f"  {'std':4}{d.turn.std():>9.2f}{d.gross_ew.std():>12.4f}{d.net_ew.std():>10.4f}"
              f"{d.ir_ew.std():>10.2f}{d.net_rsp.std():>11.4f}")
        print(f"  {'min':4}{d.turn.min():>9.2f}{d.gross_ew.min():>+12.4f}{d.net_ew.min():>+10.4f}"
              f"{d.ir_ew.min():>+10.2f}{d.net_rsp.min():>+11.4f}")
        print(f"  {'max':4}{d.turn.max():>9.2f}{d.gross_ew.max():>+12.4f}{d.net_ew.max():>+10.4f}"
              f"{d.ir_ew.max():>+10.2f}{d.net_rsp.max():>+11.4f}")
        pos = int((d.net_ew > 0).sum())
        print(f"  phases with POSITIVE net excess vs EW: {pos}/21     vs RSP: {int((d.net_rsp>0).sum())}/21")
        be = d.gross_ew.mean() / d.turn.mean() * 1e4 if d.turn.mean() > 0 else 0
        print(f"  breakeven cost (vs EW): {be:.1f} bps per unit turnover   (actual model ~{(OPEN_C+CLOSE_C)*1e4:.0f} bps round trip)\n")

    # ---------- rebalance frequency sweep (mom_12_1) ----------
    print("=== mom_12_1: rebalance frequency sweep (long-only Top-50, vs EW own universe) ===")
    print(f"{'rebal':>8}{'turn/yr':>9}{'grossEx_EW':>12}{'netEx_EW':>10}{'netIR_EW':>10}{'netSharpe':>11}{'maxDD':>9}")
    for f in [1, 5, 10, 21, 63]:
        g_ = []; n_ = []; i_ = []; t_ = []; s_ = []; dd_ = []
        for ph in range(min(f, 21)):
            rb = np.zeros(T, bool); rb[ph::f] = True
            net, gross, cost, at = simulate(sigs["mom_12_1"], retv, rb)
            g_.append(ann(gross - ewv)); n_.append(ann(net - ewv)); i_.append(ir(net - ewv))
            t_.append(at); s_.append(ir(net)); dd_.append(maxdd(net))
        print(f"{f:>7}d{np.mean(t_):>8.2f}x{np.mean(g_):>+12.4f}{np.mean(n_):>+10.4f}"
              f"{np.mean(i_):>+10.2f}{np.mean(s_):>+11.2f}{np.mean(dd_):>+9.3f}")

    # ---------- long-short (academic form) ----------
    print("\n=== mom_12_1: LONG-SHORT top50 - bottom50, monthly, mean over 21 phases ===")
    ls = []
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        net, gross, cost, at = simulate(sigs["mom_12_1"], retv, rb, short=True)
        ls.append(dict(turn=at, gross=ann(gross), net=ann(net), sh=ir(net), dd=maxdd(net)))
    d = pd.DataFrame(ls)
    print(f"  turnover {d.turn.mean():.2f}x/yr   gross {d.gross.mean():+.4f}   net {d.net.mean():+.4f}"
          f"   netSharpe {d.sh.mean():+.2f}   maxDD {d.dd.mean():+.3f}")
    print(f"  phases with positive net: {int((d.net>0).sum())}/21")

    # ---------- year by year (mom_12_1, monthly, phase 0) ----------
    print("\n=== mom_12_1 long-only Top-50 monthly: YEAR BY YEAR (mean over 21 phases) ===")
    print(f"{'year':>6}{'net':>9}{'EW':>9}{'netEx_EW':>10}{'RSP':>9}{'netEx_RSP':>11}")
    allnet = np.zeros((21, T))
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        net, _, _, _ = simulate(sigs["mom_12_1"], retv, rb)
        allnet[ph] = net
    netm = allnet.mean(axis=0)
    yrs = pd.Series(netm, index=dates)
    for y in sorted(set(dates.year)):
        m = dates.year == y
        if m.sum() < 60:
            continue
        print(f"{y:>6}{ann(netm[m]):>+9.4f}{ann(ewv[m]):>+9.4f}{ann(netm[m]-ewv[m]):>+10.4f}"
              f"{ann(rspv[m]):>+9.4f}{ann(netm[m]-rspv[m]):>+11.4f}")
    wins = sum(1 for y in sorted(set(dates.year)) if (dates.year == y).sum() >= 60
               and ann(netm[dates.year == y] - ewv[dates.year == y]) > 0)
    tot = sum(1 for y in sorted(set(dates.year)) if (dates.year == y).sum() >= 60)
    print(f"\n  years beating EW own universe (net): {wins}/{tot}")


if __name__ == "__main__":
    main()
