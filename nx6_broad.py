"""K=20 12-1 momentum on the BROAD universe vs SP500-only, same window, same everything.

Design notes (each one is a lesson already paid for):
  * WINDOW HELD CONSTANT. NYSE data ends 2024-01-08, so all three universes are run on
    2020-01..2024-01. Comparing a 4-yr broad result to the 15.5-yr SP500 result would
    confound universe with period.
  * LIQUIDITY SCREEN IS CAUSAL: trailing 63-day median dollar volume >= $5M, known as of
    t-1. A full-sample liquidity filter would be a look-ahead, and we are here precisely
    because of a look-ahead.
  * BENCHMARK = equal-weight of the SAME causally-screened universe, not RSP. RSP would
    hand back ~4%/yr of fake alpha on a survivor universe (today's lesson).
  * SURVIVORSHIP CAVEAT: the NYSE set is ALSO survivorship-biased (0 delistings >1yr
    before its end). The EW-own-universe benchmark makes the comparison internally
    consistent, but neither universe is clean in absolute terms.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SRC = Path("git_ignore_folder/factor_implementation_source_data")
S, E = "2020-01-02", "2024-01-08"
TRADE_GATE = 5e6
OPEN_C, CLOSE_C = 0.0005, 0.0015
K = 20


def simulate(sig, ret, elig, rebal, k):
    """Long-only top-k among ELIGIBLE names. sig/ret/elig are (T,N); sig already causal."""
    T, N = sig.shape
    hold = np.zeros(N); gross = np.zeros(T); cost = np.zeros(T); turn = 0.0
    for t in range(T):
        r = np.nan_to_num(ret[t])
        if hold.any():
            gross[t] = hold @ r
            hold = hold * (1 + r)
            s = hold.sum()
            if s > 0:
                hold /= s
        if rebal[t]:
            s = np.where(elig[t] & np.isfinite(sig[t]), sig[t], np.nan)
            ok = ~np.isnan(s)
            if ok.sum() < k:
                continue
            idx = np.argsort(np.where(ok, s, -np.inf))[-k:]
            tw = np.zeros(N); tw[idx] = 1.0 / k
            dh = tw - hold
            buys = dh[dh > 0].sum(); sells = -dh[dh < 0].sum()
            cost[t] = OPEN_C * buys + CLOSE_C * sells
            turn += buys
            hold = tw
    return gross - cost, gross, turn / T * 252


def ann(x):
    x = x[np.isfinite(x)]; return x.mean() * 252


def ir(x):
    x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


def tstat(x):
    x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else 0.0


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        a, b, m = sig[t], fwd[t], mask[t]
        g = m & np.isfinite(a) & np.isfinite(b)
        if g.sum() > 30:
            ra = pd.Series(a[g]).rank(); rb = pd.Series(b[g]).rank()
            out.append(np.corrcoef(ra, rb)[0, 1])
    out = np.array(out); out = out[np.isfinite(out)]
    return out.mean(), out


def main():
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    ny = pd.read_csv(SRC / "nyse_store_universe.csv")["ticker"].tolist()

    close = comb["$close"].unstack("instrument").sort_index()
    vol = comb["$volume"].unstack("instrument").sort_index()
    cols = [c for c in close.columns if c in set(sp) | set(ny)]
    close, vol = close[cols], vol[cols]

    # causal liquidity: trailing 63d median $vol, known as of t-1
    liq = (close * vol).rolling(63, min_periods=40).median().shift(1)
    sig_full = (close.shift(21) / close.shift(252) - 1).shift(1)     # causal 12-1 momentum
    ret_full = close.pct_change()
    fwd21 = close.shift(-22) / close.shift(-1) - 1                   # forward 21d, causal

    win = (close.index >= S) & (close.index <= E)
    dates = close.index[win]
    T = len(dates)
    retv = ret_full.loc[dates].values
    sigv = sig_full.loc[dates].values
    fwdv = fwd21.loc[dates].values
    liqv = liq.loc[dates].values
    tradable = np.isfinite(close.loc[dates].values) & (liqv >= TRADE_GATE)

    spset = set(sp)
    is_sp = np.array([c in spset for c in cols])
    universes = {
        "SP500 only":  is_sp,
        "NYSE only":   ~is_sp,
        "COMBINED":    np.ones(len(cols), bool),
    }

    print(f"window {dates[0].date()} .. {dates[-1].date()}  ({T} days)")
    print(f"causal liquidity screen: trailing 63d median $vol >= ${TRADE_GATE/1e6:.0f}M\n")
    print(f"{'universe':12}{'names avail':>13}{'avg eligible/day':>18}")
    eligs = {}
    for nm, colmask in universes.items():
        e = tradable & colmask[None, :]
        eligs[nm] = e
        print(f"{nm:12}{int(colmask.sum()):>13}{e.sum(axis=1).mean():>18.0f}")

    # ---------- IC vs noise floor, per universe ----------
    print(f"\n=== IC (RankIC vs forward 21d), within each universe's eligible set ===")
    print(f"{'universe':12}{'RankIC':>10}{'IC t-stat':>11}{'noise sd':>11}{'|z| vs floor':>14}")
    rng = np.random.default_rng(0)
    for nm, e in eligs.items():
        mu, series = rank_ic(sigv, fwdv, e)
        null = []
        for _ in range(15):
            r = rng.standard_normal(sigv.shape)
            null.append(rank_ic(r, fwdv, e)[0])
        nsd = float(np.std(null))
        print(f"{nm:12}{mu:>+10.4f}{tstat(series):>+11.2f}{nsd:>11.5f}{abs(mu)/nsd:>14.1f}")

    # ---------- portfolio: K=20, monthly, all 21 phases ----------
    print(f"\n=== K={K} 12-1 momentum, monthly rebalance, ALL 21 phases, net 5/15bps ===")
    print("    benchmark = equal-weight of the SAME causally-screened universe")
    print(f"\n{'universe':12}{'EWbench':>9}{'turn/yr':>9}{'grossEx':>9}{'netEx':>9}{'netIR':>8}{'t-stat':>8}{'phases>0':>10}")
    res = {}
    for nm, e in eligs.items():
        ew = np.array([np.nanmean(np.where(e[t], retv[t], np.nan)) for t in range(T)])
        nets = np.zeros((21, T)); gros = np.zeros((21, T)); turns = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            n_, g_, at = simulate(sigv, retv, e, rb, K)
            nets[ph], gros[ph] = n_, g_
            turns.append(at)
        net = nets.mean(axis=0); gross = gros.mean(axis=0)
        ex = net - ew
        ph_pos = int(sum((ann(nets[p] - ew) > 0) for p in range(21)))
        res[nm] = dict(ew=ann(ew), turn=np.mean(turns), gx=ann(gross - ew), nx=ann(ex),
                       ir=ir(ex), t=tstat(ex), pp=ph_pos)
        r = res[nm]
        print(f"{nm:12}{r['ew']:>+9.4f}{r['turn']:>8.2f}x{r['gx']:>+9.4f}{r['nx']:>+9.4f}"
              f"{r['ir']:>+8.2f}{r['t']:>+8.2f}{ph_pos:>7}/21")

    print(f"\n=== comparison to the SP500-only 15.5-yr result (ac3/ac4) ===")
    print(f"  SP500 K=20, 2011-2026 (15.5y):  netEx vs EW +0.0742  IR +0.46  t +1.83")
    r = res["SP500 only"]
    print(f"  SP500 K=20, 2020-2024 (4.0y) :  netEx vs EW {r['nx']:+.4f}  IR {r['ir']:+.2f}  t {r['t']:+.2f}   <- same universe, short window")
    r = res["COMBINED"]
    print(f"  COMBINED  K=20, 2020-2024    :  netEx vs EW {r['nx']:+.4f}  IR {r['ir']:+.2f}  t {r['t']:+.2f}")
    print(f"\n  (the SP500 2020-2024 row is the correct control: it isolates the effect of")
    print(f"   BREADTH from the effect of the shorter, momentum-hostile 2020-2024 period.)")


if __name__ == "__main__":
    main()
