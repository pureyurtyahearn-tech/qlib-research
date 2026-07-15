"""Is the +22.7%/yr NYSE momentum result real, or an artifact? Four discriminating tests.

Hypothesis: survivorship bias interacts MULTIPLICATIVELY with momentum on small caps.
Momentum buys past winners. In a universe with zero delistings, the past winners that
later collapsed and delisted are absent BY CONSTRUCTION -- so the strategy is protected
from precisely the losses that make real momentum dangerous. Large caps rarely delist
(SP500 barely feels this); microcaps delist constantly (NYSE feels it enormously).

Tests:
  1. PLACEBO: random K=20 from the same eligible set. If random ALSO crushes the EW
     benchmark, the mechanics/benchmark are broken, not the signal.
  2. LIQUIDITY LADDER: raise the causal gate $5M -> $100M. A real risk premium should
     survive into liquid names (weaker, not vanishing). A microcap/data artifact dies.
  3. YEAR BY YEAR: is it all 2020-21 (the meme/COVID small-cap melt-up)?
  4. WHAT DID IT ACTUALLY HOLD: position-level returns; look for implausible moves that
     the +100% winsorization cap would have let through cumulatively.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from nx6_broad import simulate, ann, ir, tstat, SRC, S, E, OPEN_C, CLOSE_C, K

TRADE_GATE = 5e6


def main():
    comb = pd.read_hdf(SRC / "daily_pv.h5")
    sp = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                    .index.get_level_values("instrument").unique()))
    ny = pd.read_csv(SRC / "nyse_store_universe.csv")["ticker"].tolist()
    close = comb["$close"].unstack("instrument").sort_index()
    vol = comb["$volume"].unstack("instrument").sort_index()
    cols = [c for c in close.columns if c in set(ny)]          # NYSE-only: the suspicious one
    close, vol = close[cols], vol[cols]

    liq = (close * vol).rolling(63, min_periods=40).median().shift(1)
    sig_full = (close.shift(21) / close.shift(252) - 1).shift(1)
    ret_full = close.pct_change()
    win = (close.index >= S) & (close.index <= E)
    dates = close.index[win]; T = len(dates)
    retv = ret_full.loc[dates].values
    sigv = sig_full.loc[dates].values
    liqv = liq.loc[dates].values
    pxv = close.loc[dates].values
    base_ok = np.isfinite(pxv)

    def run(elig, sig, k=K):
        nets = np.zeros((21, T)); gros = np.zeros((21, T)); turns = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            n_, g_, at = simulate(sig, retv, elig, rb, k)
            nets[ph], gros[ph] = n_, g_; turns.append(at)
        ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
        return nets.mean(axis=0), gros.mean(axis=0), ew, np.mean(turns)

    e5 = base_ok & (liqv >= TRADE_GATE)

    # ---- 1. PLACEBO ----
    print("=== 1. PLACEBO: random K=20 vs the SAME EW benchmark (NYSE, $5M gate) ===")
    net, gross, ew, turn = run(e5, sigv)
    print(f"  momentum : netEx {ann(net-ew):+.4f}  IR {ir(net-ew):+.2f}  t {tstat(net-ew):+.2f}")
    rng = np.random.default_rng(7)
    pl = []
    for i in range(10):
        rs = rng.standard_normal(sigv.shape)
        n_, g_, ew_, _ = run(e5, rs)
        pl.append(ann(n_ - ew_))
    pl = np.array(pl)
    print(f"  placebo  : netEx mean {pl.mean():+.4f}  sd {pl.std():.4f}  "
          f"range {pl.min():+.4f}..{pl.max():+.4f}   (n=10 random signals)")
    print(f"  -> momentum sits {(ann(net-ew)-pl.mean())/pl.std():+.1f} sd above the placebo mean")

    # ---- 2. LIQUIDITY LADDER ----
    print("\n=== 2. LIQUIDITY LADDER (causal gate). A real premium weakens; an artifact dies. ===")
    print(f"{'gate':>8}{'elig/day':>10}{'EWbench':>10}{'turn':>7}{'grossEx':>10}{'netEx':>9}{'IR':>7}{'t':>7}")
    for g in [5e6, 10e6, 20e6, 50e6, 100e6]:
        e = base_ok & (liqv >= g)
        if e.sum(axis=1).mean() < 40:
            print(f"{g/1e6:>7.0f}M   too few names"); continue
        n_, g_, ew_, t_ = run(e, sigv)
        ex = n_ - ew_
        print(f"{g/1e6:>7.0f}M{e.sum(axis=1).mean():>10.0f}{ann(ew_):>+10.4f}{t_:>6.1f}x"
              f"{ann(g_-ew_):>+10.4f}{ann(ex):>+9.4f}{ir(ex):>+7.2f}{tstat(ex):>+7.2f}")

    # ---- 2b. causal at-purchase price floor (legitimate: uses only price AT the trade) ----
    print("\n=== 2b. causal at-purchase price floor (NOT the rejected full-sample filter) ===")
    print(f"{'min px':>8}{'elig/day':>10}{'netEx':>9}{'IR':>7}{'t':>7}")
    for mp in [0, 5, 10]:
        e = base_ok & (liqv >= TRADE_GATE) & (pxv >= mp)
        n_, g_, ew_, _ = run(e, sigv)
        ex = n_ - ew_
        print(f"{mp:>7}${e.sum(axis=1).mean():>10.0f}{ann(ex):>+9.4f}{ir(ex):>+7.2f}{tstat(ex):>+7.2f}")

    # ---- 3. YEAR BY YEAR ----
    print("\n=== 3. YEAR BY YEAR (NYSE $5M gate) ===")
    print(f"{'year':>6}{'strategy':>10}{'EWbench':>10}{'netEx':>10}")
    for y in sorted(set(dates.year)):
        m = np.asarray(dates.year == y)
        if m.sum() < 40:
            continue
        print(f"{y:>6}{ann(net[m]):>+10.4f}{ann(ew[m]):>+10.4f}{ann(net[m]-ew[m]):>+10.4f}")

    # ---- 4. WHAT DID IT HOLD ----
    print("\n=== 4. POSITION-LEVEL SANITY (phase 0, $5M gate) ===")
    rb = np.zeros(T, bool); rb[0::21] = True
    held_ret = []
    names = np.array(cols)
    for t in np.where(rb)[0]:
        s = np.where(e5[t] & np.isfinite(sigv[t]), sigv[t], np.nan)
        ok = ~np.isnan(s)
        if ok.sum() < K:
            continue
        idx = np.argsort(np.where(ok, s, -np.inf))[-K:]
        end = min(t + 21, T - 1)
        for i in idx:
            p0, p1 = pxv[t, i], pxv[end, i]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                held_ret.append((names[i], dates[t].date(), p1 / p0 - 1))
    hr = pd.DataFrame(held_ret, columns=["name", "date", "ret1m"])
    print(f"  {len(hr)} monthly holdings   mean 1m return {hr.ret1m.mean():+.4f}  "
          f"median {hr.ret1m.median():+.4f}")
    print(f"  worst: {hr.ret1m.min():+.3f}   best: {hr.ret1m.max():+.3f}")
    print(f"  holdings with 1m return > +100%: {(hr.ret1m > 1.0).sum()}   > +50%: {(hr.ret1m > 0.5).sum()}")
    big = hr.nlargest(8, "ret1m")
    print("  biggest single-month gains (check these look like real stocks, not artifacts):")
    for _, r in big.iterrows():
        print(f"    {r['name']:8} {r['date']}  {r.ret1m:+.2f}")
    top = hr.groupby("name").ret1m.agg(["mean", "count"]).query("count>=3").nlargest(6, "mean")
    print("  names held >=3 times, by mean 1-month return:")
    print(top.to_string())


if __name__ == "__main__":
    main()
