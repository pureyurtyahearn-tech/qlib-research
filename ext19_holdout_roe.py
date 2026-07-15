"""ONE-SHOT HOLDOUT TEST #2 — pre-registered factor: $roe (Profitability/Quality).
Sign LOCKED from development (<=2023). Evaluated ONCE on 2024-01-01..2026-06-29.

Monotonicity check is BUILT IN from the start (the lesson from ext18/$fcfy, where a
positive long-only number masked a U-shaped, non-monotonic, sign-flipped signal). A genuine
factor must satisfy ALL of:
  (1) holdout IC has the SAME SIGN as development and clears the placebo floor (|z|>2);
  (2) quintiles are ~monotonic in the factor direction (Q5 highest-signal > Q1);
  (3) long-short (top50-bottom50) is positive;
  (4) long-only top-50 beats EW AND the opposite extreme (bottom-50) does NOT beat EW by a
      similar amount (else it's an 'extremeness' artifact, not the factor).
Report exactly as it comes out.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from ext6_momentum_full import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
FACTOR = "$roe"
DEV_END = "2023-12-31"
HOLD_START, HOLD_END = "2024-01-01", "2026-06-29"


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        g = mask[t] & np.isfinite(sig[t]) & np.isfinite(fwd[t])
        if g.sum() > 30:
            out.append(np.corrcoef(pd.Series(sig[t][g]).rank(), pd.Series(fwd[t][g]).rank())[0, 1])
    return np.array([x for x in out if np.isfinite(x)])


def frame(close, mat, fund, s, e):
    w = (close.index >= s) & (close.index <= e)
    d = close.index[w]; T = len(d); cols = close.columns
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(columns=cols, fill_value=False)
    retv = close.pct_change().loc[d].values
    elig = memb.loc[d].values & np.isfinite(close.loc[d].values)
    fwd = (close.shift(-22) / close.shift(-1) - 1).loc[d].values
    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
    fac = fund[FACTOR].unstack("instrument").reindex(index=d, columns=cols).values.astype(float)
    return d, T, cols, retv, elig, fwd, ew, fac


def phase_book(sig, retv, elig, ew, T, k=50, bottom=False):
    exs = []; turns = []
    ssig = -sig if bottom else sig
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        net, gross, turn = simulate(ssig, retv, elig, rb, k)
        exs.append(ann(net - ew)); turns.append(turn)
    return np.array(exs), np.mean(turns)


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    fund = pd.read_hdf(SH / "fundamentals_daily.h5")
    close = px["$close"].unstack("ticker").sort_index()

    # ---- lock sign on DEV ----
    _, _, _, _, elig_d, fwd_d, _, fac_d = frame(close, mat, fund, "1999-06-01", DEV_END)
    ic_dev = rank_ic(fac_d, fwd_d, elig_d).mean()
    sign = 1.0 if ic_dev >= 0 else -1.0
    print(f"PRE-REGISTERED: {FACTOR} (ROE quality). DEV(<=2023) RankIC {ic_dev:+.4f} "
          f"-> LOCKED sign {int(sign):+d} (high ROE {'LONG' if sign > 0 else 'SHORT'})\n")

    # ---- holdout ----
    d, T, cols, retv, elig, fwd, ew, fac = frame(close, mat, fund, HOLD_START, HOLD_END)
    print(f"=== HOLDOUT {d[0].date()}..{d[-1].date()} ({T/252:.1f}y)  EW mkt {ann(ew):+.4f}/yr ===\n")
    sig_raw = sign * fac
    sig = pd.DataFrame(sig_raw, index=d, columns=cols).shift(1).values

    # (1) IC
    rng = np.random.default_rng(0)
    nsd = float(np.std([rank_ic(rng.standard_normal((T, len(cols))), fwd, elig).mean() for _ in range(15)]))
    ic = rank_ic(fac, fwd, elig)          # raw IC (report sign vs dev)
    ic_signed = sign * ic.mean()
    print(f"(1) IC: raw {ic.mean():+.5f}  signed(dev) {ic_signed:+.5f}  placebo sd {nsd:.5f}  "
          f"|z| {abs(ic.mean())/nsd:.1f}  {'SAME sign as dev' if ic_signed > 0 else 'SIGN FLIPPED vs dev'}")

    # (2) quintiles: mean gross daily return by signal quintile (Q5 = highest signal)
    qr = [[] for _ in range(5)]
    for t in range(1, T):
        s = sig[t]; m = elig[t] & np.isfinite(s)
        if m.sum() < 50: continue
        r = np.where(np.isfinite(retv[t]), retv[t], 0.0)[m]
        qs = pd.qcut(pd.Series(s[m]).rank(method="first"), 5, labels=False).values
        for q in range(5): qr[q].append(r[qs == q].mean())
    qann = [np.nanmean(x) * 252 for x in qr]
    mono = qann[4] > qann[2] > qann[0]
    print(f"(2) quintiles gross ann (Q1 low-signal..Q5 high-signal): "
          + "  ".join(f"Q{i+1} {qann[i]:+.3f}" for i in range(5))
          + f"   {'MONOTONIC-ish' if mono else 'NON-monotonic'}")

    # (3) long-short top50-bottom50 (gross), (4) long-only top & bottom vs EW
    top_ex, top_turn = phase_book(sig, retv, elig, ew, T, 50, bottom=False)
    bot_ex, _ = phase_book(sig, retv, elig, ew, T, 50, bottom=True)
    ls = top_ex.mean() - bot_ex.mean()    # top-vs-EW minus bottom-vs-EW = long-short proxy
    print(f"(3) long-short (top50 minus bottom50, vs same EW): {ls:+.4f}/yr")
    print(f"(4) long-only TOP-50 vs EW: {top_ex.mean():+.4f}/yr ({100*(top_ex>0).mean():.0f}% phases>0), "
          f"turnover {top_turn:.2f}x")
    print(f"    long-only BOTTOM-50 vs EW: {bot_ex.mean():+.4f}/yr  "
          f"(if ~same as top -> EXTREMENESS artifact, not the factor)")

    # ---- verdict (all four must hold) ----
    c1 = ic_signed > 0 and abs(ic.mean()) / nsd > 2
    c2 = mono
    c3 = ls > 0
    c4 = top_ex.mean() > 0 and top_ex.mean() > bot_ex.mean() + 0.01
    print(f"\n=== VERDICT ===")
    print(f"  (1) IC same sign & significant: {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) quintiles monotonic:        {'PASS' if c2 else 'FAIL'}")
    print(f"  (3) long-short positive:        {'PASS' if c3 else 'FAIL'}")
    print(f"  (4) top>EW & not extremeness:   {'PASS' if c4 else 'FAIL'}")
    ok = c1 and c2 and c3 and c4
    print(f"  --> {FACTOR} {'CONFIRMS on the holdout' if ok else 'does NOT confirm on the holdout'}")


if __name__ == "__main__":
    main()
