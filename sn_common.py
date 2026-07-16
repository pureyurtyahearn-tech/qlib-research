"""Shared infrastructure for the sector-neutralization study (sn1_dev / sn2_holdout).

Builds RAW and SECTOR-NEUTRAL versions of three factors on the PIT S&P 500 universe:
  $fcfy (FCF yield, value), $roe (quality), and 12-1 price momentum (technical).

Neutralization (pre-registered): within-(date, sector) Z-SCORE of the raw factor, over
eligible index members only, using Sharadar TICKERS.sector (11 groups). This removes both
the sector-mean tilt and the sector-dispersion artifact -- the hypothesis being that raw
FCF/ROE were really a mega-cap-tech sector-concentration bet in the 2024-26 regime.

PIT caveat on sector: TICKERS.sector is the CURRENT classification (one static value per
ticker), not point-in-time. GICS-style reclassifications are rare (a few names/decade), a
far smaller concern than price/fundamentals PIT. No PIT sector history exists in Sharadar.

Reuses ext6's PIT-enforcing simulator (force-exit on index removal).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from ext6_momentum_full import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")


def load_base():
    """close (wide), membership matrix, fundamentals (daily), sector map (ticker->sector)."""
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    fund = pd.read_hdf(SH / "fundamentals_daily.h5")
    close = px["$close"].unstack("ticker").sort_index()
    tk = pd.read_csv(SH / "tickers_sep_all.csv", low_memory=False)
    sec = (tk[tk.table == "SEP"][["ticker", "sector"]].drop_duplicates("ticker")
           .set_index("ticker")["sector"])
    sector = sec.reindex(close.columns)          # aligned to close cols; NaN -> "Unknown"
    sector = sector.fillna("Unknown")
    return close, mat, fund, sector


def build_raw_factors(close, fund):
    """Return dict name -> wide DataFrame (T x N) of the RAW factor, aligned to close.
    Momentum is causal 12-1 (shift(21)/shift(252)-1); fundamentals are already daily/causal."""
    cols = close.columns
    mom = (close.shift(21) / close.shift(252) - 1)           # 12-1 momentum
    out = {"mom": mom}
    for f in ["$fcfy", "$roe"]:
        w = fund[f].unstack("instrument").reindex(index=close.index, columns=cols)
        out[f.strip("$")] = w.astype(float)
    return out


def sector_zscore(fac_wide, sector, elig_wide):
    """Within-(date, sector) z-score over eligible names. Non-eligible/NaN stay NaN."""
    z = pd.DataFrame(np.nan, index=fac_wide.index, columns=fac_wide.columns)
    masked = fac_wide.where(elig_wide)
    for s, g in sector.groupby(sector):
        cols = [c for c in g.index if c in masked.columns]
        if not cols:
            continue
        sub = masked[cols]
        mu = sub.mean(axis=1)
        sd = sub.std(axis=1)
        z[cols] = sub.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z


def make_windows(close, mat):
    """Precompute membership ffill + returns/fwd on the full index (slice per window later)."""
    memb = mat.reindex(index=close.index, method="ffill").fillna(False)
    memb = memb.reindex(columns=close.columns, fill_value=False)
    retv_full = close.pct_change()
    fwd_full = (close.shift(-22) / close.shift(-1) - 1)       # 21d fwd, causal (skip today)
    return memb, retv_full, fwd_full


def slice_window(close, memb, retv_full, fwd_full, s, e):
    w = (close.index >= s) & (close.index <= e)
    d = close.index[w]
    elig = (memb.loc[d].values & np.isfinite(close.loc[d].values))
    retv = retv_full.loc[d].values
    fwd = fwd_full.loc[d].values
    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(len(d))])
    return d, elig, retv, fwd, ew


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        g = mask[t] & np.isfinite(sig[t]) & np.isfinite(fwd[t])
        if g.sum() > 30:
            out.append(np.corrcoef(pd.Series(sig[t][g]).rank(),
                                   pd.Series(fwd[t][g]).rank())[0, 1])
    return np.array([x for x in out if np.isfinite(x)])


def placebo_sd(fwd, elig, T, N, n=15):
    rng = np.random.default_rng(0)
    return float(np.std([rank_ic(rng.standard_normal((T, N)), fwd, elig).mean()
                         for _ in range(n)]))


def quintiles(sig, retv, elig, T):
    """Mean gross ann return by signal quintile (Q5 = highest signal). sig already shifted."""
    qr = [[] for _ in range(5)]
    for t in range(1, T):
        s = sig[t]; m = elig[t] & np.isfinite(s)
        if m.sum() < 50:
            continue
        r = np.where(np.isfinite(retv[t]), retv[t], 0.0)[m]
        qs = pd.qcut(pd.Series(s[m]).rank(method="first"), 5, labels=False).values
        for q in range(5):
            qr[q].append(r[qs == q].mean())
    return [np.nanmean(x) * 252 for x in qr]


def phased_book(sig, retv, elig, ew, T, k=50, bottom=False):
    """21-phase monthly-rebalance net book vs EW. Returns (ann net excess, tstat, turnover,
    frac phases>0). sig already sign-applied and shifted."""
    ssig = -sig if bottom else sig
    nets = np.zeros((21, T)); turns = []
    for ph in range(21):
        rb = np.zeros(T, bool); rb[ph::21] = True
        net, gross, turn = simulate(ssig, retv, elig, rb, k)
        nets[ph] = net; turns.append(turn)
    net_mean = nets.mean(axis=0); ex = net_mean - ew
    pp = float(np.mean([ann(nets[p] - ew) > 0 for p in range(21)]))
    return ann(ex), tstat(ex), float(np.mean(turns)), pp


def evaluate(fac_wide, d, elig, retv, fwd, ew, sign, k=50, label=""):
    """Full metric bundle for one factor variant on one window. sign locked externally."""
    T = len(d); N = fac_wide.shape[1]
    fac = fac_wide.loc[d].values
    ic_arr = rank_ic(fac, fwd, elig)
    ic = ic_arr.mean()
    nsd = placebo_sd(fwd, elig, T, N)
    sig = pd.DataFrame(sign * fac, index=d, columns=fac_wide.columns).shift(1).values
    qann = quintiles(sig, retv, elig, T)
    mono = qann[4] > qann[2] > qann[0]
    top_ex, top_t, top_turn, top_pp = phased_book(sig, retv, elig, ew, T, k, bottom=False)
    bot_ex, _, _, _ = phased_book(sig, retv, elig, ew, T, k, bottom=True)
    return {
        "label": label, "ic": ic, "ic_signed": sign * ic, "nsd": nsd,
        "z": abs(ic) / nsd if nsd > 0 else 0.0,
        "q": qann, "mono": mono,
        "top_ex": top_ex, "top_t": top_t, "turn": top_turn, "top_pp": top_pp,
        "bot_ex": bot_ex, "ls": top_ex - bot_ex,
    }


def print_eval(m):
    print(f"  [{m['label']}]  RankIC {m['ic']:+.5f} (signed {m['ic_signed']:+.5f}, "
          f"|z| {m['z']:.1f}, placebo sd {m['nsd']:.5f})")
    print(f"    quintiles gross ann Q1..Q5: " + "  ".join(f"{q:+.3f}" for q in m["q"])
          + f"   {'MONO' if m['mono'] else 'non-mono'}")
    print(f"    net top-{50} vs EW {m['top_ex']:+.4f}/yr (t {m['top_t']:+.2f}, "
          f"{100*m['top_pp']:.0f}% phases>0, turn {m['turn']:.2f}x)")
    print(f"    net bottom-50 vs EW {m['bot_ex']:+.4f}/yr    long-short {m['ls']:+.4f}/yr")
