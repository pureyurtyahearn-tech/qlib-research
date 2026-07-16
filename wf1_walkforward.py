"""ROLLING WALK-FORWARD over the PIT S&P 500 universe — the decisive test of whether the
FCF-yield / ROE edge ever generalized out-of-sample, or was always in-sample-only.

Method: for each test year t, lock the factor SIGN on the EXPANDING window of all history
< Jan 1 of year t (causal, no peeking), then score year t out-of-sample:
  - OOS annual RankIC (signed by the locked sign);
  - top-50 net-of-cost book excess vs equal-weight own universe (21 monthly-phase avg);
  - long-short (top50 - bottom50 vs EW).
This turns 24 years into ~23 genuine OOS observations instead of one holdout window.

Alongside: an annual MARKET-BREADTH series (fraction of eligible members whose annual return
beats the equal-weight universe) + cross-sectional dispersion. Narrow / mega-cap-led markets
have LOW breadth. We then correlate annual factor OOS excess with breadth to test the
regime-dependence hypothesis (do the factors fail specifically in low-breadth years?).

2001-2023 is the legitimate walk-forward evidence; 2024-2026 is the already-observed holdout,
computed here for continuity and flagged as such (not a new test).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sn_common as C

FULL = ("1999-06-01", "2026-06-29")
FACTORS = {"fcfy": "$fcfy (FCF yield)", "roe": "$roe (ROE)"}
FIRST_TEST, LAST_WF, LAST_ALL = 2001, 2023, 2026


def daily_rank_ic(fac_vals, fwd_vals, elig, dates):
    """Per-day cross-sectional RankIC aligned to `dates` (NaN where <30 names)."""
    out = np.full(len(dates), np.nan)
    for t in range(len(dates)):
        g = elig[t] & np.isfinite(fac_vals[t]) & np.isfinite(fwd_vals[t])
        if g.sum() > 30:
            out[t] = np.corrcoef(pd.Series(fac_vals[t][g]).rank(),
                                 pd.Series(fwd_vals[t][g]).rank())[0, 1]
    return out


def annual_breadth(retv_y, elig_y):
    """From one year's daily member returns: breadth = frac of names (>=100 elig days) whose
    annual compounded return beats the equal-weight universe; dispersion = std of those."""
    T, N = retv_y.shape
    acc = np.ones(N); days = np.zeros(N)
    for t in range(T):
        m = elig_y[t] & np.isfinite(retv_y[t])
        acc[m] *= (1 + retv_y[t][m]); days[m] += 1
    keep = days >= 100
    cr = acc[keep] - 1
    if len(cr) < 30:
        return np.nan, np.nan
    return float((cr >= cr.mean()).mean()), float(cr.std())


def main():
    close, mat, fund, sector = C.load_base()
    memb, retv_full, fwd_full = C.make_windows(close, mat)
    raw = C.build_raw_factors(close, fund)

    d_all, elig_all, retv_all, fwd_all, ew_all = C.slice_window(
        close, memb, retv_full, fwd_full, *FULL)
    yrs = d_all.year
    dic = {k: daily_rank_ic(raw[k].loc[d_all].values, fwd_all, elig_all, d_all) for k in FACTORS}

    rows = []
    for y in range(FIRST_TEST, LAST_ALL + 1):
        d, elig, retv, fwd, ew = C.slice_window(
            close, memb, retv_full, fwd_full, f"{y}-01-01", f"{y}-12-31")
        if len(d) < 120:
            continue
        breadth, disp = annual_breadth(retv, elig)
        rec = {"year": y, "ew": C.ann(ew), "breadth": breadth, "disp": disp,
               "n": int(elig.sum(1).mean())}
        for k in FACTORS:
            prior = dic[k][yrs < y]                       # expanding-window sign fit (causal)
            sign = 1.0 if np.nanmean(prior) >= 0 else -1.0
            m = C.evaluate(raw[k], d, elig, retv, fwd, ew, sign, k=50, label=k)
            rec[f"{k}_sign"] = int(sign)
            rec[f"{k}_ic"] = m["ic_signed"]               # OOS IC, signed by the locked sign
            rec[f"{k}_ex"] = m["top_ex"]                  # top-50 net excess vs EW
            rec[f"{k}_ls"] = m["ls"]
        rows.append(rec)
        print(f"  {y}: breadth {breadth:.0%}  EW {rec['ew']:+.1%}  "
              + "  ".join(f"{k} IC {rec[f'{k}_ic']:+.3f} ex {rec[f'{k}_ex']:+.1%}"
                          for k in FACTORS), flush=True)

    wf = pd.DataFrame(rows).set_index("year")
    wf.to_csv(C.SH / "wf_walkforward.csv")
    dev = wf.loc[wf.index <= LAST_WF]
    hold = wf.loc[wf.index > LAST_WF]

    print(f"\n{'='*72}\nWALK-FORWARD SUMMARY  (2001-2023 = {len(dev)} genuine OOS years)\n{'='*72}")
    for k, name in FACTORS.items():
        ic = dev[f"{k}_ic"]; ex = dev[f"{k}_ex"]
        t_ex = ex.mean() / ex.std() * np.sqrt(len(ex)) if ex.std() > 0 else 0
        print(f"\n{name}:")
        print(f"  OOS IC same-sign years : {(ic > 0).sum()}/{len(ic)} ({(ic>0).mean():.0%})   "
              f"mean OOS IC {ic.mean():+.4f}")
        print(f"  top-50 net excess/yr   : mean {ex.mean():+.2%}   median {ex.median():+.2%}   "
              f"years>0 {(ex>0).sum()}/{len(ex)} ({(ex>0).mean():.0%})   t-stat {t_ex:+.2f}")
        r = np.corrcoef(dev["breadth"], ex)[0, 1]
        print(f"  corr(annual excess, breadth): {r:+.2f}  "
              f"(+ => factor does BETTER in high-breadth years / worse when narrow)")
    print(f"\n  breadth: 2001-2023 mean {dev['breadth'].mean():.0%} "
          f"(min {dev['breadth'].min():.0%} in {dev['breadth'].idxmin()}, "
          f"max {dev['breadth'].max():.0%} in {dev['breadth'].idxmax()})")
    if len(hold):
        print(f"\n  --- 2024-2026 (already-observed holdout, not a new test) ---")
        for y, r in hold.iterrows():
            print(f"  {y}: breadth {r['breadth']:.0%}  "
                  + "  ".join(f"{k} IC {r[f'{k}_ic']:+.3f} ex {r[f'{k}_ex']:+.1%}" for k in FACTORS))
    print(f"\nsaved wf_walkforward.csv")


if __name__ == "__main__":
    main()
