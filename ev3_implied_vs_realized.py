"""MILESTONE 2 calculator — implied earnings move (variance decomposition) vs our predicted
realized move, per name. Feed it straddle quotes (manual IBKR pulls for now), it does the rest.

Input CSV git_ignore_folder/sharadar/straddle_quotes.csv, one row per name:
  ticker,spot,pre_dte,pre_straddle,post_dte,post_straddle
  pre  = expiry BEFORE the upcoming earnings (no event variance)
  post = expiry AFTER it (event + extra diffusive days)

Method (validated by hand on NVDA 2026-07):
  sigma_total = (straddle/spot)/0.7979          # ATM straddle ~ 0.8*S*sigma*sqrt(T)
  var_event = var_post - var_pre - var_pre*(post_dte-pre_dte)/pre_dte
  implied_sigma_event = sqrt(max(var_event,0))

Realized side (from earnings_events.h5), in MATCHED sigma units:
  pred_sigma = trailing-8 RMS of signed moves     # recency-weighted (ev2: prior-k wins)
  trail4     = trailing-4 mean |move| (the ev2-validated predictor; /0.7979 -> sigma)
Premium = implied_sigma_event / pred_sigma - 1.  Screen bar: >= +30% (thin premia are
within estimation noise on 4-8 samples and one fat tail erases them).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
K = 0.7979          # E|N(0,1)|
TODAY = pd.Timestamp("2026-07-16")


def implied_sigma_event(spot, pre_dte, pre_straddle, post_dte, post_straddle):
    s_pre = (pre_straddle / spot) / K
    s_post = (post_straddle / spot) / K
    var_pre, var_post = s_pre ** 2, s_post ** 2
    var_decay = var_pre * (post_dte - pre_dte) / pre_dte
    var_event = var_post - var_pre - var_decay
    return float(np.sqrt(max(var_event, 0.0))), var_event


def bracket_warn(row):
    """Catch the silent failure: expiries that don't straddle the true earnings date, or a
    quote error. Returns a warning string or ''. Needs optional 'earnings_date' column."""
    if "earnings_date" in row and pd.notna(row.get("earnings_date")):
        de = (pd.Timestamp(row["earnings_date"]) - TODAY).days
        if not (row.pre_dte < de < row.post_dte):
            return (f"BRACKET ERROR: earnings in {de}d not between pre {int(row.pre_dte)}d "
                    f"and post {int(row.post_dte)}d -> decomposition invalid, re-pick expiries")
    return ""


def main():
    ev = pd.read_hdf(SH / "earnings_events.h5")
    qf = SH / "straddle_quotes.csv"
    if not qf.exists():
        pd.DataFrame([["NVDA", 209.85, 36, 21.51, 63, 30.81]],
                     columns=["ticker", "spot", "pre_dte", "pre_straddle",
                              "post_dte", "post_straddle"]).to_csv(qf, index=False)
        print(f"wrote template {qf} (seeded with the NVDA hand-calc row) -- add rows and rerun")
    q = pd.read_csv(qf)
    print(f"{'tkr':>6} {'impl_sig':>9} {'pred_sig':>9} {'premium':>8} {'trail4':>7} "
          f"{'hist_med':>9} {'P>10%':>6} {'n':>4}  verdict")
    warns = []
    for _, r in q.iterrows():
        if pd.isna(r.get("pre_straddle")) or pd.isna(r.get("post_straddle")):
            print(f"{r.ticker:>6} {'--':>9}  (awaiting straddle quotes)")
            continue
        isig, var_event = implied_sigma_event(r.spot, r.pre_dte, r.pre_straddle,
                                              r.post_dte, r.post_straddle)
        h = ev[ev.ticker == r.ticker].sort_values("event_day")
        if len(h) < 8:
            print(f"{r.ticker:>6} {isig:>9.2%} {'--':>9}  (insufficient event history)")
            continue
        m = h["move"].values
        pred_sig = float(np.sqrt((m[-8:] ** 2).mean()))       # trailing-8 RMS
        trail4 = float(np.abs(m[-4:]).mean())
        prem = isig / pred_sig - 1
        # silent-failure guards: bad bracketing / quote makes var_event tiny or negative,
        # which reads as a fake huge premium -- flag instead of trusting it
        bw = bracket_warn(r)
        if bw:
            warns.append(f"{r.ticker}: {bw}")
        elif var_event <= 0 or isig < 0.4 * pred_sig:
            warns.append(f"{r.ticker}: implied σ {isig:.1%} implausibly low vs realized "
                         f"{pred_sig:.1%} -- likely both expiries same side of earnings, "
                         f"or a quote/DTE error (NOT a real premium)")
        verdict = ("HARVEST candidate" if prem >= .30 else
                   "thin -- skip" if prem >= 0 else "implied CHEAP -- do not sell")
        if bw or var_event <= 0:
            verdict = "⚠ CHECK BRACKETING (see warning)"
        print(f"{r.ticker:>6} {isig:>9.2%} {pred_sig:>9.2%} {prem:>+8.0%} {trail4:>7.2%} "
              f"{np.median(np.abs(m)):>9.2%} {(np.abs(m)>.10).mean():>6.0%} {len(m):>4}  {verdict}")
    for w in warns:
        print(f"  ⚠ {w}")


if __name__ == "__main__":
    main()
