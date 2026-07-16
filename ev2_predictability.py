"""MILESTONE 1b — is the earnings-move MAGNITUDE predictable? (the covered-call question)

Predictors, all strictly causal (known before the event):
  base_vol : trailing 60d daily vol ending 5d before the event (the naive baseline)
  prior4   : mean of the stock's previous 4 earnings |moves| (persistence)
  combo    : prior4_norm x base_vol  (typical earnings multiple x current vol regime)

Evaluation: CROSS-SECTIONAL Spearman per calendar quarter (predict which stocks move most
this earnings season), reported separately for DEV (<=2023) and HOLDOUT (2024-2026).
PRE-REGISTERED success bar: holdout Spearman > 0.3 for combo + monotone tail calibration.

Tail calibration = the assignment-risk gauge in action: bucket events by predicted-move
quintile (within quarter), report realized P(|move|>5%) and P(>10%) per bucket.

Also: per-stock earnings-move PROFILES saved to earnings_move_profiles.csv (median/p90 move,
tail probs, current trailing-4 expected move) -- the live covered-call lookup table.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
DEV_END = pd.Timestamp("2023-12-31")


def qspearman(df, pred, n_min=30):
    """Mean per-quarter cross-sectional Spearman corr(pred, absmove) + stats."""
    out = []
    for q, g in df.groupby("qtr"):
        g = g.dropna(subset=[pred, "absmove"])
        if len(g) >= n_min:
            out.append(g[pred].corr(g["absmove"], method="spearman"))
    s = pd.Series(out)
    t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else 0
    return s.mean(), t, len(s), (s > 0).mean()


def main():
    ev = pd.read_hdf(SH / "earnings_events.h5")
    ev = ev.dropna(subset=["absmove", "base_vol"])
    ev["qtr"] = ev.event_day.dt.to_period("Q")
    g = ev.groupby("ticker")
    ev["prior4"] = g["absmove"].transform(lambda s: s.shift(1).rolling(4).mean())
    ev["prior4_norm"] = g["norm"].transform(lambda s: s.shift(1).rolling(4).mean())
    ev["combo"] = ev["prior4_norm"] * ev["base_vol"]
    evp = ev.dropna(subset=["prior4", "combo"])
    dev = evp[evp.event_day <= DEV_END]
    hold = evp[evp.event_day > DEV_END]
    print(f"events with full predictors: {len(evp):,} "
          f"(dev {len(dev):,}, holdout {len(hold):,})\n")

    print(f"=== cross-sectional Spearman(pred, realized |move|) per quarter ===")
    print(f"{'predictor':>10} | {'DEV mean':>9}{'t':>7}{'q>0':>6} | {'HOLD mean':>10}{'t':>7}{'q>0':>6}")
    for p in ["base_vol", "prior4", "combo"]:
        dm, dt_, dn, dp = qspearman(dev, p)
        hm, ht, hn, hp = qspearman(hold, p)
        print(f"{p:>10} | {dm:>9.3f}{dt_:>7.1f}{dp:>6.0%} | {hm:>10.3f}{ht:>7.1f}{hp:>6.0%}")

    # per-stock persistence (split-half): stocks with >=16 events
    prof_src = ev.groupby("ticker").filter(lambda x: len(x) >= 16)
    halves = prof_src.groupby("ticker").apply(
        lambda x: pd.Series({"h1": x.absmove.iloc[:len(x)//2].mean(),
                             "h2": x.absmove.iloc[len(x)//2:].mean()}))
    print(f"\nper-stock persistence (split-half corr of mean |move|, "
          f"{len(halves)} stocks >=16 events): {halves.h1.corr(halves.h2):+.2f}")

    # tail calibration on HOLDOUT = the assignment-risk gauge
    print(f"\n=== HOLDOUT tail calibration by predicted-move quintile (combo, within-qtr) ===")
    h = hold.copy()
    h["bucket"] = h.groupby("qtr")["combo"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    tab = h.groupby("bucket").agg(pred=("combo", "median"), realized=("absmove", "median"),
                                  p5=("absmove", lambda s: (s > .05).mean()),
                                  p10=("absmove", lambda s: (s > .10).mean()),
                                  n=("absmove", "size"))
    for b, r in tab.iterrows():
        print(f"  Q{int(b)+1}: pred {r['pred']:.1%}  realized median {r['realized']:.1%}  "
              f"P(>5%) {r['p5']:.0%}  P(>10%) {r['p10']:.0%}  (n={int(r['n']):,})")
    mono = tab["realized"].is_monotonic_increasing and tab["p5"].is_monotonic_increasing
    print(f"  calibration monotone: {mono}")

    # per-stock profiles: the live covered-call lookup
    prof = prof_src.groupby("ticker").agg(
        universe=("universe", "last"), n=("absmove", "size"),
        med_move=("absmove", "median"), p75=("absmove", lambda s: s.quantile(.75)),
        p90=("absmove", lambda s: s.quantile(.9)),
        p_gt5=("absmove", lambda s: (s > .05).mean()),
        p_gt10=("absmove", lambda s: (s > .10).mean()),
        last_event=("event_day", "max"),
        trail4=("absmove", lambda s: s.iloc[-4:].mean()))
    prof.to_csv(SH / "earnings_move_profiles.csv")
    print(f"\nsaved earnings_move_profiles.csv: {len(prof):,} stocks (>=16 events)")
    print("sample (largest current expected movers, active since 2025):")
    act = prof[prof.last_event >= "2025-06-01"].nlargest(6, "trail4")
    for t, r in act.iterrows():
        print(f"  {t:6} {r.universe:6} trail4 {r.trail4:.1%}  med {r.med_move:.1%}  "
              f"P(>10%) {r.p_gt10:.0%}")
    act2 = prof[prof.last_event >= "2025-06-01"].nsmallest(6, "trail4")
    print("quietest (safest covered-call earnings cycles):")
    for t, r in act2.iterrows():
        print(f"  {t:6} {r.universe:6} trail4 {r.trail4:.1%}  med {r.med_move:.1%}  "
              f"P(>5%) {r.p_gt5:.0%}")

    # verdict vs pre-registered bar
    hm, ht, hn, hp = qspearman(hold, "combo")
    ok = hm > 0.3 and mono
    print(f"\n=== PRE-REGISTERED VERDICT: holdout combo Spearman {hm:.3f} (bar 0.3), "
          f"calibration monotone {mono} -> {'PASS' if ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
