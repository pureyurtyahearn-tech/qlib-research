"""ONE-SHOT HOLDOUT TEST #3 -- pre-registered factor: SECTOR-NEUTRAL $fcfy (FCF yield).
This is the ONLY factor that passed the sn1 dev screen (ROE and momentum neutral variants
failed on development data itself and are NOT tested on the holdout).

Hypothesis (legitimately distinct from holdout #1): does the SECTOR-NEUTRAL FCF yield factor
work out-of-sample, where the RAW version failed (ext18) via a U-shaped, mega-cap-tech
sector-concentration mechanism? Sign LOCKED from dev = +1 (high FCF yield = long).
Evaluated ONCE on 2024-01-01..2026-06-29.

Pre-registered 'CONFIRMS' = ALL four (identical criteria to ext19):
  (1) holdout RankIC SAME SIGN as dev + |z|>2 vs placebo;
  (2) quintiles ~monotonic (Q5 > Q3 > Q1);
  (3) long-short (top50 - bottom50 vs EW) positive;
  (4) long-only top-50 beats EW AND bottom-50 does NOT beat EW similarly (not extremeness).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sn_common as C

DEV_START, DEV_END = "1999-06-01", "2023-12-31"
HOLD_START, HOLD_END = "2024-01-01", "2026-06-29"
KEY = "fcfy"


def main():
    close, mat, fund, sector = C.load_base()
    memb, retv_full, fwd_full = C.make_windows(close, mat)
    elig_full = pd.DataFrame(memb.values & np.isfinite(close.values),
                             index=close.index, columns=close.columns)
    neu = C.sector_zscore(C.build_raw_factors(close, fund)[KEY], sector, elig_full)

    # ---- lock sign on DEV (neutral variant) ----
    d0, elig0, retv0, fwd0, ew0 = C.slice_window(close, memb, retv_full, fwd_full, DEV_START, DEV_END)
    ic_dev = C.rank_ic(neu.loc[d0].values, fwd0, elig0).mean()
    sign = 1.0 if ic_dev >= 0 else -1.0
    print(f"PRE-REGISTERED: SECTOR-NEUTRAL {KEY} (FCF yield). DEV(<=2023) neutral RankIC "
          f"{ic_dev:+.5f} -> LOCKED sign {int(sign):+d} (high FCF yield = "
          f"{'LONG' if sign > 0 else 'SHORT'})\n")

    # ---- holdout (touched once) ----
    d, elig, retv, fwd, ew = C.slice_window(close, memb, retv_full, fwd_full, HOLD_START, HOLD_END)
    print(f"=== HOLDOUT {d[0].date()}..{d[-1].date()} ({len(d)/252:.1f}y)  "
          f"EW mkt {C.ann(ew):+.4f}/yr ===\n")
    m = C.evaluate(neu, d, elig, retv, fwd, ew, sign, label="SECTOR-NEUTRAL holdout")

    print(f"(1) IC: raw {m['ic']:+.5f}  signed(dev) {m['ic_signed']:+.5f}  placebo sd "
          f"{m['nsd']:.5f}  |z| {m['z']:.1f}  "
          f"{'SAME sign as dev' if m['ic_signed'] > 0 else 'SIGN FLIPPED vs dev'}")
    print(f"(2) quintiles gross ann Q1(low)..Q5(high): " + "  ".join(f"Q{i+1} {m['q'][i]:+.3f}"
          for i in range(5)) + f"   {'MONOTONIC-ish' if m['mono'] else 'NON-monotonic'}")
    print(f"(3) long-short (top50 - bottom50 vs EW): {m['ls']:+.4f}/yr")
    print(f"(4) long-only TOP-50 vs EW: {m['top_ex']:+.4f}/yr "
          f"({100*m['top_pp']:.0f}% phases>0, turn {m['turn']:.2f}x, t {m['top_t']:+.2f})")
    print(f"    long-only BOTTOM-50 vs EW: {m['bot_ex']:+.4f}/yr  "
          f"(if ~same as top -> EXTREMENESS artifact)")

    c1 = m["ic_signed"] > 0 and m["z"] > 2
    c2 = m["mono"]
    c3 = m["ls"] > 0
    c4 = m["top_ex"] > 0 and m["top_ex"] > m["bot_ex"] + 0.01
    print(f"\n=== VERDICT (sector-neutral FCF yield) ===")
    print(f"  (1) IC same sign & significant: {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) quintiles monotonic:        {'PASS' if c2 else 'FAIL'}")
    print(f"  (3) long-short positive:        {'PASS' if c3 else 'FAIL'}")
    print(f"  (4) top>EW & not extremeness:   {'PASS' if c4 else 'FAIL'}")
    ok = c1 and c2 and c3 and c4
    print(f"  --> SECTOR-NEUTRAL {KEY} {'CONFIRMS on the holdout' if ok else 'does NOT confirm on the holdout'}")


if __name__ == "__main__":
    main()
