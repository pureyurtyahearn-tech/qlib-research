"""NASDAQ-only fundamentals — ONE-SHOT HOLDOUT (2024-01-01..2026-06-29).
Runs ONLY on factors that passed all 4 dev criteria in nqf1 (read from nqf_dev_pass.csv),
sign locked from TRAIN. One shot, no revisits. Same 4-criteria verdict as the dev screen.

NOTE: if nqf_dev_pass.csv is empty, NOTHING is tested and the holdout is NOT touched --
that is the correct behavior and this script exits without loading holdout returns.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nqf_common as N
import sn_common as C

TRAIN = ("2000-01-01", "2019-12-31")
HOLD = ("2024-01-01", "2026-06-29")


def main():
    pf = N.SH / "nqf_dev_pass.csv"
    passed = pd.read_csv(pf) if pf.exists() else pd.DataFrame(columns=["factor", "sign"])
    if len(passed) == 0:
        print("no factors passed the dev screen -> holdout NOT touched. Nothing to test.")
        return

    close, vol, fund = N.load_nasdaq()
    elig = N.build_elig(close, vol)
    retv_full, fwd_full = N.make_windows(close)
    cols = close.columns
    dh, eh, rh, fh, ewh = N.slice_window(close, elig, retv_full, fwd_full, *HOLD)
    print(f"=== HOLDOUT {dh[0].date()}..{dh[-1].date()} ({len(dh)/252:.1f}y, "
          f"EW {C.ann(ewh):+.4f}/yr) ===  testing {list(passed.factor)}\n")

    for _, row in passed.iterrows():
        fk, sign = row["factor"], float(row["sign"])
        fac = N.factor_wide(fund, fk, close.index, cols)
        m = C.evaluate(fac, dh, eh, rh, fh, ewh, sign, k=50, label="HOLDOUT")
        c1 = m["ic_signed"] > 0 and m["z"] > 2
        c2 = m["mono"]; c3 = m["ls"] > 0
        c4 = m["top_ex"] > 0 and m["top_ex"] > m["bot_ex"] + 0.01
        print(f"--- {fk} (sign {int(sign):+d}) ---")
        print(f"  IC {m['ic']:+.5f} signed {m['ic_signed']:+.5f} |z| {m['z']:.1f}  "
              f"{'SAME sign' if m['ic_signed'] > 0 else 'FLIPPED'}")
        print(f"  quintiles Q1..Q5: " + "  ".join(f"{q:+.3f}" for q in m["q"])
              + f"  {'MONO' if m['mono'] else 'NON-mono'}")
        print(f"  top-50 vs EW {m['top_ex']:+.4f}  bottom-50 {m['bot_ex']:+.4f}  "
              f"long-short {m['ls']:+.4f}")
        ok = c1 and c2 and c3 and c4
        print(f"  VERDICT: (1){'P' if c1 else 'F'} (2){'P' if c2 else 'F'} "
              f"(3){'P' if c3 else 'F'} (4){'P' if c4 else 'F'} -> "
              f"{fk} {'CONFIRMS' if ok else 'does NOT confirm'} on holdout\n")


if __name__ == "__main__":
    main()
