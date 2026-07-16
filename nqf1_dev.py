"""NASDAQ-only fundamentals — DEVELOPMENT screen (touches TRAIN + DEV only; holdout locked).

Pre-registered split (do NOT change after seeing data):
  TRAIN 2000-2019 (lock factor sign)  DEV 2020-2023 (4-criteria screen)  HOLDOUT 2024-2026 (locked)

Factors: $fcfy (FCF yield), $roe (ROE) -- direct comparison to the SP500 holdout failures.
Dev screen (all 4, same as ext19/sn2):
  (1) IC same sign as TRAIN + significant (|z|>2 vs placebo) on DEV;
  (2) quintiles monotonic on DEV (Q5 > Q3 > Q1 in signal direction);
  (3) long-short (top50 - bottom50 vs EW) positive on DEV;
  (4) top-50 book > EW AND not an extremeness artifact (bottom-50 not ~= top-50).
Only factors passing ALL 4 are written to nqf_dev_pass.csv for the one-shot holdout (nqf2).
Research question: does the SP500 U-shaped extremeness signature appear here, or does the
small/mid, less-crowded universe behave differently?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import nqf_common as N
import sn_common as C

TRAIN = ("2000-01-01", "2019-12-31")
DEV = ("2020-01-01", "2023-12-31")
FACTORS = ["$fcfy", "$roe"]


def main():
    close, vol, fund = N.load_nasdaq()
    print(f"universe: {close.columns.size} NASDAQ-only names w/ fundamentals, "
          f"{close.index.min().date()}..{close.index.max().date()}", flush=True)
    elig = N.build_elig(close, vol)
    retv_full, fwd_full = N.make_windows(close)
    cols = close.columns

    dtr, etr, rtr, ftr, ewtr = N.slice_window(close, elig, retv_full, fwd_full, *TRAIN)
    ddv, edv, rdv, fdv, ewdv = N.slice_window(close, elig, retv_full, fwd_full, *DEV)
    print(f"TRAIN {dtr[0].date()}..{dtr[-1].date()} ({len(dtr)/252:.1f}y, "
          f"elig/day {int(etr.sum(1).mean())})   DEV {ddv[0].date()}..{ddv[-1].date()} "
          f"({len(ddv)/252:.1f}y, elig/day {int(edv.sum(1).mean())}, EW {C.ann(ewdv):+.4f}/yr)\n",
          flush=True)

    passed = []
    for fk in FACTORS:
        fac = N.factor_wide(fund, fk, close.index, cols)
        ic_tr = C.rank_ic(fac.loc[dtr].values, ftr, etr).mean()
        sign = 1.0 if ic_tr >= 0 else -1.0
        print(f"--- {fk} ---  TRAIN RankIC {ic_tr:+.5f} -> LOCKED sign {int(sign):+d} "
              f"({'high=long' if sign > 0 else 'high=short'})", flush=True)
        m = C.evaluate(fac, ddv, edv, rdv, fdv, ewdv, sign, k=50, label="DEV")
        print(f"  DEV RankIC {m['ic']:+.5f} (signed {m['ic_signed']:+.5f}, |z| {m['z']:.1f}, "
              f"placebo sd {m['nsd']:.5f})")
        print(f"  DEV quintiles gross ann Q1(low)..Q5(high): "
              + "  ".join(f"{q:+.3f}" for q in m["q"])
              + f"   {'MONOTONIC' if m['mono'] else 'NON-monotonic (U-shape?)'}")
        print(f"  DEV top-50 vs EW {m['top_ex']:+.4f}/yr (t {m['top_t']:+.2f}, "
              f"{100*m['top_pp']:.0f}% phases>0, turn {m['turn']:.2f}x)")
        print(f"  DEV bottom-50 vs EW {m['bot_ex']:+.4f}/yr    long-short {m['ls']:+.4f}/yr")
        c1 = m["ic_signed"] > 0 and m["z"] > 2
        c2 = m["mono"]
        c3 = m["ls"] > 0
        c4 = m["top_ex"] > 0 and m["top_ex"] > m["bot_ex"] + 0.01
        ok = c1 and c2 and c3 and c4
        print(f"  DEV SCREEN: (1)IC {'P' if c1 else 'F'} (2)mono {'P' if c2 else 'F'} "
              f"(3)LS {'P' if c3 else 'F'} (4)top>EW&not-extreme {'P' if c4 else 'F'} "
              f"-> {'PASS (goes to holdout)' if ok else 'FAIL (no holdout)'}\n", flush=True)
        if ok:
            passed.append((fk, int(sign)))

    pd.DataFrame(passed, columns=["factor", "sign"]).to_csv(N.SH / "nqf_dev_pass.csv", index=False)
    print(f"=== dev-passing factors -> holdout: {[p[0] for p in passed] or 'NONE'} ===")


if __name__ == "__main__":
    main()
