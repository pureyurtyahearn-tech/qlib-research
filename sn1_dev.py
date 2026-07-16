"""STEP 3 -- development-only (<=2023) comparison of RAW vs SECTOR-NEUTRAL factors.
Looks at DEV data ONLY. No holdout is touched here. Decides which neutral variants (if any)
look genuinely promising enough to justify the one-shot 2024-26 holdout test in sn2.

Factors: $fcfy (value), $roe (quality), 12-1 momentum (technical).
Neutralization: within-(date,sector) z-score, 11 Sharadar sectors, eligible members only.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sn_common as C

DEV_START, DEV_END = "1999-06-01", "2023-12-31"


def main():
    close, mat, fund, sector = C.load_base()
    print(f"universe cols {close.columns.size}, sectors {sector.nunique()}: "
          + ", ".join(f"{s}={n}" for s, n in sector.value_counts().items()))
    memb, retv_full, fwd_full = C.make_windows(close, mat)
    elig_full = pd.DataFrame(memb.values & np.isfinite(close.values),
                             index=close.index, columns=close.columns)

    raw = C.build_raw_factors(close, fund)
    neu = {k: C.sector_zscore(v, sector, elig_full) for k, v in raw.items()}

    d, elig, retv, fwd, ew = C.slice_window(close, memb, retv_full, fwd_full, DEV_START, DEV_END)
    print(f"\n=== DEV {d[0].date()}..{d[-1].date()} ({len(d)/252:.1f}y)  "
          f"EW mkt {C.ann(ew):+.4f}/yr ===\n")

    names = {"fcfy": "FCF yield (value)", "roe": "ROE (quality)", "mom": "12-1 momentum"}
    dev_signs = {}
    for key in ["fcfy", "roe", "mom"]:
        print(f"--- {names[key]} ---")
        # lock sign from RAW dev IC (same sign used for both variants, for comparability)
        ic_raw = C.rank_ic(raw[key].loc[d].values, fwd, elig).mean()
        sign = 1.0 if ic_raw >= 0 else -1.0
        dev_signs[key] = sign
        print(f"  DEV raw RankIC {ic_raw:+.5f} -> LOCKED sign {int(sign):+d} "
              f"({'high=long' if sign > 0 else 'high=short'})")
        mr = C.evaluate(raw[key], d, elig, retv, fwd, ew, sign, label="RAW")
        mn = C.evaluate(neu[key], d, elig, retv, fwd, ew, sign, label="SECTOR-NEUTRAL")
        C.print_eval(mr)
        C.print_eval(mn)
        print()

    print("=== promising-on-dev screen (neutral variant): IC |z|>2, monotonic quintiles, "
          "long-short>0, top>EW ===")
    for key in ["fcfy", "roe", "mom"]:
        sign = dev_signs[key]
        mn = C.evaluate(neu[key], d, elig, retv, fwd, ew, sign, label="NEU")
        promising = (mn["z"] > 2 and mn["ic_signed"] > 0 and mn["mono"]
                     and mn["ls"] > 0 and mn["top_ex"] > 0)
        print(f"  {names[key]:22} neutral: |z| {mn['z']:.1f}  mono {mn['mono']}  "
              f"LS {mn['ls']:+.4f}  top-EW {mn['top_ex']:+.4f}  "
              f"-> {'PROMISING (justifies holdout)' if promising else 'weak on dev (skip holdout)'}")


if __name__ == "__main__":
    main()
