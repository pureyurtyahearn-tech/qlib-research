"""HONEST DEVELOPMENT-ONLY re-evaluation. The 2024-2026 holdout is NOT loaded at all.

All data hard-capped at 2023-12-31. Within development: sign fit 1999-2011, evaluate
(IC vs placebo, turnover, net-of-cost vs EW) on 2012-2023. This removes the evaluation-side
holdout contamination. (The generation-side contamination from the 10-loop run's 2023-2026
backtest feedback still stands -- fixed only by a re-run capped at 2023.)

The single best factor selected here on DEV data is the ONE we would later confirm on the
untouched 2024-2026 holdout.
"""
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from ext6_momentum_full import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
WS = Path("git_ignore_folder/RD-Agent_workspace")
CUTOFF = datetime(2026, 7, 15, 13, 35).timestamp()
FUND_COLS = {"$pe", "$pb", "$ey", "$de", "$roe", "$rgrow", "$fcfy"}
DEV_END = "2023-12-31"          # <<< holdout begins 2024-01-01; nothing past DEV_END is touched
FIT_END = "2011-12-31"          # sign fit in-sample within development


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        g = mask[t] & np.isfinite(sig[t]) & np.isfinite(fwd[t])
        if g.sum() > 30:
            out.append(np.corrcoef(pd.Series(sig[t][g]).rank(), pd.Series(fwd[t][g]).rank())[0, 1])
    out = np.array(out); return out[np.isfinite(out)]


def collect():
    facs = {}
    for fp in WS.glob("*/factor.py"):
        if fp.stat().st_mtime < CUTOFF:
            continue
        res = fp.parent / "result.h5"
        if not res.exists():
            continue
        txt = fp.read_text(errors="ignore")
        m = re.search(r"to_frame\(name=['\"]([^'\"]+)['\"]", txt)
        cols = set(re.findall(r"\$[a-z_]+", txt))
        nm = (m.group(1) if m else "") or ("|".join(sorted(cols & FUND_COLS)) or fp.parent.name[:8])
        facs[fp.parent.name] = (nm, bool(cols & FUND_COLS), res)
    return facs


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = px["$close"].unstack("ticker").sort_index()
    close = close.loc[close.index <= DEV_END]           # <<< HARD CAP: holdout never loaded
    print(f"DEVELOPMENT data only: {close.index.min().date()} .. {close.index.max().date()}")
    print(f"(2024-01-01 -> 2026-06 is HELD OUT and not present in this evaluation)\n")

    S = "1999-06-01"
    dates = close.index[close.index >= S]; T = len(dates); cols = close.columns
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(columns=cols, fill_value=False)
    retv = close.pct_change().loc[dates].values
    elig = memb.loc[dates].values & np.isfinite(close.loc[dates].values)
    fwd21 = (close.shift(-22) / close.shift(-1) - 1).loc[dates].values
    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
    fitmask = dates <= FIT_END
    evalmask = dates > FIT_END                          # 2012-2023 dev-OOS

    rng = np.random.default_rng(0)
    nsd = float(np.std([rank_ic(rng.standard_normal((T, len(cols))), fwd21, elig).mean() for _ in range(12)]))
    print(f"eval window {dates[evalmask][0].date()}..{dates[evalmask][-1].date()}  "
          f"EW bench {ann(ew[evalmask]):+.4f}/yr  IC noise sd {nsd:.5f}\n")

    rows = []
    for wsid, (nm, isf, res) in collect().items():
        try:
            s = pd.read_hdf(res); s = s.iloc[:, 0] if s.ndim > 1 else s
            raw = s.unstack("instrument").reindex(index=dates, columns=cols).values.astype(float)
        except Exception:
            continue
        if np.isfinite(raw).mean() < 0.2:
            continue
        ic = rank_ic(raw[evalmask], fwd21[evalmask], elig[evalmask])
        if len(ic) == 0:
            continue
        sgn = 1.0 if rank_ic(raw[fitmask], fwd21[fitmask], elig[fitmask]).mean() >= 0 else -1.0
        sig = pd.DataFrame(sgn * raw, index=dates, columns=cols).shift(1).values
        # net-of-cost across all 21 phases on 2012-2023 (phase-robust)
        exph = []
        for ph in range(21):
            rb = np.zeros(T, bool); rb[ph::21] = True
            net, _, turn = simulate(sig, retv, elig, rb, 50)
            exph.append(ann((net - ew)[evalmask]))
        exph = np.array(exph)
        rows.append(dict(name=nm[:32], fund=isf, absIC=abs(ic.mean()), z=abs(ic.mean()) / nsd,
                         turn=turn, netEx=exph.mean(), netMin=exph.min(), pctpos=100 * (exph > 0).mean()))
    df = pd.DataFrame(rows).sort_values("absIC", ascending=False)

    print(f"{'factor':34}{'|IC|':>8}{'|z|':>6}{'turn':>7}{'netEx':>9}{'phMin':>9}{'ph>0%':>7}  DEV 2012-2023")
    for _, r in df[df.fund].head(12).iterrows():
        print(f"{r['name']:34}{r['absIC']:>8.4f}{r['z']:>6.1f}{r['turn']:>6.1f}x"
              f"{r['netEx']:>+9.4f}{r['netMin']:>+9.4f}{r['pctpos']:>6.0f}%")
    print()
    fund = df[df.fund]
    # SELECT: best by dev IC among low-turnover (<3x) fundamentals with robust net excess
    cand = fund[(fund.turn < 3.0) & (fund.pctpos >= 90)].sort_values("z", ascending=False)
    if len(cand):
        b = cand.iloc[0]
        print(f"SELECTED for holdout confirmation (best dev IC among low-turnover, phase-robust):")
        print(f"  {b['name']}  |IC| {b['absIC']:.4f} (z {b['z']:.1f}), turnover {b['turn']:.1f}x, "
              f"dev netEx {b['netEx']:+.4f} ({b['pctpos']:.0f}% phases>0)")
    df.to_csv(SH / "devonly_factor_eval.csv", index=False)
    print(f"\nHoldout 2024-2026 remains UNTOUCHED. Run the final test only when ready.")


if __name__ == "__main__":
    main()
