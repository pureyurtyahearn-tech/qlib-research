"""Evaluate the factors RD-Agent generated in the 10-loop fundamentals run, on the
PIT-correct pandas simulator (NOT qlib's ghost-buggy native backtest).

For each generated factor result.h5: |RankIC| vs the placebo noise floor, signal
persistence AC(21), TURNOVER (headline), and net-of-cost excess vs the EW PIT benchmark
(sign fit in-sample on 1999-2012, evaluated 2013-2026). Flags fundamentals vs technical.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from ext6_momentum_full import simulate, ann, ir, tstat

SH = Path("git_ignore_folder/sharadar")
WS = Path("git_ignore_folder/RD-Agent_workspace")
CUTOFF = datetime(2026, 7, 15, 13, 35).timestamp()
FUND_COLS = {"$pe", "$pb", "$ey", "$de", "$roe", "$rgrow", "$fcfy"}
FIT_END = "2012-12-31"


def rank_ic(sig, fwd, mask):
    out = []
    for t in range(len(sig)):
        a, b, m = sig[t], fwd[t], mask[t]
        g = m & np.isfinite(a) & np.isfinite(b)
        if g.sum() > 30:
            out.append(np.corrcoef(pd.Series(a[g]).rank(), pd.Series(b[g]).rank())[0, 1])
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
        nm = ""
        import re
        m = re.search(r"to_frame\(name=['\"]([^'\"]+)['\"]", txt)
        if m: nm = m.group(1)
        cols = set(re.findall(r"\$[a-z_]+", txt))
        uses_fund = bool(cols & FUND_COLS)
        if not nm:
            nm = "|".join(sorted(cols & FUND_COLS)) or fp.parent.name[:8]
        facs[fp.parent.name] = (nm, uses_fund, res)
    return facs


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    mat = pd.read_hdf(SH / "sp500_pit_membership_full.h5")
    close = px["$close"].unstack("ticker").sort_index()
    S = "1999-06-01"
    w = close.index >= S
    dates = close.index[w]; T = len(dates); cols = close.columns
    memb = mat.reindex(index=close.index, method="ffill").fillna(False).reindex(columns=cols, fill_value=False)
    retv = close.pct_change().loc[dates].values
    elig = memb.loc[dates].values & np.isfinite(close.loc[dates].values)
    fwd21 = (close.shift(-22) / close.shift(-1) - 1).loc[dates].values
    ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(T)])
    fitmask = dates <= FIT_END

    rng = np.random.default_rng(0)
    nsd = float(np.std([rank_ic(rng.standard_normal((T, len(cols))), fwd21, elig).mean() for _ in range(12)]))
    print(f"window {dates[0].date()}..{dates[-1].date()} ({T/252:.1f}y)  EW bench {ann(ew):+.4f}/yr  IC noise sd {nsd:.5f}\n")

    facs = collect()
    print(f"evaluating {len(facs)} generated factors\n")
    rows = []
    for wsid, (nm, isf, res) in facs.items():
        try:
            s = pd.read_hdf(res); s = s.iloc[:, 0] if s.ndim > 1 else s
            wide = s.unstack("instrument").reindex(index=dates, columns=cols)
        except Exception:
            continue
        raw = wide.values.astype(float)
        if np.isfinite(raw).mean() < 0.2:
            continue
        ic = rank_ic(raw, fwd21, elig)
        if len(ic) == 0:
            continue
        icm = ic.mean()
        # sign from in-sample IC
        ic_is = rank_ic(raw[fitmask], fwd21[fitmask], elig[fitmask])
        sgn = 1.0 if ic_is.mean() >= 0 else -1.0
        sig = pd.DataFrame(sgn * raw, index=dates, columns=cols).shift(1).values
        # turnover + net excess, OOS (2013+), monthly phase 0
        oos = dates > FIT_END
        rb = np.zeros(T, bool); rb[::21] = True
        net, _, turn = simulate(sig, retv, elig, rb, 50)
        exo = (net - ew)[oos]
        rows.append(dict(name=nm[:34], fund=isf, absIC=abs(icm), z=abs(icm) / nsd,
                         turn=turn, netEx=ann(exo), ir=ir(exo), t=tstat(exo)))
    df = pd.DataFrame(rows).sort_values("absIC", ascending=False)

    def show(d, title):
        print(f"=== {title} ===")
        print(f"{'factor':36}{'|IC|':>8}{'|z|':>6}{'turn/yr':>9}{'netEx(OOS)':>12}{'IR':>7}{'t':>7}")
        for _, r in d.iterrows():
            print(f"{r['name']:36}{r['absIC']:>8.4f}{r['z']:>6.1f}{r['turn']:>8.2f}x"
                  f"{r['netEx']:>+12.4f}{r['ir']:>+7.2f}{r['t']:>+7.2f}")
        print()
    show(df[df.fund].head(15), "FUNDAMENTALS-BASED factors (incl. fund x momentum), by |IC|")
    show(df[~df.fund].head(8), "PURE TECHNICAL factors (control), by |IC|")

    fund = df[df.fund]
    print("=== TURNOVER SUMMARY (the headline) ===")
    print(f"  fundamentals factors: turnover median {fund['turn'].median():.2f}x/yr, "
          f"range {fund['turn'].min():.2f}-{fund['turn'].max():.2f}x")
    print(f"  vs this week: 12-1 momentum ~4.4x, fast-technical ~10x, daily L/S 180-755x")
    tech = df[~df.fund]
    print(f"  (pure-technical from THIS run: median {tech['turn'].median():.2f}x/yr)")
    best = fund.loc[fund['z'].idxmax()]
    print(f"\n  strongest-IC fundamentals factor: {best['name']} "
          f"(|IC| {best['absIC']:.4f}, z {best['z']:.1f}, turnover {best['turn']:.2f}x, "
          f"netEx {best['netEx']:+.4f}, t {best['t']:+.2f})")
    df.to_csv(SH / "rdagent_factor_eval.csv", index=False)


if __name__ == "__main__":
    main()
