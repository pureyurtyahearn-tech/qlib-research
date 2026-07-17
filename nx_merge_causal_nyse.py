"""Causally merge SF1-ART fundamentals onto daily SEP prices for the NYSE-only universe.
Mechanically identical to ext11 (SP500) / nq7 (NASDAQ): merge_asof(direction='backward') on
datekey (filing date), same 7 factors, same perturbation-based look-ahead check. Only
inputs/outputs differ:
  prices  <- sep_nyse_panel.h5   fundamentals <- sf1_nyse_raw.h5
  output  -> fundamentals_nyse_daily.h5
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
START = "1999-01-01"
RAW = SH / "sf1_nyse_raw.h5"
PANEL = SH / "sep_nyse_panel.h5"
OUT = SH / "fundamentals_nyse_daily.h5"


def load_fundamentals():
    f = pd.read_hdf(RAW)
    for c in ["datekey", "reportperiod", "calendardate"]:
        f[c] = pd.to_datetime(f[c])
    f = f[f["datekey"] >= f["reportperiod"]]
    f = f.sort_values(["ticker", "datekey"])
    f["revenue_yoy"] = f.groupby("ticker")["revenue"].transform(lambda s: s / s.shift(4) - 1)
    return f


def causal_merge(px_long, fund, on_col):
    cols = ["ticker", on_col, "eps", "bvps", "de", "roe", "revenue_yoy", "fcf", "sharesbas"]
    right = fund[cols].dropna(subset=[on_col]).sort_values(on_col)
    return pd.merge_asof(px_long.sort_values("datetime"), right,
                         left_on="datetime", right_on=on_col, by="ticker", direction="backward")


def main():
    px = pd.read_hdf(PANEL)
    close = px["$close"].unstack("ticker").sort_index()
    close = close.loc[close.index >= START]
    px_long = close.stack().rename("close").reset_index()
    px_long.columns = ["datetime", "ticker", "close"]
    print(f"price cells (NYSE-only priced (ticker,date) since {START}): {len(px_long):,}, "
          f"{close.columns.size} tickers", flush=True)

    fund = load_fundamentals()
    print(f"fundamentals: {len(fund):,} filings, {fund.ticker.nunique()} tickers", flush=True)

    m = causal_merge(px_long, fund, "datekey")
    eps, bvps, sh, cl = m["eps"], m["bvps"], m["sharesbas"], m["close"]
    m["$pe"] = np.where(eps > 0, cl / eps, np.nan)
    m["$pb"] = np.where(bvps > 0, cl / bvps, np.nan)
    m["$ey"] = eps / cl
    m["$de"] = m["de"]
    m["$roe"] = m["roe"]
    m["$rgrow"] = m["revenue_yoy"]
    mcap = cl * sh
    m["$fcfy"] = np.where(mcap > 0, m["fcf"] / mcap, np.nan)

    fac_cols = ["$pe", "$pb", "$ey", "$de", "$roe", "$rgrow", "$fcfy"]
    out = m[["datetime", "ticker"] + fac_cols].copy()
    out = out.rename(columns={"ticker": "instrument"}).set_index(["datetime", "instrument"]).sort_index()
    for c in fac_cols:
        out[c] = out[c].astype(np.float32)
    out.to_hdf(OUT, key="f", complevel=5)
    print(f"\nsaved {OUT.name}: {len(out):,} rows x {len(fac_cols)} factors", flush=True)

    # ---- ACTUAL usable coverage (the number that decides feasibility) ----
    print(f"\n=== USABLE COVERAGE (NYSE-only names with a non-null factor) ===")
    for c in fac_cols:
        s = out[c]
        n_names = s.dropna().index.get_level_values("instrument").nunique()
        print(f"    {c:8} {s.notna().mean():5.1%} of cells   {n_names:>4} names   "
              f"median {np.nanmedian(s.values):+.3f}")
    key = "$fcfy"
    per = out[key].dropna().groupby(level="instrument").size()
    feas = int((per >= 252).sum())          # >= ~1yr of daily fundamental coverage
    print(f"  names with >=1yr of {key} coverage (feasible to test): {feas} of {close.columns.size}")

    # ================= LOOK-AHEAD CHECK (same as ext11/nq7) =================
    print("\n=== LOOK-AHEAD CHECK (perturbation) ===")
    chk = m.dropna(subset=["datekey"])
    viol = int((chk["datekey"] > chk["datetime"]).sum())
    print(f"  (a) rows where source filing-date > price-date (leak): {viol} of {len(chk):,}  "
          f"{'PASS' if viol == 0 else 'FAIL'}")
    rp = fund[["ticker", "reportperiod", "datekey"]].dropna(subset=["reportperiod"]).sort_values("reportperiod")
    leak2 = pd.merge_asof(px_long.sort_values("datetime"), rp, left_on="datetime",
                          right_on="reportperiod", by="ticker", direction="backward")
    future = (leak2["datekey"] > leak2["datetime"])
    n_future = int(future.sum())
    print(f"  (b) leaky merge-on-fiscal-period would use a NOT-YET-FILED report on "
          f"{n_future:,} of {len(leak2):,} cells ({n_future/len(leak2):.1%})")
    if n_future:
        de = (leak2.loc[future, "datekey"] - leak2.loc[future, "datetime"]).dt.days
        print(f"      median {int(de.median())}d early (up to {int(de.quantile(.99))}d); "
              f"our datekey merge avoids all of them.")


if __name__ == "__main__":
    main()
