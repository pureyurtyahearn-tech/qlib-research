"""Causally merge SF1-ART fundamentals onto daily SEP prices, and prove it is look-ahead-free.

Merge rule: each daily (ticker, date) row carries the fundamental record with the latest
FILING DATE (datekey) <= date -- i.e. the most recently KNOWN value. Implemented with
pandas.merge_asof(direction='backward') on datekey. Nothing from a future filing can leak.

7 value/quality factors (the user's list, in cross-sectionally sensible form):
  $pe   = close / eps_ttm            (price/earnings)
  $pb   = close / bvps               (price/book)
  $ey   = eps_ttm / close            (earnings yield = the signed, tradeable value signal)
  $de   = debt / equity
  $roe  = return on equity (TTM)
  $rgrow= revenue TTM YoY growth
  $fcfy = fcf_ttm / (close * shares)  (free-cash-flow yield, marketcap recomputed DAILY)

LOOK-AHEAD CHECK (perturbation, same standard as eda4_lookahead): build a deliberately
LEAKY merge on reportperiod (fiscal period-end, ~40d before filing) and show (a) our causal
merge never uses a value before its datekey, (b) the leaky one would, on a large fraction of
cells, "know" a value 30-90 days early.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
START = "1999-01-01"
FUND = ["eps", "bvps", "de", "roe", "revenue", "fcf", "sharesbas"]


def load_fundamentals():
    f = pd.read_hdf(SH / "sf1_art_raw.h5")
    for c in ["datekey", "reportperiod", "calendardate"]:
        f[c] = pd.to_datetime(f[c])
    f = f[f["datekey"] >= f["reportperiod"]]                 # drop the 1 pre-period glitch
    f = f.sort_values(["ticker", "datekey"])
    # revenue TTM YoY growth: 4 TTM quarters ~ 1 year apart, within ticker
    f["revenue_yoy"] = f.groupby("ticker")["revenue"].transform(lambda s: s / s.shift(4) - 1)
    return f


def causal_merge(px_long, fund, on_col):
    """merge_asof backward: each price row gets the latest fundamental with on_col <= date."""
    cols = ["ticker", on_col, "eps", "bvps", "de", "roe", "revenue_yoy", "fcf", "sharesbas"]
    right = fund[cols].dropna(subset=[on_col]).sort_values(on_col)
    m = pd.merge_asof(px_long.sort_values("datetime"), right,
                      left_on="datetime", right_on=on_col, by="ticker", direction="backward")
    return m


def main():
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    close = px["$close"].unstack("ticker").sort_index()
    close = close.loc[close.index >= START]
    px_long = close.stack().rename("close").reset_index()
    px_long.columns = ["datetime", "ticker", "close"]
    print(f"price cells (priced (ticker,date) since {START}): {len(px_long):,}")

    fund = load_fundamentals()
    print(f"fundamentals: {len(fund):,} filings, {fund.ticker.nunique()} tickers")

    # ---- causal merge on datekey (filing date) ----
    m = causal_merge(px_long, fund, "datekey")
    # ---- factors ----
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
    out.to_hdf(SH / "fundamentals_daily.h5", key="f", complevel=5)
    print(f"\nsaved fundamentals_daily.h5: {len(out):,} rows x {len(fac_cols)} factors")
    print(f"  coverage (non-null) per factor:")
    for c in fac_cols:
        print(f"    {c:8} {out[c].notna().mean():.1%}   sample median {np.nanmedian(out[c].values):+.3f}")

    # ================= LOOK-AHEAD CHECK =================
    print("\n=== LOOK-AHEAD CHECK (perturbation) ===")
    # (a) direct: the datekey used on each row must be <= the row's date. merge_asof
    #     guarantees this; assert on the merged source key.
    chk = m.dropna(subset=["datekey"])
    viol = int((chk["datekey"] > chk["datetime"]).sum())
    print(f"  (a) rows where source filing-date > price-date (leak): {viol} of {len(chk):,}  "
          f"{'PASS' if viol == 0 else 'FAIL'}")

    # (b) leaky counterfactual: merge on reportperiod (fiscal end, pre-filing). Compare which
    #     filing each method attributes to a given (ticker,date); count cells where the leaky
    #     method uses a filing whose datekey is still in the FUTURE relative to that date.
    leak = causal_merge(px_long.assign(), fund, "reportperiod")
    # bring the filing's true datekey along for the leaky merge to test it
    rp = fund[["ticker", "reportperiod", "datekey"]].dropna(subset=["reportperiod"]).sort_values("reportperiod")
    leak2 = pd.merge_asof(px_long.sort_values("datetime"), rp, left_on="datetime",
                          right_on="reportperiod", by="ticker", direction="backward")
    future = (leak2["datekey"] > leak2["datetime"])
    n_future = int(future.sum())
    print(f"  (b) leaky merge-on-fiscal-period would use a NOT-YET-FILED report on "
          f"{n_future:,} of {len(leak2):,} cells ({n_future/len(leak2):.1%})")
    days_early = (leak2.loc[future, "datekey"] - leak2.loc[future, "datetime"]).dt.days
    if len(days_early):
        print(f"      those cells see fundamentals a median {int(days_early.median())}d "
              f"(up to {int(days_early.quantile(.99))}d) before they were filed")
    print(f"  -> our causal merge avoids all {n_future:,} of those leaks by keying on datekey.")


if __name__ == "__main__":
    main()
