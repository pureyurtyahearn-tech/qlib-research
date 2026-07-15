"""Reconcile PIT membership against our universe on a STABLE ID, not a ticker string.

Sharadar maps a company's whole index history onto its CURRENT ticker (Bank of New York's
2010 membership is filed under BNY, though it traded as BK). Comparing raw ticker strings
therefore reports renames as 'missing', which is wrong. SHARADAR/TICKERS gives permaticker
(stable company ID) + relatedtickers (former symbols), so we match on those instead.

The genuinely-missing set then splits into TWO different problems, which must not be
conflated:
  A. SURVIVORSHIP HOLE  -- was in the index, got REMOVED, and we have no prices. This is
     the bias today's investigation identified.
  B. STALE-LIST HOLE    -- is in the index TODAY but was never in our ticker list (our
     SP500 list is evidently an older snapshot). A coverage gap, not survivorship.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nasdaqdatalink as ndl

load_dotenv(".env", override=True)
ndl.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
OUT = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    tp = OUT / "tickers.csv"
    if not tp.exists():
        tk = ndl.get_table("SHARADAR/TICKERS", table="SEP", paginate=True)
        tk.to_csv(tp, index=False)
    tk = pd.read_csv(tp, low_memory=False)
    print(f"SHARADAR/TICKERS (SEP): {len(tk):,} rows, cols={list(tk.columns)[:12]}...")

    mat = pd.read_hdf(OUT / "sp500_pit_membership.h5")
    raw = pd.read_csv(OUT / "sp500_raw.csv", parse_dates=["date"])
    ours = sorted(set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")
                      .index.get_level_values("instrument").unique()))

    # ticker -> permaticker, plus every alias (relatedtickers) -> permaticker
    alias = {}
    for _, r in tk.iterrows():
        pm = r["permaticker"]
        alias[str(r["ticker"]).upper()] = pm
        rt = r.get("relatedtickers")
        if isinstance(rt, str):
            for a in rt.replace(",", " ").split():
                alias.setdefault(a.strip().upper(), pm)
    print(f"  ticker/alias -> permaticker map: {len(alias):,} symbols")

    ever = sorted(mat.columns[mat.any(axis=0)])
    today = set(mat.columns[mat.iloc[-1]])
    pm_ever = {t: alias.get(t.upper()) for t in ever}
    pm_ours = {t: alias.get(t.upper()) for t in ours}
    ours_pm = {v for v in pm_ours.values() if v is not None}

    unres_ever = [t for t, v in pm_ever.items() if v is None]
    unres_ours = [t for t, v in pm_ours.items() if v is None]
    print(f"  unresolved: {len(unres_ever)} of {len(ever)} PIT tickers, "
          f"{len(unres_ours)} of {len(ours)} of ours {unres_ours[:6]}")

    # genuinely missing = in PIT, and its permaticker is NOT covered by any of our tickers
    missing = [t for t in ever if (pm_ever[t] is None and t not in set(ours))
               or (pm_ever[t] is not None and pm_ever[t] not in ours_pm)]
    renamed = [t for t in ever if t not in set(ours) and pm_ever[t] in ours_pm]
    print(f"\n=== reconciliation on permaticker ===")
    print(f"  ever in index 2010-2026        : {len(ever)}")
    print(f"  naive ticker-string 'missing'  : {len([t for t in ever if t not in set(ours)])}")
    print(f"  of those, actually RENAMES we DO have: {len(renamed)}   e.g. "
          f"{[f'{t}' for t in renamed[:8]]}")
    print(f"  >>> GENUINELY MISSING          : {len(missing)}")

    # split A vs B
    A = [t for t in missing if t not in today]      # removed from index, we lack prices
    B = [t for t in missing if t in today]          # still in index, our list never had it
    print(f"\n  A. SURVIVORSHIP HOLE (removed from index, we have no data): {len(A)}")
    print(f"  B. STALE-LIST HOLE  (in the index TODAY, absent from our list): {len(B)}")
    print(f"     B examples: {sorted(B)[:12]}")

    tot = int(mat.values.sum())
    print(f"\n=== how much index-time are we blind to? ===")
    for lbl, s in [("A survivorship", A), ("B stale-list", B), ("A+B combined", missing)]:
        d = int(mat[s].values.sum())
        print(f"  {lbl:16} {d:>9,} of {tot:,} membership-days  ({d/tot:>5.1%})")

    print(f"\n  avg # index members per day we cannot see (survivorship hole A only):")
    for y in range(2010, 2027):
        m = mat[mat.index.year == y]
        if len(m):
            print(f"    {y}: {m[A].sum(axis=1).mean():5.0f}", end="" if y % 6 else "\n")
    print()

    # who are they, and why did they go?
    rem = raw[(raw.action == "removed") & (raw.date >= "2010-01-01")]
    rem = rem.sort_values("date").drop_duplicates("ticker", keep="last").set_index("ticker")
    rows = []
    dur = mat[A].sum(axis=0)
    for t in A:
        r = rem.loc[t] if t in rem.index else None
        rows.append(dict(ticker=t,
                         name=str(r["name"])[:30] if r is not None else "?",
                         left=pd.Timestamp(r["date"]).date() if r is not None else None,
                         acquirer_or_replacement=str(r["contraname"])[:28] if r is not None else "",
                         yrs_in_index=round(dur[t] / 252, 1)))
    dfA = pd.DataFrame(rows).sort_values("yrs_in_index", ascending=False)
    print(f"=== the survivorship hole: longest-tenured index members we are MISSING ===")
    print(dfA.head(25).to_string(index=False))
    dfA.to_csv(OUT / "survivorship_hole.csv", index=False)

    # failures vs acquisitions -- this is what determines the DIRECTION of the bias
    note = raw[(raw.action == "removed") & (raw.date >= "2010-01-01")][["ticker", "note"]]
    print(f"\n  (saved full list -> {OUT/'survivorship_hole.csv'})")
    print(f"\n  departures by year (survivorship hole A):")
    yr = pd.to_datetime(dfA["left"].dropna()).dt.year.value_counts().sort_index()
    print("   " + "  ".join(f"{y}:{n}" for y, n in yr.items()))


if __name__ == "__main__":
    main()
