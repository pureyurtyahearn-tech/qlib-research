"""Can we actually GET PRICES for the 236 removed constituents?

A point-in-time universe is useless without price history for the names that left. yfinance
does not serve most delisted tickers (that is exactly why the NYSE penny stocks 404'd).
Sharadar's price table is SHARADAR/SEP, which is a SEPARATE product from the Core US
Fundamentals bundle (SF1/TICKERS/DAILY/ACTIONS/SP500/EVENTS) that the about-page describes.
So: probe what this key can actually read, on names we know are dead (Kodak, PG&E, Celgene).
"""
import warnings; warnings.filterwarnings("ignore")
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nasdaqdatalink as ndl

load_dotenv(".env", override=True)
ndl.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
OUT = Path("git_ignore_folder/sharadar")

DEAD = ["EKDKQ", "CELG", "RHT", "ETFC", "AGN"]   # Kodak, Celgene, Red Hat, E*Trade, Allergan


def probe(_tbl, **kw):
    try:
        df = ndl.get_table(_tbl, **kw)
        return True, df
    except Exception as e:
        return False, str(e).split("\n")[0][:110]


def main():
    print("=== which Sharadar tables can this key read? ===")
    for tbl, kw in [("SHARADAR/SP500", dict(action="current")),
                    ("SHARADAR/TICKERS", dict(table="SEP", ticker="AAPL")),
                    ("SHARADAR/SEP", dict(ticker="AAPL", date={"gte": "2024-01-02"})),
                    ("SHARADAR/SFP", dict(ticker="SPY", date={"gte": "2024-01-02"}))]:
        ok, r = probe(tbl, **kw)
        n = len(r) if ok else 0
        print(f"  {tbl:20} {'OK' if ok else 'DENIED'}   {f'{n} rows' if ok else r}")

    ok, _ = probe("SHARADAR/SEP", ticker="AAPL", date={"gte": "2024-01-02"})
    if not ok:
        print("\n>>> SHARADAR/SEP (equity prices) is NOT accessible on this subscription.")
        print(">>> The PIT universe is correct but UNUSABLE until we can source prices for")
        print(">>> the delisted names. Options: add the Sharadar Equity Prices (SEP) product,")
        print(">>> or the 'Core US Equities Bundle' which includes it.")
        return

    print("\n=== SEP is available: can it serve DELISTED names? ===")
    for t in DEAD:
        ok, df = probe("SHARADAR/SEP", ticker=t)
        if ok and len(df):
            d = pd.to_datetime(df["date"])
            print(f"  {t:6} {len(df):>6,} rows   {d.min().date()} .. {d.max().date()}")
        else:
            print(f"  {t:6} {'no rows' if ok else df}")


if __name__ == "__main__":
    main()
