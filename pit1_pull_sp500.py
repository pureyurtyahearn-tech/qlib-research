"""Pull SHARADAR/SP500 (point-in-time index membership changes) and inspect its shape.
No assumptions about the schema -- look before building on it."""
import warnings; warnings.filterwarnings("ignore")
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import nasdaqdatalink as ndl

load_dotenv(".env", override=True)
ndl.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
OUT = Path("git_ignore_folder/sharadar"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = ndl.get_table("SHARADAR/SP500", paginate=True)
    df.to_csv(OUT / "sp500_raw.csv", index=False)
    print(f"rows={len(df):,}  cols={list(df.columns)}")
    print(f"\naction value counts:\n{df['action'].value_counts().to_string()}")
    d = pd.to_datetime(df["date"])
    print(f"\ndate range: {d.min().date()} .. {d.max().date()}")
    print(f"unique tickers: {df['ticker'].nunique():,}")

    print("\n--- sample rows per action ---")
    for a in df["action"].unique():
        s = df[df["action"] == a].head(3)
        print(f"\n[{a}]")
        print(s.to_string(index=False))

    # how many add/remove events per year?
    ev = df[df["action"].isin(["added", "removed"])].copy()
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    print("\n--- added/removed events per year ---")
    print(ev.pivot_table(index="year", columns="action", values="ticker",
                         aggfunc="count").fillna(0).astype(int).to_string())
    print(f"\nsaved -> {OUT/'sp500_raw.csv'}")


if __name__ == "__main__":
    main()
