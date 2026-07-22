"""Lightweight paper-trading log for the NYSE daily signal (nyse_daily_signal.py).

Each run:
  1. Reads the LATEST nyse_signal_YYYYMMDD.csv (its date becomes this log row's "date",
     and its top-10 BUY/SELL become this row's top10_buy/top10_sell -- "what we'd be
     holding as of this date").
  2. Reads the PREVIOUS signal CSV (the one immediately before it, by filename date), if
     one exists. If there isn't one yet (first-ever run), nothing can be scored yet --
     prints a message and exits without writing a row.
  3. Scores the PREVIOUS file's BUY/SELL lists (all names tagged BUY/SELL there, not just
     its top 10) by comparing each ticker's current_price recorded in that CSV against its
     price NOW, read live from the qlib store via D.features (not re-read from any CSV --
     the store is the source of truth for "current").
  4. Appends one row to git_ignore_folder/sharadar/nyse_paper_log.csv and prints a summary.

Return convention: "1w" is nominal (whatever the actual gap between the two signal files
is -- printed explicitly, since signals aren't guaranteed to run exactly weekly).
"""
import warnings; warnings.filterwarnings("ignore")
import re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parent
STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
LOG_PATH = Path("git_ignore_folder/sharadar/nyse_paper_log.csv")
N_TOP = 10


def find_signal_files():
    files = []
    for p in ROOT.glob("nyse_signal_*.csv"):
        m = re.match(r"nyse_signal_(\d{8})\.csv$", p.name)
        if m:
            files.append((pd.Timestamp(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return files


def main():
    files = find_signal_files()
    if not files:
        print("FAIL: no nyse_signal_*.csv files found. Run nyse_daily_signal.py first.")
        return
    latest_date, latest_path = files[-1]
    print(f"latest signal: {latest_path.name} ({latest_date.date()})")

    if len(files) < 2:
        print("No previous signal file exists yet -- nothing to score. "
              "This is expected on the first run; a row will be logged once a second "
              "signal file exists to compare against.")
        return
    prev_date, prev_path = files[-2]
    print(f"previous signal: {prev_path.name} ({prev_date.date()})")
    gap_days = (latest_date - prev_date).days
    print(f"gap between signals: {gap_days} days (nominal '1w' -- actual gap may differ)")

    latest_df = pd.read_csv(latest_path)
    prev_df = pd.read_csv(prev_path)

    top10_buy = latest_df[latest_df.signal == "BUY"].sort_values("rank").head(N_TOP)["ticker"].tolist()
    top10_sell = latest_df[latest_df.signal == "SELL"].sort_values("rank", ascending=False).head(N_TOP)["ticker"].tolist()

    prev_buy = prev_df[prev_df.signal == "BUY"][["ticker", "current_price"]].set_index("ticker")["current_price"]
    prev_sell = prev_df[prev_df.signal == "SELL"][["ticker", "current_price"]].set_index("ticker")["current_price"]
    prev_tickers = list(set(prev_buy.index) | set(prev_sell.index))

    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(STORE), region="us", kernels=1)
    now_px = D.features(prev_tickers, ["$close"], start_time=str(latest_date.date()), end_time=str(latest_date.date()))
    now_px.columns = ["now_price"]
    now_px.index = now_px.index.set_names(["instrument", "datetime"])
    now_price = now_px.reset_index().set_index("instrument")["now_price"]

    def realized_returns(prev_prices):
        matched = prev_prices.index.intersection(now_price.index)
        missing = len(prev_prices) - len(matched)
        rets = (now_price.loc[matched] / prev_prices.loc[matched] - 1)
        return rets, missing

    buy_rets, buy_missing = realized_returns(prev_buy)
    sell_rets, sell_missing = realized_returns(prev_sell)
    if buy_missing or sell_missing:
        print(f"WARNING: {buy_missing} prior BUY names and {sell_missing} prior SELL names "
              f"had no current price in the store (delisted/renamed/no data) -- excluded from the mean.")

    mean_buy_return_1w = float(buy_rets.mean()) if len(buy_rets) else np.nan
    mean_sell_return_1w = float(sell_rets.mean()) if len(sell_rets) else np.nan
    long_short_spread = mean_buy_return_1w - mean_sell_return_1w

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
    else:
        log = pd.DataFrame(columns=["date", "top10_buy", "top10_sell", "mean_buy_return_1w",
                                     "mean_sell_return_1w", "long_short_spread",
                                     "running_mean_buy", "running_mean_sell", "running_spread"])

    date_str = str(latest_date.date())
    if (log["date"] == date_str).any():
        print(f"NOTE: a row for {date_str} already exists in {LOG_PATH} -- replacing it (not duplicating).")
        log = log[log["date"] != date_str]

    new_row = {
        "date": date_str,
        "top10_buy": "|".join(top10_buy),
        "top10_sell": "|".join(top10_sell),
        "mean_buy_return_1w": mean_buy_return_1w,
        "mean_sell_return_1w": mean_sell_return_1w,
        "long_short_spread": long_short_spread,
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True).sort_values("date").reset_index(drop=True)

    log["running_mean_buy"] = log["mean_buy_return_1w"].expanding().mean()
    log["running_mean_sell"] = log["mean_sell_return_1w"].expanding().mean()
    log["running_spread"] = log["long_short_spread"].expanding().mean()

    log.to_csv(LOG_PATH, index=False)

    row = log[log["date"] == date_str].iloc[0]
    print(f"\n{'='*60}")
    print(f"NYSE PAPER TRADING LOG -- {date_str}")
    print(f"{'='*60}")
    print(f"scoring previous week's picks ({prev_date.date()}, {gap_days}d ago):")
    print(f"  BUY  ({len(buy_rets)}/{len(prev_buy)} priced): mean return {mean_buy_return_1w:+.2%}")
    print(f"  SELL ({len(sell_rets)}/{len(prev_sell)} priced): mean return {mean_sell_return_1w:+.2%}")
    print(f"  long-short spread (BUY - SELL): {long_short_spread:+.2%}")
    print(f"\nrunning averages across {len(log)} logged week(s):")
    print(f"  running mean BUY return   : {row['running_mean_buy']:+.2%}")
    print(f"  running mean SELL return  : {row['running_mean_sell']:+.2%}")
    print(f"  running long-short spread : {row['running_spread']:+.2%}")
    print(f"\nthis week's ({date_str}) top 10 BUY : {', '.join(top10_buy)}")
    print(f"this week's ({date_str}) top 10 SELL: {', '.join(top10_sell)}")
    print(f"\nsaved {LOG_PATH}")


if __name__ == "__main__":
    main()
