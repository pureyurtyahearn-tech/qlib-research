"""Re-derive the NYSE liquidity filter from the ACTUAL Sharadar source (sep_nyse_panel.h5),
using the same $1M median daily dollar-volume store gate as nq2_quality_liquidity.py used for
NASDAQ. Fixes the provenance mismatch preflight_check.py caught: the original NYSE ticker
selection (nx3_select.py) was computed on the old Kaggle/yfinance daily_pv.h5, not on the
Sharadar-native panel that actually got ingested into the store.

Updates:
  - writes git_ignore_folder/sharadar/nyse_liquid_universe.csv (nasdaq_liquid_universe.csv
    equivalent)
  - rewrites us_data_pit_full/instruments/nyse.txt to the newly-filtered set
  - removes exclusively-NYSE tickers that fail the gate from instruments/all.txt (never
    touches a ticker that's also present in nasdaq.txt or sp500pit.txt)
  - backs up both instrument files before writing (*.pre_liquidity_refilter)
  - does NOT delete any feature/ bin directories -- excluded tickers' bins are simply no
    longer listed, which is enough to remove them from D.instruments(market=...); leaving
    the bins in place is the safer, reversible choice.
"""
import warnings; warnings.filterwarnings("ignore")
import shutil
from pathlib import Path
import pandas as pd

SH = Path("git_ignore_folder/sharadar")
STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
STORE_GATE = 1e6


def read_tickers(path):
    return {l.split("\t")[0].strip().upper() for l in path.read_text().splitlines() if l.strip()}


def main():
    panel = pd.read_hdf(SH / "sep_nyse_panel.h5").reset_index()
    panel["dv"] = panel["$close"] * panel["$volume"]
    med_dv = panel.groupby("ticker")["dv"].median()
    keep = set(med_dv[med_dv >= STORE_GATE].index)
    total = panel.ticker.nunique()
    print(f"sep_nyse_panel.h5: {total} tickers")
    print(f"pass ${STORE_GATE/1e6:.0f}M median-$vol gate: {len(keep)}   fail: {total - len(keep)}")

    out_csv = SH / "nyse_liquid_universe.csv"
    pd.Series(sorted(keep)).to_csv(out_csv, index=False, header=["ticker"])
    print(f"saved {out_csv}")

    # ---- update instruments/nyse.txt ----
    nyse_path = STORE / "instruments" / "nyse.txt"
    nyse_bak = STORE / "instruments" / "nyse.txt.pre_liquidity_refilter"
    if not nyse_bak.exists():
        shutil.copy(nyse_path, nyse_bak)
        print(f"backed up nyse.txt -> {nyse_bak.name}")

    nyse_lines = [l.rstrip("\n") for l in nyse_bak.read_text().splitlines() if l.strip()]
    before_tickers = {l.split("\t")[0].strip().upper() for l in nyse_lines}
    kept_lines = [l for l in nyse_lines if l.split("\t")[0].strip().upper() in keep]
    dropped = before_tickers - keep
    nyse_path.write_text("\n".join(kept_lines) + "\n")
    print(f"nyse.txt: {len(nyse_lines)} -> {len(kept_lines)} tickers "
          f"({len(dropped)} dropped for failing the corrected liquidity gate)")

    # ---- update instruments/all.txt: remove exclusively-NYSE tickers that were dropped ----
    all_path = STORE / "instruments" / "all.txt"
    all_bak = STORE / "instruments" / "all.txt.pre_liquidity_refilter"
    if not all_bak.exists():
        shutil.copy(all_path, all_bak)
        print(f"backed up all.txt -> {all_bak.name}")

    nasdaq_tickers = read_tickers(STORE / "instruments" / "nasdaq.txt")
    sp500_tickers = read_tickers(STORE / "instruments" / "sp500pit.txt")
    protected = nasdaq_tickers | sp500_tickers
    remove_from_all = {t for t in dropped if t not in protected}
    kept_in_other_lists = dropped - remove_from_all
    if kept_in_other_lists:
        print(f"  {len(kept_in_other_lists)} dropped-from-nyse tickers are still needed by "
              f"nasdaq.txt/sp500pit.txt -- left in all.txt: {sorted(kept_in_other_lists)[:10]}")

    all_lines = [l.rstrip("\n") for l in all_bak.read_text().splitlines() if l.strip()]
    all_kept = [l for l in all_lines if l.split("\t")[0].strip().upper() not in remove_from_all]
    all_path.write_text("\n".join(all_kept) + "\n")
    print(f"all.txt: {len(all_lines)} -> {len(all_kept)} instruments "
          f"({len(remove_from_all)} exclusively-NYSE tickers removed)")

    print(f"\ndone. Feature bin directories for dropped tickers were NOT deleted "
          f"(reversible -- they're just no longer listed).")


if __name__ == "__main__":
    main()
