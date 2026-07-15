"""Rebuild the primary daily_pv.h5 factor sandbox from Sharadar SEP as the SINGLE source
of truth (retires yfinance+Kaggle stitching and fix_and_build_nyse.py's manual adjustment).

Universe = SP500 ever-members (full history) + NYSE-only broad names, all from Sharadar.
Output matches the existing schema exactly: index (datetime, instrument);
columns [$open,$high,$low,$close,$volume,$factor]; $factor=1.0 (prices pre-adjusted);
float32 OHLC, float64 volume/factor; key='/data'. Old file backed up first.
"""
import warnings; warnings.filterwarnings("ignore")
import shutil
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")
OUT = SRC / "daily_pv.h5"
BAK = SRC / "daily_pv_pre_sharadar.h5"


def main():
    sp = pd.read_hdf(SH / "sep_panel_full.h5")
    parts = [sp]
    if (SH / "sep_nyse_panel.h5").exists():
        parts.append(pd.read_hdf(SH / "sep_nyse_panel.h5"))
        print("including NYSE-only Sharadar panel")
    else:
        print("WARNING: sep_nyse_panel.h5 missing -- SP500-only consolidation")
    comb = pd.concat(parts)
    # panels are indexed (date, ticker); daily_pv wants (datetime, instrument)
    comb.index = comb.index.set_names(["datetime", "instrument"])
    comb = comb[~comb.index.duplicated(keep="first")].sort_index()

    out = pd.DataFrame(index=comb.index)
    for c in ["$open", "$high", "$low", "$close"]:
        out[c] = comb[c].astype(np.float32)
    out["$volume"] = comb["$volume"].astype(np.float64)
    out["$factor"] = np.float64(1.0)
    out = out[["$open", "$high", "$low", "$close", "$volume", "$factor"]]

    n_inst = out.index.get_level_values("instrument").nunique()
    d = out.index.get_level_values("datetime")
    print(f"consolidated: {len(out):,} rows, {n_inst} instruments, {d.min().date()}..{d.max().date()}")

    if OUT.exists() and not BAK.exists():
        shutil.copy(OUT, BAK)
        print(f"backed up old daily_pv.h5 -> {BAK.name}")
    out.to_hdf(OUT, key="data", complevel=5, format="table" if False else "fixed")
    print(f"wrote {OUT}")

    # sanity vs old
    old = pd.read_hdf(BAK) if BAK.exists() else None
    if old is not None:
        oi = old.index.get_level_values("instrument").nunique()
        print(f"  old: {len(old):,} rows, {oi} instruments; new adds full pre-2019 history + PIT SP500")


if __name__ == "__main__":
    main()
