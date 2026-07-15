"""Verify the migrated full-history dataset didn't break anything.
1. Coverage + NaN rates on the new panel.
2. Cross-source spot-check vs the OLD us_data_pit (2010-2026 SEP) -- must be identical where
   they overlap (same source, extended range -> the overlap must match exactly).
3. Cross-source vs yfinance (the reference we keep) on a sample, same as pit7 -- returns
   must still agree (median corr ~1.0), confirming the migration preserved adjustment quality.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    new = pd.read_hdf(SH / "sep_panel_full.h5")
    newc = new["$close"].unstack("ticker").sort_index()
    print(f"NEW panel: {newc.shape[0]} days x {newc.shape[1]} tickers  "
          f"{newc.index.min().date()}..{newc.index.max().date()}")

    # 1. NaN within active spans
    active = newc.notna()
    span_cells = 0; nan_cells = 0
    for t in newc.columns:
        s = newc[t]
        fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None:
            continue
        seg = s.loc[fv:lv]
        span_cells += len(seg); nan_cells += seg.isna().sum()
    print(f"  interior NaN rate (gaps inside a name's active span): {nan_cells/span_cells:.4%}")

    # 2. vs OLD SEP panel (2010-2026) -- exact where overlap
    old = pd.read_hdf(SH / "sep_panel.h5")["$close"].unstack("ticker").sort_index()
    both_t = sorted(set(newc.columns) & set(old.columns))
    idx = newc.index.intersection(old.index)
    a = newc.loc[idx, both_t]; b = old.loc[idx, both_t]
    diff = (a - b).abs()
    maxd = np.nanmax(diff.values)
    print(f"\n  vs old us_data_pit SEP panel ({len(both_t)} common tickers, {len(idx)} dates):")
    print(f"    max |new-old| close: {maxd:.6f}   (expect ~0: same source)")
    mism = int((diff > 0.01).sum().sum())
    print(f"    cells differing >$0.01: {mism}")

    # 3. vs yfinance (reference) -- returns still agree
    yf = pd.read_hdf(SRC / "daily_pv_sp500_backup.h5")["$close"].unstack("instrument").sort_index()
    both_y = sorted(set(newc.columns) & set(yf.columns))
    idx2 = newc.index.intersection(yf.index)
    ra = newc.loc[idx2, both_y].pct_change()
    rb = yf.loc[idx2, both_y].pct_change()
    corrs = []
    for t in both_y:
        m = ra[t].notna() & rb[t].notna()
        if m.sum() > 250:
            corrs.append(ra[t][m].corr(rb[t][m]))
    corrs = np.array(corrs)
    print(f"\n  vs yfinance reference ({len(corrs)} tickers): median return corr {np.median(corrs):.5f}, "
          f">=0.99: {(corrs>=0.99).mean():.1%}")

    # coverage by year already reported in ext4; here just the pre-2010 extension gained names
    print(f"\n  tickers ONLY in the new full panel (pre-2010 members we didn't have): "
          f"{len(set(newc.columns) - set(old.columns))}")


if __name__ == "__main__":
    main()
