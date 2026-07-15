"""Verify the consolidated daily_pv.h5 (Sharadar single-source) didn't break anything.
Coverage, NaN rates, and cross-source spot-check vs the OLD daily_pv (yfinance+Kaggle) --
same role Sharadar played against yfinance. Differences on NYSE names are EXPECTED where the
old Kaggle adjustment was wrong (that is why we migrated); we quantify agreement, not demand
identity.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SRC = Path("git_ignore_folder/factor_implementation_source_data")


def main():
    new = pd.read_hdf(SRC / "daily_pv.h5")
    old = pd.read_hdf(SRC / "daily_pv_pre_sharadar.h5")
    nc = new["$close"].unstack("instrument").sort_index()
    oc = old["$close"].unstack("instrument").sort_index()
    print(f"NEW daily_pv: {len(new):,} rows, {nc.shape[1]} instruments, "
          f"{nc.index.min().date()}..{nc.index.max().date()}")
    print(f"OLD daily_pv: {len(old):,} rows, {oc.shape[1]} instruments, "
          f"{oc.index.min().date()}..{oc.index.max().date()}")

    # interior NaN
    span_cells = nan_cells = 0
    for t in nc.columns:
        s = nc[t]; fv, lv = s.first_valid_index(), s.last_valid_index()
        if fv is None: continue
        seg = s.loc[fv:lv]; span_cells += len(seg); nan_cells += seg.isna().sum()
    print(f"\ninterior NaN rate: {nan_cells/span_cells:.4%}")
    print(f"negative/zero close: {int((new['$close'] <= 0).sum())}")

    # cross-source vs old. CRITICAL: align per-ticker on each name's OWN overlap and
    # pct_change on that overlap. Using a global reindexed index injects spurious returns
    # at each series' start/end boundary (e.g. UBER ends 2024 in Kaggle, 2026 in Sharadar),
    # which falsely depressed the correlation in an earlier version of this check.
    both = sorted(set(nc.columns) & set(oc.columns))
    print(f"\ncommon instruments: {len(both)}")
    sp = set(pd.read_hdf(SRC / "daily_pv_sp500_backup.h5").index.get_level_values("instrument").unique())
    for grp, lbl in [(sorted(set(both) & sp), "SP500 (old=yfinance)"),
                     (sorted(set(both) - sp), "NYSE-only (old=Kaggle)")]:
        if not grp: continue
        corrs = []
        for t in grp:
            no = nc[t].dropna(); oo = oc[t].dropna()
            ov = no.index.intersection(oo.index)
            if len(ov) < 100:
                continue
            ra = no.reindex(ov).pct_change(); rb = oo.reindex(ov).pct_change()
            m = ra.notna() & rb.notna()
            if m.sum() > 100:
                corrs.append(ra[m].corr(rb[m]))
        corrs = np.array([c for c in corrs if np.isfinite(c)])
        print(f"  {lbl:26} n={len(corrs):>4}  median return corr {np.median(corrs):.4f}  "
              f">=0.99: {(corrs>=0.99).mean():.0%}  >=0.95: {(corrs>=0.95).mean():.0%}  "
              f"<0.5 (ticker collisions): {int((corrs<0.5).sum())}")

    print(f"\ninstruments dropped vs old: {len(set(oc.columns)-set(nc.columns))} "
          f"(unfiltered Kaggle penny/illiquid + names not in Sharadar)")
    print(f"instruments gained vs old:  {len(set(nc.columns)-set(oc.columns))} "
          f"(delisted SP500 members + pre-2019 history)")


if __name__ == "__main__":
    main()
