"""Build the fundamentals-augmented RD-Agent sandbox in a DEDICATED folder (keeps the
consolidated daily_pv.h5 and script backups untouched, and lighter on RAM).

sandbox daily_pv.h5 = SP500 PIT names only, OHLCV + $factor + 7 fundamental columns,
1999-2026. A small debug subset (100 names) carries the SAME schema so RD-Agent's
data-description step (which reads data_folder_debug) shows the fundamental columns to the
LLM without loading the full panel or the backup files.
"""
import warnings; warnings.filterwarnings("ignore")
import shutil
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
DATA = Path("git_ignore_folder/factor_fundamentals_data")
DEBUG = Path("git_ignore_folder/factor_fundamentals_data_debug")
START = "1999-01-01"
FCOLS = ["$pe", "$pb", "$ey", "$de", "$roe", "$rgrow", "$fcfy"]

README = """# How to read files
```python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")   # key is always "data"
```
MultiIndex (datetime, instrument). Universe: point-in-time S&P 500 members, 1999-2026.

# Columns

## Daily price/volume (adjusted; split+dividend+spinoff via Sharadar closeadj)
$open, $high, $low, $close, $volume, $factor(=1.0, prices already adjusted)

## Fundamentals (Sharadar SF1, As-Reported TTM, point-in-time by SEC FILING date)
These update only when a company FILES (quarterly); the value is carried forward daily
until the next filing. They change slowly -> low-turnover factors are possible.
$pe    : price / trailing-12m EPS      (low = cheap; high = expensive)
$pb    : price / book value per share  (low = cheap)
$ey    : trailing-12m EPS / price       (earnings yield; high = cheap; = 1/$pe, signed)
$de    : total debt / equity            (leverage)
$roe   : return on equity, TTM          (quality; high = profitable)
$rgrow : revenue TTM year-over-year growth
$fcfy  : trailing-12m free cash flow / market cap  (free-cash-flow yield; high = cheap)

Note: fundamentals are NaN before a company's first filing and for a few names lacking
SF1 coverage. Handle NaN in cross-sectional operations.
"""


def main():
    for d in (DATA, DEBUG):
        d.mkdir(parents=True, exist_ok=True)
    px = pd.read_hdf(SH / "sep_panel_full.h5")
    fund = pd.read_hdf(SH / "fundamentals_daily.h5")     # (datetime, instrument) x 7
    # price frame -> long, restrict window
    px = px[px.index.get_level_values("date") >= START]
    px.index = px.index.set_names(["datetime", "instrument"])
    out = pd.DataFrame(index=px.index)
    for c in ["$open", "$high", "$low", "$close"]:
        out[c] = px[c].astype(np.float32)
    out["$volume"] = px["$volume"].astype(np.float64)
    out["$factor"] = np.float64(1.0)
    # join fundamentals (align on the shared MultiIndex)
    f = fund.reindex(out.index)
    for c in FCOLS:
        out[c] = f[c].astype(np.float32)
    out = out.sort_index()
    n_inst = out.index.get_level_values("instrument").nunique()
    print(f"sandbox: {len(out):,} rows, {n_inst} instruments, "
          f"{out.index.get_level_values('datetime').min().date()}.."
          f"{out.index.get_level_values('datetime').max().date()}")
    print("fundamental coverage in sandbox:")
    for c in FCOLS:
        print(f"  {c:8} {out[c].notna().mean():.1%}")
    out.to_hdf(DATA / "daily_pv.h5", key="data", complevel=5)
    (DATA / "README.md").write_text(README)
    print(f"wrote {DATA/'daily_pv.h5'} + README")

    # debug subset: 100 instruments, same columns/schema
    insts = out.index.get_level_values("instrument").unique()[:100]
    dbg = out[out.index.get_level_values("instrument").isin(insts)]
    dbg.to_hdf(DEBUG / "daily_pv.h5", key="data", complevel=5)
    (DEBUG / "README.md").write_text(README)
    print(f"wrote debug ({dbg.index.get_level_values('instrument').nunique()} insts) + README")


if __name__ == "__main__":
    main()
