"""Build a NYSE-specific RD-Agent factor sandbox, mechanically identical to
ext13_build_sandbox.py (the SP500-PIT sandbox) -- same schema, same dedicated-folder
pattern, same debug subset. Fixes the universe-mismatch bug found by hand on 2026-07-17:
FACTOR_CoSTEER_DATA_FOLDER was pointing at the SP500-ish sandbox (factor_fundamentals_data,
2327 instruments incl. AAPL/AAMRQ/ABT) with ZERO ticker overlap against the actual NYSE
training universe, so every custom factor computed by CoSTEER came out NaN for every NYSE
row and got Fillna'd to a constant -- explaining why NYSE factors never moved a prediction
in either NYSE run so far.

sandbox daily_pv.h5 = NYSE-only names (from sep_nyse_panel.h5, the same Sharadar-native
panel actually loaded into the store), OHLCV + $factor + 7 fundamental columns, 1999-2026.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SH = Path("git_ignore_folder/sharadar")
DATA = Path("git_ignore_folder/factor_fundamentals_data_nyse")
DEBUG = Path("git_ignore_folder/factor_fundamentals_data_nyse_debug")
START = "1999-01-01"
FCOLS = ["$pe", "$pb", "$ey", "$de", "$roe", "$rgrow", "$fcfy"]

README = """# How to read files
```python
import pandas as pd
df = pd.read_hdf("daily_pv.h5", key="data")   # key is always "data"
```
MultiIndex (datetime, instrument). Universe: NYSE-only names (Sharadar-native, liquidity-
filtered, median $vol >= $1M), 1999-2026.

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
    px = pd.read_hdf(SH / "sep_nyse_panel.h5")
    fund = pd.read_hdf(SH / "fundamentals_nyse_daily.h5")     # (datetime, instrument) x 7
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
