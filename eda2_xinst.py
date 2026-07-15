"""EDA part 2: cross-instrument consistency — calendar alignment + correlation sanity."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

SRC = "git_ignore_folder/factor_implementation_source_data"
df = pd.read_hdf(f"{SRC}/daily_pv.h5")
sp500 = pd.read_hdf(f"{SRC}/daily_pv_sp500_backup.h5").index.get_level_values("instrument").unique()
d = df[df.index.get_level_values("instrument").isin(sp500)]
dts = d.index.get_level_values("datetime")
d = d[(dts >= "2023-01-01") & (dts <= "2026-06-16")]
close = d["$close"].unstack("instrument").sort_index()
ret = close.pct_change()

# ---- calendar alignment vs qlib day.txt ----
qcal = [l.strip() for l in open(Path.home()/".qlib"/"qlib_data"/"us_data"/"calendars"/"day.txt") if l.strip()]
qcal = pd.to_datetime([c for c in qcal if "2023-01-01" <= c <= "2026-06-16"])
pv_dates = pd.to_datetime(close.index)
print("=== 2a. calendar alignment (2023-01-01..2026-06-16) ===")
print(f"  qlib calendar trading days: {len(qcal)}")
print(f"  daily_pv trading days:      {len(pv_dates)}")
print(f"  dates in pv NOT in qlib cal: {len(set(pv_dates)-set(qcal))}")
print(f"  dates in qlib cal NOT in pv: {len(set(qcal)-set(pv_dates))}")

# per-instrument: any price on a non-calendar date? (built from cal, so should be 0)
offcal = 0
for inst in close.columns[:50]:
    s = close[inst].dropna()
    offcal += len(set(pd.to_datetime(s.index)) - set(qcal))
print(f"  (sample 50 insts) price points on non-calendar dates: {offcal}")

# ---- correlation sanity: sector peers vs unrelated ----
groups = {
    "Banks":   ["JPM","BAC","WFC","C"],
    "BigTech": ["AAPL","MSFT","NVDA","GOOGL"],
    "Energy":  ["XOM","CVX","COP"],
    "Staples": ["KO","PEP","PG","WMT"],
    "Airlines":["AAL","DAL","UAL","LUV"],
}
groups = {k:[t for t in v if t in ret.columns] for k,v in groups.items()}
print("\n=== 2b. within-sector avg return correlation (expect ~0.4-0.85) ===")
for k,v in groups.items():
    if len(v)<2: continue
    cm = ret[v].corr()
    off = cm.values[np.triu_indices(len(v),1)]
    print(f"  {k:9} {v}  mean_corr={np.nanmean(off):+.3f}  range=[{np.nanmin(off):+.3f},{np.nanmax(off):+.3f}]")

print("\n=== 2c. cross-sector correlation (Banks vs BigTech, expect lower) ===")
b, tg = groups["Banks"], groups["BigTech"]
cross = ret[b+tg].corr().loc[b, tg].values
print(f"  mean={np.nanmean(cross):+.3f}  range=[{np.nanmin(cross):+.3f},{np.nanmax(cross):+.3f}]")

print("\n=== 2d. leakage/alignment red-flags across full 505 universe ===")
# equal-weight market proxy; each stock's beta-corr to market should be mostly positive
mkt = ret.mean(axis=1)
bcorr = ret.apply(lambda s: s.corr(mkt))
print(f"  stock-vs-market corr: mean={bcorr.mean():+.3f}  min={bcorr.min():+.3f}  max={bcorr.max():+.3f}")
print(f"  stocks with market-corr<0 (suspicious): {int((bcorr<0).sum())}")
# off-diagonal exactly 1.0 => duplicate/leaked series
sample = ret[[c for c in ret.columns[:120]]].corr().values
np.fill_diagonal(sample, np.nan)
ones = np.sum(np.isclose(sample, 1.0, atol=1e-9))
print(f"  (120x120 sample) off-diagonal corr == 1.0 (duplicate series): {int(ones)}")
print(f"  (120x120 sample) off-diagonal |corr|<0.001 (broken alignment): {int(np.sum(np.abs(sample)<0.001))} of {np.sum(~np.isnan(sample))}")
