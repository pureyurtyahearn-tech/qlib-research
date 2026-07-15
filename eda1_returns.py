"""EDA part 1: statistical sanity of raw returns from daily_pv.h5 (factor input source),
SP500 universe, 2023-01-01..2026-06-16. Pure pandas."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats as sstats

SRC = "git_ignore_folder/factor_implementation_source_data"
df = pd.read_hdf(f"{SRC}/daily_pv.h5")
sp500 = pd.read_hdf(f"{SRC}/daily_pv_sp500_backup.h5").index.get_level_values("instrument").unique()
print(f"SP500 universe: {len(sp500)} instruments")

# filter to SP500 + window
m = df.index.get_level_values("instrument").isin(sp500)
d = df[m]
dts = d.index.get_level_values("datetime")
d = d[(dts >= "2023-01-01") & (dts <= "2026-06-16")]
print(f"rows in window: {len(d)}  | instruments: {d.index.get_level_values('instrument').nunique()}"
      f"  | dates: {d.index.get_level_values('datetime').nunique()}")

# ---- daily returns per instrument ----
close = d["$close"].unstack("instrument").sort_index()
ret = close.pct_change()
r = ret.stack().dropna().values
print("\n=== 1a. daily return distribution (pooled, all names/days) ===")
print(f"  n={len(r):,}")
print(f"  mean   = {r.mean():+.6f}   (~{r.mean()*252:+.3f}/yr)")
print(f"  std    = {r.std():.6f}    (~{r.std()*np.sqrt(252):.3f}/yr)")
print(f"  skew   = {sstats.skew(r):+.3f}")
print(f"  kurt   = {sstats.kurtosis(r):+.3f}  (excess; equities typically +3 to +15)")
print(f"  min    = {r.min():+.4f}   max = {r.max():+.4f}")
for p in [0.01,0.1,1,5,50,95,99,99.9,99.99]:
    print(f"    p{p:<5} = {np.percentile(r,p):+.4f}")

# ---- extreme moves (possible bad ticks) ----
ext = np.abs(r) > 0.5
print(f"\n=== 1b. extreme single-day moves |ret|>50% : {int(ext.sum())} "
      f"( >30%: {int((np.abs(r)>0.3).sum())}, >20%: {int((np.abs(r)>0.2).sum())} ) ===")

# ---- zero-return days ----
z = (r == 0.0)
print(f"\n=== 1c. exact zero-return days: {int(z.sum())} ({100*z.mean():.3f}% of obs) ===")

# ---- value clustering: are returns suspiciously rounded / repeated? ----
vc = pd.Series(r).round(6).value_counts().head(6)
print("\n=== 1d. most frequent exact return values (clustering check) ===")
for val, cnt in vc.items():
    print(f"    {val:+.6f} : {cnt}  ({100*cnt/len(r):.3f}%)")

# ---- duplicates ----
print("\n=== 1e. duplicate checks ===")
print(f"  duplicate (datetime,instrument) index entries: {int(d.index.duplicated().sum())}")
print(f"  fully-duplicate rows: {int(d.duplicated().sum())}")

# ---- flat-price stretches (stale/ffill detection) ----
print("\n=== 1f. flat $close stretches (consecutive identical closes; excludes NaN) ===")
def max_run(s):
    s = s.dropna().values
    if len(s) < 2: return 0
    same = s[1:] == s[:-1]
    best = run = 0
    for x in same:
        run = run+1 if x else 0
        best = max(best, run)
    return best
runs = close.apply(max_run)
print(f"  instruments with a flat run >=5 days: {int((runs>=5).sum())}")
print(f"  instruments with a flat run >=10 days: {int((runs>=10).sum())}")
print("  worst offenders (instrument: longest flat run of identical close):")
for inst, v in runs.sort_values(ascending=False).head(8).items():
    print(f"    {inst:8} {int(v)} days")
