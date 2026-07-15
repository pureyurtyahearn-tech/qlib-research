"""Cross-source agreement: SP500(yfinance-lineage) vs Kaggle-NYSE, for tickers in BOTH.
Compares adjusted closes on their overlapping dates: return correlation, distribution of
daily-return differences, systematic bias, and price-ratio drift (catches a missed
split/dividend adjustment in one source)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats as sstats

SP = "git_ignore_folder/factor_implementation_source_data/daily_pv_sp500_backup.h5"
NY = "git_ignore_folder/nyse_adjusted_daily_pv.h5"

sp = pd.read_hdf(SP); ny = pd.read_hdf(NY)
spI = set(sp.index.get_level_values("instrument").unique())
nyI = set(ny.index.get_level_values("instrument").unique())
common = sorted(spI & nyI)
print(f"SP500 instruments={len(spI)}  Kaggle-NYSE instruments={len(nyI)}  IN BOTH={len(common)}")

spc = sp["$close"].unstack("instrument")
nyc = ny["$close"].unstack("instrument")
dates = spc.index.intersection(nyc.index)
spc = spc.loc[dates, common]; nyc = nyc.loc[dates, common]
print(f"overlapping dates: {len(dates)}  ({dates.min().date()} -> {dates.max().date()})")

spr = spc.pct_change(); nyr = nyc.pct_change()
diff = (spr - nyr)

# ---- per-ticker stats ----
rows=[]
for t in common:
    a, b = spr[t], nyr[t]
    m = a.notna() & b.notna()
    if m.sum() < 50: continue
    a, b = a[m], b[m]
    ratio = (spc[t] / nyc[t]).dropna()
    drift = (ratio.iloc[-1]/ratio.iloc[0] - 1) if len(ratio) > 1 and ratio.iloc[0] != 0 else np.nan
    rows.append(dict(t=t, n=int(m.sum()), corr=a.corr(b),
                     mdiff=(a-b).mean(), sdiff=(a-b).std(), maxdiff=(a-b).abs().max(),
                     ratio_cv=ratio.std()/abs(ratio.mean()) if ratio.mean() else np.nan,
                     ratio_drift=drift))
r = pd.DataFrame(rows)

print("\n=== per-ticker daily-return correlation between the two sources ===")
print(f"  mean={r['corr'].mean():.6f}  median={r['corr'].median():.6f}  min={r['corr'].min():.6f}")
for th in [0.9999, 0.999, 0.99, 0.95]:
    print(f"  tickers with corr >= {th}: {int((r['corr']>=th).sum())}/{len(r)}")

d = diff.stack().dropna().values
print(f"\n=== pooled daily-return DIFFERENCE (SP500 - NYSE), n={len(d):,} ===")
print(f"  mean   = {d.mean():+.3e}   (systematic bias)")
print(f"  median = {np.median(d):+.3e}")
print(f"  std    = {d.std():.3e}")
t_stat = d.mean()/(d.std()/np.sqrt(len(d)))
print(f"  t-stat of mean vs 0 = {t_stat:+.2f}   ({'SIGNIFICANT BIAS' if abs(t_stat)>3 else 'no meaningful bias'})")
for p in [0.1,1,50,99,99.9]:
    print(f"    p{p:<5} = {np.percentile(d,p):+.3e}")
print(f"  |diff| > 1e-6 : {100*(np.abs(d)>1e-6).mean():.3f}% of obs")
print(f"  |diff| > 1e-3 : {100*(np.abs(d)>1e-3).mean():.3f}%")
print(f"  |diff| > 1%   : {100*(np.abs(d)>0.01).mean():.3f}%")

print("\n=== price-ratio drift (SP500close/NYSEclose end-vs-start; ~0 = same adjustment basis) ===")
print(f"  median |drift| = {r['ratio_drift'].abs().median():.4f}")
print(f"  tickers with |drift| > 1%: {int((r['ratio_drift'].abs()>0.01).sum())}/{len(r)}")
print(f"  tickers with |drift| > 10%: {int((r['ratio_drift'].abs()>0.10).sum())}/{len(r)}")

bad = r[(r['corr'] < 0.999) | (r['ratio_drift'].abs() > 0.10)].sort_values('corr')
print(f"\n=== FLAGGED tickers (corr<0.999 or ratio-drift>10%): {len(bad)} ===")
if len(bad):
    print(bad.head(15).to_string(index=False,
        float_format=lambda x: f"{x:+.5f}"))
else:
    print("  none — the two sources agree everywhere")
