"""EDA part 4: causality/look-ahead trace for vwap_deviation_10d on one (instrument,date).
Perturbation test: corrupting future/current data must NOT change factor[t];
corrupting t-1 MUST change it."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

SRC = "git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
df = pd.read_hdf(SRC).sort_index()
close0  = df["$close"].unstack("instrument")
volume  = df["$volume"].unstack("instrument")

INST = "AAPL"
dates = close0.index
t = dates.get_loc(pd.Timestamp("2024-06-03"))
tD, tm1, tp1 = dates[t], dates[t-1], dates[t+1]
print(f"instrument={INST}  t={tD.date()}  t-1={tm1.date()}  t+1={tp1.date()}")

def vwap_dev_at(close, inst, when):
    dv = close * volume
    vwap = (dv.rolling(10, min_periods=10).sum().shift(1)
            / volume.rolling(10, min_periods=10).sum().shift(1))
    fac = (close.shift(1) - vwap) / vwap
    return fac.loc[when, inst]

base = vwap_dev_at(close0, INST, tD)
print(f"\nbaseline factor[t]          = {base:.10f}")

# corrupt close[t] and close[t+1] (future/current) x1000
c1 = close0.copy(); c1.loc[tD, INST] *= 1000; c1.loc[tp1, INST] *= 1000
f1 = vwap_dev_at(c1, INST, tD)
print(f"factor[t] after close[t],close[t+1] x1000 = {f1:.10f}   -> delta={abs(f1-base):.2e}  "
      + ("NO LEAK (unchanged)" if abs(f1-base) < 1e-12 else "*** CHANGED = LOOK-AHEAD ***"))

# corrupt close[t-1] (legitimate past input) x1000
c2 = close0.copy(); c2.loc[tm1, INST] *= 1000
f2 = vwap_dev_at(c2, INST, tD)
print(f"factor[t] after close[t-1] x1000          = {f2:.10f}   -> delta={abs(f2-base):.2e}  "
      + ("changed (correctly uses t-1)" if abs(f2-base) > 1e-12 else "*** unchanged = ignores t-1? ***"))

# also verify the label side is forward-looking (should be), so factor(t) vs label(t) has no overlap
print("\n(For reference: factor[t] uses close[t-10..t-1]; a forward return label uses close[t..t+k]."
      "\n No shared date => no leakage between factor and label.)")
