"""Extend a single benchmark ticker's qlib bins to the calendar end (generalized from
extend_spy.py). Usage: python extend_benchmark.py RSP"""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path

TICKER = sys.argv[1] if len(sys.argv) > 1 else "RSP"
QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "us_data"
BOUNDARY = "2020-11-10"
END_DATE = "2026-06-30"
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]

cal = [l.strip() for l in open(QLIB_DIR/"calendars"/"day.txt") if l.strip()]
n_cal = len(cal); cal_end = cal[-1]
feat_dir = QLIB_DIR/"features"/TICKER.lower()
assert feat_dir.exists(), f"{feat_dir} missing"

def read_bin(p):
    d = np.frombuffer(Path(p).read_bytes(), dtype="<f"); return int(d[0]), d[1:]

start_idx, existing = read_bin(feat_dir/"close.day.bin")
n_have = start_idx + len(existing)
print(f"{TICKER}: has {len(existing)} vals -> {cal[n_have-1]}  (calendar end {cal_end})")
if n_have >= n_cal:
    print("already full."); sys.exit(0)
assert cal[n_have-1] == BOUNDARY, f"expected boundary {BOUNDARY}, got {cal[n_have-1]}"
new_dates = cal[n_have:]

spy = yf.download(TICKER, start=BOUNDARY, end=END_DATE, auto_adjust=False, progress=False)
if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
spy.index = pd.to_datetime(spy.index).strftime("%Y-%m-%d")
cd, od, hd, ld, vd = (spy[c].to_dict() for c in ["Close","Open","High","Low","Volume"])

bq = float(existing[-1]); byf = cd.get(BOUNDARY)
scale = bq/float(byf); factor_last = float(np.frombuffer((feat_dir/"factor.day.bin").read_bytes(), dtype="<f")[-1])
print(f"  scale={scale:.6f}")
vals = {f: [] for f in FEATURES}; prev = bq
for d in new_dates:
    c = cd.get(d)
    if c is None or (isinstance(c,float) and np.isnan(c)):
        for f in FEATURES: vals[f].append(np.nan)
        prev = np.nan; continue
    qc = scale*float(c)
    vals["open"].append(scale*float(od.get(d,np.nan))); vals["close"].append(qc)
    vals["high"].append(scale*float(hd.get(d,np.nan))); vals["low"].append(scale*float(ld.get(d,np.nan)))
    rv = vd.get(d); vals["volume"].append(float(rv) if rv and not np.isnan(float(rv)) else np.nan)
    vals["factor"].append(factor_last)
    vals["change"].append((qc/prev-1) if (not np.isnan(prev) and prev!=0) else np.nan); prev = qc
for f in FEATURES:
    p = feat_dir/f"{f}.day.bin"
    if p.exists(): np.array(vals[f], dtype="<f").tofile(open(p,"ab"))
_, chk = read_bin(feat_dir/"close.day.bin")
assert start_idx+len(chk) == n_cal, "length mismatch!"
print(f"  appended {len(new_dates)} rows -> now {start_idx+len(chk)} (cal {n_cal})")

# update all.txt end date
ap = QLIB_DIR/"instruments"/"all.txt"; lines = ap.read_text().splitlines()
for i,ln in enumerate(lines):
    p = ln.split("\t")
    if p and p[0].strip().upper() == TICKER:
        p[-1] = cal_end; lines[i] = "\t".join(p); print(f"  all.txt: {TICKER} end -> {cal_end}"); break
ap.write_text("\n".join(lines)+"\n")
print("done.")
