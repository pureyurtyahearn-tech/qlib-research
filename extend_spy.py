"""
Targeted extension of the SPY benchmark in the qlib us_data store.

Background: extend_qlib_data.py extended the SP500 *constituents* to 2026 but only
downloaded SPY to derive the new trading calendar (line 30) — it never extended
SPY's own feature bins, because SPY is not in sp500.txt. As a result the qlib
store has SPY only through 2020-11-10, while the RD-Agent backtest window is
2023-01-01..2026-06-16 with `benchmark: SPY`. With no benchmark data in that
window, qlib's PortAnaRecord emits no excess_return_with_cost.* metrics, and
RD-Agent's feedback step raises KeyError on those metrics.

This script appends SPY bins from the boundary (2020-11-10) to the current
calendar end, reusing the exact scale-at-boundary convention from
extend_qlib_data.py, and updates SPY's listing range in instruments/all.txt so
qlib will load the benchmark past 2020. It is idempotent-guarded: it refuses to
run if SPY already covers the full calendar.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path

QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "us_data"
BOUNDARY = "2020-11-10"
END_DATE = "2026-06-30"
FEATURES = ["open", "close", "high", "low", "volume", "factor", "change"]
TICKER   = "SPY"

# ── 1. Full (already-extended) calendar ───────────────────────────────────────
with open(QLIB_DIR / "calendars" / "day.txt") as f:
    cal = [l.strip() for l in f if l.strip()]
n_cal = len(cal)
print(f"Calendar: {n_cal} days, {cal[0]} -> {cal[-1]}")

# ── 2. SPY current bins ───────────────────────────────────────────────────────
feat_dir = QLIB_DIR / "features" / TICKER.lower()
assert feat_dir.exists(), f"{feat_dir} missing"

def read_bin(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype="<f")
    return int(data[0]), data[1:]

start_idx, existing = read_bin(feat_dir / "close.day.bin")
n_spy = start_idx + len(existing)
print(f"SPY currently covers {len(existing)} values, calendar[{start_idx}..{n_spy-1}] = "
      f"{cal[start_idx]} -> {cal[n_spy-1]}")

if n_spy >= n_cal:
    print("SPY already covers the full calendar — nothing to do.")
    raise SystemExit(0)

assert cal[n_spy - 1] == BOUNDARY, f"Expected SPY to end at {BOUNDARY}, got {cal[n_spy-1]}"
new_dates = cal[n_spy:]           # exactly the dates SPY is missing
print(f"Missing {len(new_dates)} days: {new_dates[0]} -> {new_dates[-1]}")

# ── 3. Download SPY (auto_adjust=False → same split convention as the store) ───
print("Downloading SPY from yfinance...")
spy = yf.download(TICKER, start=BOUNDARY, end=END_DATE, auto_adjust=False, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy.index = pd.to_datetime(spy.index).strftime("%Y-%m-%d")
close_d  = spy["Close"].to_dict()
open_d   = spy["Open"].to_dict()
high_d   = spy["High"].to_dict()
low_d    = spy["Low"].to_dict()
volume_d = spy["Volume"].to_dict()

# ── 4. Scale yfinance prices to qlib convention at the boundary ───────────────
boundary_qlib = float(existing[-1])
boundary_yf   = close_d.get(BOUNDARY)
assert boundary_yf and not np.isnan(boundary_yf), "No SPY yfinance close at boundary"
scale = boundary_qlib / float(boundary_yf)
factor_last = float(np.frombuffer((feat_dir / "factor.day.bin").read_bytes(), dtype="<f")[-1])
print(f"boundary qlib close={boundary_qlib:.4f}, yf close={boundary_yf:.4f}, scale={scale:.6f}, factor={factor_last:.6f}")

new_vals = {f: [] for f in FEATURES}
prev = boundary_qlib
for d in new_dates:
    c = close_d.get(d)
    if c is None or (isinstance(c, float) and np.isnan(c)):
        for feat in FEATURES:
            new_vals[feat].append(np.nan)
        prev = np.nan
        continue
    qc = scale * float(c)
    new_vals["open"].append(scale * float(open_d.get(d, np.nan)))
    new_vals["close"].append(qc)
    new_vals["high"].append(scale * float(high_d.get(d, np.nan)))
    new_vals["low"].append(scale * float(low_d.get(d, np.nan)))
    rv = volume_d.get(d)
    new_vals["volume"].append(float(rv) if rv and not np.isnan(float(rv)) else np.nan)
    new_vals["factor"].append(factor_last)
    new_vals["change"].append((qc / prev - 1) if (not np.isnan(prev) and prev != 0) else np.nan)
    prev = qc

# ── 5. Append to SPY bins ─────────────────────────────────────────────────────
for feat in FEATURES:
    bin_path = feat_dir / f"{feat}.day.bin"
    if not bin_path.exists():
        print(f"  (no {feat}.day.bin, skipping)")
        continue
    np.array(new_vals[feat], dtype="<f").tofile(open(bin_path, "ab"))

_, chk = read_bin(feat_dir / "close.day.bin")
data_days = sum(1 for v in new_vals["close"] if not np.isnan(v))
print(f"Appended {len(new_dates)} rows ({data_days} with data). "
      f"SPY close.day.bin now {start_idx}+{len(chk)} = {start_idx+len(chk)} (calendar={n_cal})")
assert start_idx + len(chk) == n_cal, "SPY bin length does not match calendar after append!"

# ── 6. Update SPY listing range in instruments/all.txt ────────────────────────
all_path = QLIB_DIR / "instruments" / "all.txt"
lines = all_path.read_text().splitlines()
updated = False
for i, ln in enumerate(lines):
    parts = ln.split("\t")
    if parts and parts[0].strip().upper() == "SPY":
        old = ln
        parts[-1] = cal[-1]                     # new end date = calendar end
        lines[i] = "\t".join(parts)
        updated = True
        print(f"all.txt SPY: '{old}' -> '{lines[i]}'")
        break
if updated:
    all_path.write_text("\n".join(lines) + "\n")
else:
    print("WARNING: SPY not found in all.txt")

print("\nSPY benchmark extension complete.")
