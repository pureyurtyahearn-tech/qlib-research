"""
Extend qlib us_data binary store from 2020-11-10 to 2026 using yfinance.

qlib binary format per instrument:
  [start_index: float32][val0: float32][val1: float32]...
where val[i] maps to calendar[start_index + i].

Strategy: scale yfinance raw prices to match qlib convention at the boundary
date (2020-11-10) so that daily returns are preserved at the seam.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf, os
from pathlib import Path

QLIB_DIR = Path("/home/codespace/.qlib/qlib_data/us_data")
BOUNDARY  = "2020-11-10"
END_DATE  = "2026-06-30"
FEATURES  = ["open", "close", "high", "low", "volume", "factor", "change"]

# ── 1. Load current calendar ──────────────────────────────────────────────────
cal_file = QLIB_DIR / "calendars" / "day.txt"
with open(cal_file) as f:
    old_cal = [l.strip() for l in f if l.strip()]
assert old_cal[-1] == BOUNDARY, f"Expected calendar to end {BOUNDARY}, got {old_cal[-1]}"
n_old = len(old_cal)   # 5250
print(f"Current calendar: {n_old} days, {old_cal[0]} → {old_cal[-1]}")

# ── 2. Get new trading dates via SPY ──────────────────────────────────────────
print("Fetching new trading dates from SPY...")
spy = yf.download("SPY", start=BOUNDARY, end=END_DATE, auto_adjust=False, progress=False)
new_dates = [d.strftime("%Y-%m-%d") for d in spy.index if d.strftime("%Y-%m-%d") > BOUNDARY]
print(f"New dates: {len(new_dates)} from {new_dates[0]} to {new_dates[-1]}")

# Update calendar on disk
with open(cal_file, "a") as f:
    for d in new_dates:
        f.write(d + "\n")
print(f"Calendar updated to {len(old_cal)+len(new_dates)} days")

# ── 3. Active nasdaq100 instruments ───────────────────────────────────────────
with open(QLIB_DIR / "instruments" / "nasdaq100.txt") as f:
    active = [ln.split("\t")[0].strip() for ln in f if "2099-12-31" in ln]
print(f"Active instruments: {len(active)}")

# ticker overrides: qlib uses old tickers that yfinance may not recognize
TICKER_MAP = {"FB": "META"}   # FB renamed to META in Oct 2021

# ── 4. Download yfinance raw data (auto_adjust=False keeps split convention) ──
print(f"\nDownloading yfinance data for {len(active)} tickers...")
yf_tickers = [TICKER_MAP.get(t, t) for t in active]

df_all = yf.download(
    yf_tickers,
    start=BOUNDARY,
    end=END_DATE,
    auto_adjust=False,
    group_by="ticker",
    progress=True,
    threads=True,
)

new_dates_set = set(new_dates)

# ── 5. Extend binary files for each instrument ────────────────────────────────
def read_bin(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype="<f")
    return int(data[0]), data[1:]   # (start_index, values)

def get_yf_series(ticker, col):
    # yfinance group_by="ticker" produces (ticker, field) MultiIndex
    yf_t = TICKER_MAP.get(ticker, ticker)
    try:
        if isinstance(df_all.columns, pd.MultiIndex):
            s = df_all[(yf_t, col)]
        else:
            s = df_all[col]
        s = s.copy()
        s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
        return s.to_dict()
    except Exception:
        return {}

ok, skip, fail = 0, 0, 0
for ticker in active:
    feat_dir = QLIB_DIR / "features" / ticker.lower()
    if not feat_dir.exists():
        print(f"  SKIP {ticker}: no feature directory")
        skip += 1
        continue

    # Verify existing binary is complete (all n_old values)
    close_path = feat_dir / "close.day.bin"
    start_idx, existing = read_bin(close_path)
    if start_idx + len(existing) != n_old:
        print(f"  SKIP {ticker}: unexpected binary length {start_idx}+{len(existing)} != {n_old}")
        skip += 1
        continue

    # Scale factor: align yfinance prices to qlib convention at boundary
    yf_close_dict = get_yf_series(ticker, "Close")
    boundary_yf   = yf_close_dict.get(BOUNDARY)
    boundary_qlib = float(existing[-1])

    if boundary_yf is None or np.isnan(boundary_yf) or boundary_qlib == 0:
        print(f"  SKIP {ticker}: no boundary price")
        skip += 1
        continue

    scale = boundary_qlib / float(boundary_yf)
    qlib_factor_last = float(np.frombuffer(
        (feat_dir / "factor.day.bin").read_bytes(), dtype="<f"
    )[-1])
    qlib_close_prev = boundary_qlib   # last known close (for change calc)

    # Build arrays of new values
    new_vals = {f: [] for f in FEATURES}
    yf_open_d   = get_yf_series(ticker, "Open")
    yf_high_d   = get_yf_series(ticker, "High")
    yf_low_d    = get_yf_series(ticker, "Low")
    yf_volume_d = get_yf_series(ticker, "Volume")

    for d in new_dates:
        c = yf_close_dict.get(d)
        if c is None or (isinstance(c, float) and np.isnan(c)):
            for feat in FEATURES:
                new_vals[feat].append(np.nan)
            qlib_close_prev = np.nan
            continue

        qlib_c = scale * float(c)
        qlib_o = scale * float(yf_open_d.get(d, np.nan) or np.nan)
        qlib_h = scale * float(yf_high_d.get(d, np.nan) or np.nan)
        qlib_l = scale * float(yf_low_d.get(d, np.nan) or np.nan)
        raw_v  = yf_volume_d.get(d)
        qlib_v = float(raw_v) if raw_v and not np.isnan(float(raw_v)) else np.nan
        chg    = (qlib_c / qlib_close_prev - 1) if (not np.isnan(qlib_close_prev) and qlib_close_prev != 0) else np.nan

        new_vals["open"].append(qlib_o)
        new_vals["close"].append(qlib_c)
        new_vals["high"].append(qlib_h)
        new_vals["low"].append(qlib_l)
        new_vals["volume"].append(qlib_v)
        new_vals["factor"].append(qlib_factor_last)
        new_vals["change"].append(chg)
        qlib_close_prev = qlib_c

    # Append to each binary file
    for feat in FEATURES:
        bin_path = feat_dir / f"{feat}.day.bin"
        if not bin_path.exists():
            continue
        arr = np.array(new_vals[feat], dtype="<f")
        with open(bin_path, "ab") as f:
            arr.tofile(f)

    data_days = sum(1 for v in new_vals["close"] if not np.isnan(v))
    print(f"  {ticker}: scale={scale:.4f}, added {data_days}/{len(new_dates)} days")
    ok += 1

print(f"\nDone: {ok} extended, {skip} skipped, {fail} failed")
print(f"Calendar now covers: {old_cal[0]} to {new_dates[-1]}")
