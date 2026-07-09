# Session Notes — 2026-07-09

## 1. NYSE data pipeline (new)

Built a complete pipeline to expand the factor sandbox universe from S&P 500 alone
to S&P 500 + NYSE, totalling **3,327 instruments**.

**Problem identified:** The Kaggle dataset (`mousemover/quant-finance-nyse-5-years`,
3168 tickers, 2019–2024) is raw unadjusted OHLCV — confirmed by checking SHOP's
10:1 split on 2022-06-29 ($350 → $33 across the date boundary, ratio 10.6×).

**Approach:** Rather than re-downloading all prices from yfinance (slow) or using
ratio-based split detection from price jumps (false positives from COVID crash,
earnings disasters, warrant swings), fetched only split and dividend *metadata*
per ticker from yfinance (`get_splits()` / `get_dividends()`) and applied backward
adjustments to the Kaggle prices. ~60 min runtime for 3168 tickers.

**Adjustment order matters:** Splits applied first (exact ratios). Dividends second,
using the now-split-adjusted prev_close to compute `adj_factor = (prev_close - div) / prev_close`.
yfinance returns dividend amounts in split-adjusted share terms, so this order is required
for correct results.

**Verification:** Post-adjustment SHOP ratio across the split date: 1.06× (was 10.6×).
The residual reflects actual intraday market movement on the split day — correct.

**Merge:** SP500 (instrument, datetime) index order; NYSE built with (datetime, instrument)
then swaplevel()'d to match. SP500 wins on the 357 overlapping tickers (e.g. JPM, KO, XOM —
these keep their longer 2010–2026 history). 2,811 NYSE-only instruments added with 2019–2024 coverage.

**Output:** `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`
- Before: 56 MB, 516 instruments (SP500 only, 2010–2026)
- After: 93 MB, 3327 instruments (SP500 2010–2026 + NYSE 2019–2024)

**Scripts added:**
- `fix_and_build_nyse.py` — the production pipeline (metadata-based adjustment + merge)
- `build_nyse_data.py` — alternative full yfinance download approach (not used; kept for reference)

---

## 2. Disk cleanup (3.3 GB freed)

| Deleted | Size | Reason |
|---|---|---|
| `~/.cache/pip/http-v2/` | 3.1 GB | pip HTTP response cache; fully regeneratable |
| `git_ignore_folder/nasdaq_factor_data/` | 77 MB | Old NASDAQ-100 factor data; superseded by SP500+NYSE |
| `rdagent_run*.log` + `selector.log` (27 files) | 15 MB | Run history preserved in `project_rdagent_setup.md` |
| 34 old `log/` session dirs | ~106 MB | Abandoned earlier sessions; SP500 10-loop checkpoint kept |

Disk before: 24 GB used / 32 GB (80%). After: 21 GB used / 32 GB (68%). 9.7 GB free.

SP500 10-loop checkpoint retained: `log/2026-06-18_11-44-18-770696/` (85 MB, IC=0.012763).

---

## 3. Reproducibility docs added

| File | Purpose |
|---|---|
| `requirements_exact.txt` | Pinned versions of all 51 pipeline packages from the working Codespace |
| `local_setup.md` | 11-step local machine guide: Python 3.12.1, CPU torch install, all three source patches with before/after diffs, data download, directory structure, RAM tuning |

---

## 4. Environment audit findings (no action taken — recorded for reference)

- **No Chinese market data anywhere.** RD-Agent's default install contains no bundled `cn_data`.
  `~/.qlib/qlib_data/` has only `us_data/`.
- **NVIDIA/torch/triton (4.6 GB):** Pre-installed in Codespace base image with CUDA build.
  No GPU in this Codespace; torch runs CPU-only. Not worth uninstalling (breaks
  sentence-transformers dependency chain). Fixed overhead.
- **RD-Agent workspaces (2.4 GB, 137 dirs):** Per-factor backtest artifacts from all prior loops.
  Left in place — decision deferred.
- **sentence-transformers model (88 MB):** `all-MiniLM-L6-v2` in
  `~/.cache/huggingface/hub/`. Active — used by Patch 1 (local embeddings).

---

## 5. Current active file inventory

| File | Size | Status |
|---|---|---|
| `git_ignore_folder/factor_implementation_source_data/daily_pv.h5` | 93 MB | Active — SP500+NYSE combined |
| `git_ignore_folder/factor_implementation_source_data/daily_pv_sp500_backup.h5` | 54 MB | SP500-only backup; can delete once confident in merge |
| `git_ignore_folder/nyse_adjusted_daily_pv.h5` | 49 MB | Intermediate NYSE file; can delete |
| `~/.qlib/qlib_data/us_data/features/` | 827 MB | Active — qlib binary store (SP500, 454 tickers, 2010–2026) |
| `log/2026-06-18_11-44-18-770696/` | 85 MB | SP500 10-loop checkpoint (IC=0.012763) |
| `git_ignore_folder/RD-Agent_workspace/` | 2.4 GB | Factor evaluation artifacts, 137 dirs — decision deferred |
