# Session Notes — 2026-07-15

## Point-in-time S&P 500 universe + qlib ghost-position fix

### Survivorship bias found and removed
The universe used all week (`daily_pv.h5`, 505 SP500 names) was **today's constituents
backfilled to 2010** — a survivor universe. Using SHARADAR/SP500 (Nasdaq Data Link) we
reconstructed true point-in-time membership: **822 tickers were in the index at some point
2010–2026**, of which 322 were genuinely missing from our data (236 removed/delisted +
86 recent additions our stale list never had). Reconstruction validated against 114
quarterly membership snapshots — **exact, 0 disagreement**.

Subscribed to SHARADAR/SEP and pulled prices for all 822 (incl. 340 names we never had —
Kodak, PG&E, Celgene, Red Hat, E*Trade…). Cross-validated vs yfinance: median daily-return
correlation **0.99998**, no systematic bias; the handful of disagreements are **spinoffs
where Sharadar's `closeadj` is correct and yfinance is wrong**. New qlib store at
`~/.qlib/qlib_data/us_data_pit` with membership encoded as spans in
`instruments/sp500pit.txt`.

**Impact:** holding prices and window fixed and varying only the universe, survivorship was
overstating 12-1 momentum's edge by **+4.02%/yr at K=20 — a 96% overstatement** that flips
the result from "significant" (backfilled t=2.04) to "not significant" (true PIT t=1.00).

### ⚠️ qlib native ghost-position bug — FIXED
qlib's stock `TopkDropoutStrategy` **never sells a holding once it leaves the index / delists**
(the sell loop skips any name that isn't currently tradable, and caps sells at `n_drop`
from *scored* holdings only). On the PIT store this froze acquired/delisted winners in the
book forever at their last price: **29,803 ghost position-days fabricating +1.84 (184 pts)
of net return** over 2016–2021.

Two-part fix:
- **`pit13_fix_store.py`** — 95 of 329 index exits are same-day delistings (acquisitions:
  ABMD, AET, AGN, ATVI, BCR…) with no price on the first non-member day, so a normal SELL
  can't execute. Appends flat liquidation bars (last close, 0% return) through the first
  non-member day so the exit fills. Not look-ahead (synthetic prints on non-member days,
  which can't be bought). Idempotent.
- **`pit_strategy.py` → `PITTopkDropoutStrategy`** — force-sells every holding not in the
  index today (bypassing the `n_drop` cap) and restricts buys to current members.

**Verified** (`pit14_verify_fixed.py`): ghost position-days **29,803 → 0**, 0 unsellable,
delisted names still genuinely held then exited on removal.

> **Any native `qrun`/`backtest` on `us_data_pit` MUST use `PITTopkDropoutStrategy`, not
> stock `TopkDropoutStrategy`** — otherwise it fabricates returns on delisted names. The
> standalone pandas simulators (`pit9_rerun.py`, `pit11_attribute.py`) already enforce PIT
> and are unaffected.

### ⚠️ Memory: `kernels=1` is required on this machine
`qlib.init(..., kernels=1)` is **required**. The default joblib parallel data loader spawns
Windows worker processes that each duplicate the panel; with ~6 GB free it OOMs and **crashed
VS Code three times on 2026-07-15**. `kernels=1` = serial in-process load, flat footprint.
Also: build signals from `sep_panel.h5` directly rather than `D.features` over the whole
`sp500pit` universe (avoids a 3M-row unstack), and pass explicit `codes=` to the exchange.

Scripts: `pit1`–`pit14`, `pit_strategy.py`. Data (gitignored, regenerable): `us_data_pit`
store, `git_ignore_folder/sharadar/` (SEP panel + membership matrix).

---

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
