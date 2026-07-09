# Local Machine Setup Guide

Full reproduction of the GitHub Codespaces environment on a local Linux/macOS machine.
For the quick public-facing setup, see README.md.
This document covers exact versions, source patches, and all non-Python steps.

---

## 1. Python version

**Requires Python 3.12.1 exactly.**
RD-Agent 0.8.0 and pyqlib 0.9.7 have subtle compatibility issues with other versions.

```bash
# Recommended: use pyenv
pyenv install 3.12.1
pyenv local 3.12.1
python --version   # must print 3.12.1
```

---

## 2. Install packages

```bash
# On a local machine WITHOUT a GPU (saves ~3 GB):
pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_exact.txt

# On a GPU machine or GitHub Codespaces (installs CUDA build automatically):
pip install -r requirements_exact.txt
```

`requirements_exact.txt` contains all packages with the exact versions from the working environment.
The base `requirements.txt` lists only direct dependencies without version pins, for flexibility.

---

## 3. Apply three source-level patches

These files in the installed packages need direct edits. They **cannot** be monkey-patched
from outside and must be applied after every `pip install --upgrade` of the affected package.

### 3a. mlflow — retry on empty metric file

**File:** `$(python -c "import mlflow, os; print(os.path.dirname(mlflow.__file__))")/store/tracking/file_store.py`

Find `_get_metric_from_file` (~line 869). Replace the first `if len(metric_objs) == 0: raise` block with a 50 ms retry before raising:

```python
# BEFORE:
if len(metric_objs) == 0:
    raise ValueError(f"Metric '{metric_name}' is malformed. No data found.")

# AFTER:
if len(metric_objs) == 0:
    # Retry once — metric file may be transiently empty due to async write race
    import time as _time
    _time.sleep(0.05)
    metric_objs = [
        FileStore._get_metric_from_line(run_id, metric_name, line, exp_id)
        for line in read_file_lines(parent_path, metric_name)
    ]
if len(metric_objs) == 0:
    raise ValueError(f"Metric '{metric_name}' is malformed. No data found.")
```

**Why:** qlib writes metrics via mlflow asynchronously. `PortAnaRecord.check()` can read the metric
file before the write is flushed, getting an empty file → `ValueError: Rank IC is malformed`.

To find the file: `python -c "import mlflow, os; print(os.path.dirname(mlflow.__file__) + '/store/tracking/file_store.py')"`

---

### 3b. qlib — make log_metrics synchronous

**File:** `$(python -c "import qlib, os; print(os.path.dirname(qlib.__file__))")/workflow/recorder.py`

Find `log_metrics` (~line 450). Remove the `@AsyncCaller.async_dec` decorator from it:

```python
# BEFORE:
@AsyncCaller.async_dec(ac_attr="async_log")
def log_metrics(self, step=None, **kwargs):
    for name, data in kwargs.items():
        self.client.log_metric(self.id, name, data, step=step)

# AFTER:
def log_metrics(self, step=None, **kwargs):
    # Synchronous (not async) to avoid race: async-created empty metric file
    # would cause _get_metric_from_file to raise "malformed. No data found."
    for name, data in kwargs.items():
        self.client.log_metric(self.id, name, data, step=step)
```

**Why:** same async write race as patch 3a — belt-and-suspenders fix.

To find the file: `python -c "import qlib, os; print(os.path.dirname(qlib.__file__) + '/workflow/recorder.py')"`

---

### 3c. rdagent — fix pandarallel deadlock

**File:** `$(python -c "import rdagent, os; print(os.path.dirname(rdagent.__file__))")/scenarios/qlib/developer/factor_runner.py`

Replace the `pandarallel.initialize(...)` call and patch `parallel_apply` at the top of the file:

```python
# BEFORE:
pandarallel.initialize(verbose=1, use_memory_fs=True)

# AFTER:
pandarallel.initialize(verbose=1, use_memory_fs=False)  # use /tmp not /dev/shm
# Patch parallel_apply to use regular apply — avoids pandarallel IPC deadlock in subprocess
import pandas.core.groupby.generic as _gpby
if not hasattr(_gpby.DataFrameGroupBy, '_orig_parallel_apply'):
    _gpby.DataFrameGroupBy._orig_parallel_apply = getattr(
        _gpby.DataFrameGroupBy, 'parallel_apply',
        _gpby.DataFrameGroupBy.apply
    )
    _gpby.DataFrameGroupBy.parallel_apply = _gpby.DataFrameGroupBy.apply
```

**Why:** `pandarallel` with `use_memory_fs=True` creates POSIX shared-memory semaphores.
Inside the qrun subprocess, these deadlock because the shm segments are inherited but the
parent's cleanup hook never fires. Switching to `/tmp` and falling back to regular `apply`
eliminates the deadlock with no correctness impact (just slightly slower).

To find the file: `python -c "import rdagent, os; print(os.path.dirname(rdagent.__file__) + '/scenarios/qlib/developer/factor_runner.py')"`

---

## 4. Configure environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY to your key
```

The `.env.example` in this repo shows all required variables. Key ones:

```
CHAT_MODEL=anthropic/claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=dummy-not-used      # required by litellm validation; not called
FACTOR_CoSTEER_DATA_FOLDER=git_ignore_folder/factor_implementation_source_data
FACTOR_CoSTEER_DATA_FOLDER_DEBUG=git_ignore_folder/factor_implementation_source_data
QLIB_PROVIDER_URI=~/.qlib/qlib_data/us_data
```

---

## 5. Download qlib US base data

```bash
python -m qlib.run.get_data qlib_data \
    --target_dir ~/.qlib/qlib_data/us_data \
    --region us
```

This downloads the qlib binary store covering the US market up to approximately 2020-11-10
(the version hosted by the qlib team). ~800 MB download.

---

## 6. Extend qlib data to present

```bash
python extend_qlib_data.py
```

Uses yfinance to append daily data for all SP500 instruments from 2020-11-10 to today,
and extends the trading calendar. Takes ~5–10 minutes. Requires internet access.

---

## 7. Build the factor sandbox data file

```bash
# SP500 only (required):
python generate_rdagent_data.py

# SP500 + NYSE combined (optional, ~60–90 min due to yfinance per-ticker API calls):
python fix_and_build_nyse.py
```

`generate_rdagent_data.py` fetches split/dividend-adjusted SP500 OHLCV from yfinance and
saves it as `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`.

`fix_and_build_nyse.py` fetches 3168 NYSE tickers from the Kaggle dataset
(`mousemover/quant-finance-nyse-5-years`), applies split and dividend adjustments sourced
from yfinance, then merges with the SP500 file (SP500 wins on overlap).

The combined file covers 3327 instruments (516 SP500 × 2010–2026, 2811 NYSE-only × 2019–2024).

---

## 8. Pre-download the sentence-transformers model (optional)

The `all-MiniLM-L6-v2` model (~90 MB) is downloaded automatically on the first run of
`run_rdagent.py`. To pre-download it:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Cached to: `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`

---

## 9. Required directory structure

```
qlib-research/
├── .env                          # your API keys (from .env.example)
├── run_rdagent.py                # entry point — all monkey-patches live here
├── extend_qlib_data.py           # extends qlib binary store via yfinance
├── generate_rdagent_data.py      # builds SP500 daily_pv.h5
├── fix_and_build_nyse.py         # builds SP500+NYSE daily_pv.h5
├── git_ignore_folder/
│   └── factor_implementation_source_data/
│       └── daily_pv.h5           # built by step 7 above
└── log/                          # created automatically on first run
```

`~/.qlib/qlib_data/us_data/` must exist (from steps 5–6).

---

## 10. Run

```bash
# Background (recommended for long runs):
nohup python run_rdagent.py > run.log 2>&1 &
tail -f run.log

# Foreground:
python run_rdagent.py
```

---

## 11. Resuming a run

RD-Agent checkpoints after every step. The checkpoint directory is printed at startup
(`log/YYYY-MM-DD_HH-MM-SS-XXXXXX/`). To resume:

```python
# In run_rdagent.py, replace the last two lines with:
loop = FactorRDLoop.load("log/<session-dir>", checkout=True)
asyncio.run(loop.run(loop_n=10))  # loop_n = total target, not additional
```

---

## Memory notes (for machines with limited RAM)

SP500 (500 stocks) is large. The `_cn_to_us` dict in `run_rdagent.py` already sets
conservative time windows. If you have ≥ 16 GB RAM you can loosen them:

| Setting | 8 GB (default) | 16 GB |
|---|---|---|
| `start_time` | 2015-01-01 | 2010-01-01 |
| `fit_start_time` | 2018-01-01 | 2015-01-01 |

For NASDAQ-100 only (~100 stocks), any time window fits in 8 GB comfortably.

---

## Verified results

| Universe | Loops | Best IC | Session |
|---|---|---|---|
| NASDAQ-100 | 10 | 0.003323 | run23 |
| **S&P 500** | **10** | **0.012763** | selector run, Loop 5 |

Test period: 2023-01-01 → 2026-06-16. Benchmark: SPY.
