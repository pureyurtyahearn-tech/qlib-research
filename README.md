# RD-Agent × Claude — US Stock Alpha Factor Discovery

RD-Agent's `FactorRDLoop` running on **Claude (Anthropic)** with **US stock data (S&P 500)** inside a **GitHub Codespace** — no conda, no Docker, no OpenAI key required.

RD-Agent autonomously proposes alpha factors, codes them, backtests them with qlib, and iterates based on IC feedback. This repo wires it up to work out-of-the-box in a plain Python environment.

---

## Results

| Universe | Loops | Best IC | Notes |
|---|---|---|---|
| NASDAQ-100 | 10 | 0.003323 | Too efficient for basic factors |
| **S&P 500** | **10** | **0.012763** | Loop 5 — exceeds 0.01 target |

Test period: 2023-01-01 → 2026-06-16. Benchmark: SPY.

---

## What this repo fixes

RD-Agent ships configured for China (CSI300, conda, OpenAI, Docker). Six patches in `run_rdagent.py` make it work on US data in a standard Python environment:

| Patch | Problem | Fix |
|---|---|---|
| 0 | Claude returns nested JSON; RD-Agent validates as `dict[str,str]` → `ValidationError` | Pass `json_target_type=None` to skip strict type enforcement |
| 1 | Embeddings call OpenAI API; no valid key available | Replace with local `sentence-transformers/all-MiniLM-L6-v2` |
| 2 | `get_factor_env` requires a conda env named `rdagent` | Return `LocalEnv` using `sys.executable` instead |
| 3 | `QlibFBWorkspace.execute` runs `qrun` inside Docker/conda | Replace with `LocalEnv.check_output`; patch YAML configs from CSI300→SP500; add `MLFLOW_ALLOW_FILE_STORE=true` |
| 3a | mlflow 2.x crashes when both workspaces share a cache key (baseline IC leaks into combined-factors result) | `enable_cache=False` |
| 3b | `read_exp_res.py` finds FAILED recorders by `end_time` then crashes loading missing portfolio artifacts | Overwrite with version that skips FAILED recorders and falls back to a dummy `ret.pkl` |

Additional source-level fixes applied to the Codespace Python install:
- **`mlflow/store/tracking/file_store.py`** — retry on empty metric file (race condition where `Rank IC` was read before mlflow flushed it → `ValueError: Metric is malformed`)
- **`qlib/workflow/recorder.py`** — removed `@AsyncCaller.async_dec` from `log_metrics` (async write before synchronous read caused same race)
- **`rdagent/scenarios/qlib/developer/factor_runner.py`** — `pandarallel` with `use_memory_fs=True` hangs in subprocess context; replaced with `use_memory_fs=False` and `apply` instead of `parallel_apply`

---

## Setup

### 1. Prerequisites

- GitHub Codespace (4-core / 16 GB recommended; 2-core / 8 GB works with SP500 start_time=2015)
- Anthropic API key

### 2. Clone and install

```bash
git clone https://github.com/pureyurtyahearn-tech/qlib-research
cd qlib-research
pip install -r requirements.txt
```

### 3. Configure

Copy the example env file and add your key:

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY
```

### 4. Download qlib US data

```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us
```

This downloads the base dataset (NASDAQ-100 + SP500, up to ~2020). The next step extends it to the present.

### 5. Extend data to present

```bash
# Extend qlib binary store from 2020-11-10 to today
python extend_qlib_data.py

# Generate daily_pv.h5 for RD-Agent's factor code sandbox
python generate_rdagent_data.py
```

`extend_qlib_data.py` uses yfinance to append data for every SP500 instrument and extends the trading calendar. `generate_rdagent_data.py` builds the `daily_pv.h5` file that RD-Agent's generated Python factor code reads inside the sandbox.

### 6. Apply source patches

Three files in the Codespace Python install need direct edits:

**`mlflow/store/tracking/file_store.py`** — around line 877, find the `_get_metric_from_file` method. Change the empty-file check from `raise ValueError(...)` to wait 50ms and retry once. Without this, qrun crashes with `Metric 'Rank IC' is malformed` before portfolio analysis runs.

**`qlib/workflow/recorder.py`** — around line 450, find `log_metrics`. Remove the `@AsyncCaller.async_dec` decorator so metric writes are synchronous before the read-back.

**`rdagent/scenarios/qlib/developer/factor_runner.py`** — lines 9-17, change `pandarallel.initialize(use_memory_fs=True)` to `use_memory_fs=False` and replace `parallel_apply` with `apply`. `pandarallel` deadlocks in subprocess context with shared memory.

```python
# Find with: python -c "import mlflow; print(mlflow.__file__)"
# Find with: python -c "import qlib; print(qlib.__file__)"
# Find with: python -c "import rdagent; print(rdagent.__file__)"
```

### 7. Run

```bash
nohup python run_rdagent.py > run.log 2>&1 &
tail -f run.log
```

Loop progress and IC scores appear in the log as each loop completes. A run of 10 loops takes roughly 1-2 hours on an 8 GB Codespace.

---

## Project layout

```
run_rdagent.py              # Entry point — all patches live here
extend_qlib_data.py         # Extends qlib binary store to present via yfinance
generate_rdagent_data.py    # Builds daily_pv.h5 for factor sandbox
.env.example                # Copy to .env and fill in your API key
requirements.txt            # Python dependencies
git_ignore_folder/          # RD-Agent workspaces and factor data (gitignored)
log/                        # RD-Agent session logs with checkpoints (gitignored)
```

---

## Resuming a run

RD-Agent checkpoints after every step. If a run is interrupted:

```python
# In run_rdagent.py, replace the last two lines with:
loop = FactorRDLoop.load("log/<session-dir>", checkout=True)
asyncio.run(loop.run(loop_n=10))  # loop_n = total target, not additional loops
```

---

## Memory notes

SP500 (500 stocks) is large. On 8 GB with no swap:

- `start_time: 2015-01-01` in the YAML patch (not 2010) — saves ~38% data load
- `fit_start_time: 2018-01-01` — prevents OOM in the combined-factors model as accumulated factors grow across loops
- Both are set in the `_cn_to_us` dict inside `run_rdagent.py`

NASDAQ-100 runs comfortably with `start_time: 2010` if you prefer a longer history.

---

## References

- [RD-Agent](https://github.com/microsoft/RD-Agent) — Microsoft Research
- [Qlib](https://github.com/microsoft/qlib) — Microsoft quantitative investment platform
- [Claude API](https://docs.anthropic.com) — Anthropic
