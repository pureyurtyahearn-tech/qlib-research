# RD-Agent Setup — Internal Notes

Public-facing documentation is in README.md. This file records implementation decisions, exact patch locations, and run history for ongoing development.

---

## Environment

| Package | Version |
|---|---|
| rdagent | 0.8.0 |
| pyqlib | 0.9.7 |
| mlflow | 3.13.0 |
| litellm | 1.89.1 |
| sentence-transformers | 5.5.1 |
| yfinance | 1.4.1 |

- Python: `/home/codespace/.python/current/` (run_rdagent.py)
- Python: `/usr/local/python/3.12.1/` (qrun subprocess — source patches target this)
- Qlib data: `~/.qlib/qlib_data/us_data/`
- Calendar: 6655 entries, 1999-12-31 → 2026-06-17

---

## Source patches (applied directly to installed packages)

These cannot be monkey-patched from outside; they require editing the installed files.

**`mlflow/store/tracking/file_store.py` ~line 877** (`_get_metric_from_file`)
- Problem: reads metric file synchronously in a brief window when qlib has created it but not yet flushed — `ValueError: Metric 'Rank IC' is malformed. No data found.`
- Fix: add 50ms retry when `len(metric_objs) == 0` before raising

**`qlib/workflow/recorder.py` ~line 450** (`log_metrics`)
- Problem: `@AsyncCaller.async_dec` makes metric writes fire-and-forget; same race as above
- Fix: remove the decorator so writes complete before read-back

**`rdagent/scenarios/qlib/developer/factor_runner.py` lines 9-17**
- Problem: `pandarallel.initialize(use_memory_fs=True)` creates shared-memory semaphores that deadlock in subprocess context
- Fix: `use_memory_fs=False`; replace all `parallel_apply` → `apply`

---

## Monkey-patches (in run_rdagent.py)

See inline comments in run_rdagent.py for full explanations. Summary:

| Patch | Target | What it does |
|---|---|---|
| 0 | `APIBackend._create_chat_completion_auto_continue` | Bypass `dict[str,str]` TypeAdapter validation for Claude responses |
| 1 | `LiteLLMAPIBackend.create_embedding` | Local sentence-transformers instead of OpenAI |
| 2 | `get_factor_env` (two import sites) | `LocalEnv` with `sys.executable` instead of conda |
| 3 | `QlibFBWorkspace.execute` | Full replacement: local qrun, YAML patches, robust read_exp_res.py, cache disabled |

---

## YAML substitution map

Key ordering is critical — the multi-line backtest block must precede the single-line `end_time` entry.

```python
_cn_to_us = {
    'provider_uri: "~/.qlib/qlib_data/cn_data"': 'provider_uri: "~/.qlib/qlib_data/us_data"',
    "region: cn":                    "region: us",
    "market: &market csi300":        "market: &market sp500",
    "benchmark: &benchmark SH000300": "benchmark: &benchmark SPY",
    "start_time: 2008-01-01":        "start_time: 2015-01-01",
    # MUST be before end_time replacement:
    "backtest:\n        start_time: 2017-01-01\n        end_time: 2020-08-01":
        "backtest:\n        start_time: 2023-01-01\n        end_time: 2026-06-16",
    "end_time: 2020-08-01":          "end_time: 2026-06-16",
    "end_time: 2022-08-01":          "end_time: 2026-06-16",
    "fit_start_time: 2008-01-01":    "fit_start_time: 2018-01-01",
    "fit_end_time: 2014-12-31":      "fit_end_time: 2022-12-31",
    "limit_threshold: 0.095":        "limit_threshold: null",
    "train: [2008-01-01, 2014-12-31]": "train: [2018-01-01, 2020-12-31]",
    "valid: [2015-01-01, 2016-12-31]": "valid: [2021-01-01, 2022-12-31]",
    "test: [2017-01-01, 2020-08-01]":  "test: [2023-01-01, 2026-06-16]",
}
```

Note: `end_time` must be `2026-06-16`, not `2026-06-17` (the last calendar date). qlib uses an exclusive end index; `2026-06-17` maps to index 6655 which is out-of-bounds for a calendar of size 6655.

---

## Data quality notes (SP500, post-audit 2026-06-18)

**Qlib binary store (454 instruments at full 6655 coverage)**
- Post-boundary NaN: 0.00% — clean
- Zero/negative prices: 0
- Boundary continuity: no >15% overnight jump at 2020-11-10 seam

**daily_pv.h5 (post-cleanup: 516 instruments, 56.3 MB)**
- 123 all-NaN historical tickers removed (renamed/delisted, not fetchable from yfinance)
- CCE removed (53.4% NaN in trading period — exceeded 20% threshold)
- 285 isolated 1-2 day gaps forward-filled
- Post-2021 NaN: 2-4% (pre-IPO absence for stocks that joined SP500 after 2021)
- `$factor`: all 1.0, no NaN

---

## Run history

| Run | Universe | Loops | Best IC | Notes |
|---|---|---|---|---|
| run15 | NASDAQ-100 | 1 | +0.004342 | First clean run after all patches |
| run16–16c | NASDAQ-100 | 3 | +0.004342 | daily_pv.h5 date range bug — IC stuck |
| run17 | NASDAQ-100 | 3 | +0.004342 | daily_pv.h5 regenerated; IC still flat (date range fix) |
| run22 | NASDAQ-100 | 1 | +0.001091 | Post-extension verification |
| run23 | NASDAQ-100 | 3 | +0.003323 | Conclusion: NASDAQ-100 too efficient |
| run (SP500 v1) | SP500 | 1 | +0.003065 | Verification run |
| run (SP500 v2) | SP500 | 3 | +0.010758 | Loop 2 exceeded 0.01 target |
| run_selector (10 loops) | SP500 | 10 | **+0.012763** | Loop 5 best; OOM at Loop 7 fixed by fit_start_time→2018 |

---

## Known limitations

- **CoSTEER knowledge retrieval degrades** when `hypothesis.concise_justification` is None. Factor code is still generated and evaluated, but the RAG step falls back to defaults rather than learning from prior successes.
- **No swap on Codespace** — combined-factors model memory spikes as loops accumulate. The 2018 fit window is the current mitigation; a larger machine (16 GB) would allow 2015.
- **CatBoost and XGBoost not installed** — rdagent logs `ModuleNotFoundError` for both on every run. This is harmless; LightGBM handles the backtest.
