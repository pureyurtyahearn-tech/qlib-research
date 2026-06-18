"""
RD-Agent fin_factor runner — Claude + US stocks, no conda/Docker required.

This script applies six monkey-patches that make RD-Agent's FactorRDLoop work with:
  - Claude (Anthropic) as the LLM instead of OpenAI
  - Local sentence-transformers for embeddings instead of OpenAI embeddings
  - The current Python interpreter instead of a conda environment
  - qrun executed locally instead of inside Docker
  - SP500 / US stock data instead of the default CSI300 China data

See README.md for full setup instructions and patch explanations.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE any rdagent imports so pydantic-settings sees the values
load_dotenv(".env")


# ── Patch 0: Relax JSON type validation for Claude responses ──────────────────
#
# Problem: RD-Agent validates every LLM JSON response with TypeAdapter(dict[str, str]).
#          Claude legitimately returns nested structures (lists, dicts inside dicts).
#          This causes a ValidationError and the loop crashes on the first response.
#
# Fix: Pass json_target_type=None so the raw JSON object is accepted as-is.
#      We still require valid JSON; we just don't enforce flat dict[str, str].
#
import rdagent.oai.backend.base as _oai_base
_original_auto_continue = _oai_base.APIBackend._create_chat_completion_auto_continue

import inspect as _inspect

async def _patched_auto_continue_async(self, messages, json_mode=False,
                                        chat_cache_prefix="", seed=None,
                                        json_target_type=None,
                                        add_json_in_prompt=False,
                                        response_format=None, **kwargs):
    return await _original_auto_continue(
        self, messages, json_mode=json_mode,
        chat_cache_prefix=chat_cache_prefix, seed=seed,
        json_target_type=None,  # <-- the actual fix
        add_json_in_prompt=add_json_in_prompt,
        response_format=response_format, **kwargs,
    )

def _patched_auto_continue_sync(self, messages, json_mode=False,
                                 chat_cache_prefix="", seed=None,
                                 json_target_type=None,
                                 add_json_in_prompt=False,
                                 response_format=None, **kwargs):
    return _original_auto_continue(
        self, messages, json_mode=json_mode,
        chat_cache_prefix=chat_cache_prefix, seed=seed,
        json_target_type=None,
        add_json_in_prompt=add_json_in_prompt,
        response_format=response_format, **kwargs,
    )

if _inspect.iscoroutinefunction(_original_auto_continue):
    _oai_base.APIBackend._create_chat_completion_auto_continue = _patched_auto_continue_async
else:
    _oai_base.APIBackend._create_chat_completion_auto_continue = _patched_auto_continue_sync


# ── Patch 1: Use local sentence-transformers instead of OpenAI embeddings ─────
#
# Problem: RD-Agent calls OpenAI's text-embedding API for its knowledge-base RAG.
#          Without a valid OpenAI key this fails after 10 retries and crashes the loop.
#
# Fix: Replace create_embedding on the backend class with a local
#      sentence-transformers model (all-MiniLM-L6-v2, ~90 MB, runs on CPU).
#      The model is lazy-loaded on first use.
#
from rdagent.oai.backend.litellm import LiteLLMAPIBackend as _BackendCls

_ST_MODEL = None

def _local_create_embedding(self, input_content, *args, **kwargs):
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    if isinstance(input_content, str):
        return _ST_MODEL.encode([input_content])[0].tolist()
    if isinstance(input_content, list):
        return [_ST_MODEL.encode([t])[0].tolist() for t in input_content]
    return _ST_MODEL.encode([str(input_content)])[0].tolist()

_BackendCls.create_embedding = _local_create_embedding


# ── Patch 2: Use current Python interpreter instead of conda ──────────────────
#
# Problem: get_factor_env() looks for a conda environment named "rdagent".
#          GitHub Codespaces has no conda; the call raises EnvironmentError.
#
# Fix: Return a LocalEnv that uses sys.executable's bin directory directly.
#      Patch both callsites where get_factor_env is imported by name.
#
from rdagent.utils.env import LocalConf, LocalEnv

def _patched_get_factor_env(
    conf_type=None,
    extra_volumes={},
    running_timeout_period=600,
    enable_cache=None,
):
    python_bin_dir = str(Path(sys.executable).parent)
    env = LocalEnv(conf=LocalConf(bin_path=python_bin_dir, default_entry="python main.py"))
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env

import rdagent.components.coder.factor_coder.config as _fc_config
_fc_config.get_factor_env = _patched_get_factor_env

import rdagent.scenarios.qlib.experiment.factor_experiment as _fe
_fe.get_factor_env = _patched_get_factor_env


# ── Patch 3: Replace QlibFBWorkspace.execute with a local implementation ──────
#
# Problem: The default execute() tries to run qrun inside Docker or a
#          QlibCondaEnv. Neither exists in Codespaces.
#
# This patch does five things:
#
#   (a) Rewrites the generated YAML configs from CSI300/China to SP500/US.
#       Key ordering matters: the multi-line backtest block must be replaced
#       BEFORE the single-line end_time replacement, or the pattern is destroyed.
#       SP500 on 7.8 GB RAM: start_time=2015 (not 2010) to stay within memory.
#       fit_start_time=2018 prevents OOM when the combined-factors model accumulates
#       many loops of factors and tries to train on the full window at once.
#
#   (b) Overwrites read_exp_res.py with a robust version that:
#       - Skips FAILED recorders when looking for portfolio_analysis artifacts
#         (RD-Agent's original version finds FAILED recorders by end_time and then
#         crashes on the missing portfolio_analysis/report_normal_1day.pkl)
#       - Falls back to a dummy ret.pkl so execute() doesn't short-circuit
#
#   (c) Runs qrun via LocalEnv with MLFLOW_ALLOW_FILE_STORE=true.
#       mlflow 2.x+ refuses to use the local file store without this env var.
#
#   (d) Sets enable_cache=False to prevent cache-key collisions between the
#       baseline workspace and the combined-factors workspace. With caching on,
#       both workspaces returned the cached baseline IC, making every loop look
#       identical.
#
#   (e) Returns (pd.Series of metrics, log) matching the expected signature.
#
import rdagent.scenarios.qlib.experiment.workspace as _ws_mod

_python_bin = str(Path(sys.executable).parent)

def _local_workspace_execute(self, qlib_config_name="conf.yaml", run_env={}, *args, **kwargs):
    from rdagent.utils.env import LocalConf, LocalEnv

    # (a) Patch YAML configs: China → US / SP500
    # NOTE: dict key ordering is intentional — backtest block before end_time.
    _cn_to_us = {
        'provider_uri: "~/.qlib/qlib_data/cn_data"': 'provider_uri: "~/.qlib/qlib_data/us_data"',
        "region: cn":                    "region: us",
        "market: &market csi300":        "market: &market sp500",
        "benchmark: &benchmark SH000300": "benchmark: &benchmark SPY",
        "start_time: 2008-01-01":        "start_time: 2015-01-01",
        # Multi-line backtest block MUST be replaced before the single-line
        # "end_time: 2020-08-01" entry below — otherwise that replacement
        # destroys this pattern before it can match.
        "backtest:\n        start_time: 2017-01-01\n        end_time: 2020-08-01":
            "backtest:\n        start_time: 2023-01-01\n        end_time: 2026-06-16",
        "end_time: 2020-08-01":          "end_time: 2026-06-16",
        "end_time: 2022-08-01":          "end_time: 2026-06-16",
        # fit window: 2018 not 2015 — combined-factors model OOMs with 500 stocks × 7yr
        "fit_start_time: 2008-01-01":    "fit_start_time: 2018-01-01",
        "fit_end_time: 2014-12-31":      "fit_end_time: 2022-12-31",
        "limit_threshold: 0.095":        "limit_threshold: null",  # CN circuit-breaker, not applicable to US
        "train: [2008-01-01, 2014-12-31]": "train: [2018-01-01, 2020-12-31]",
        "valid: [2015-01-01, 2016-12-31]": "valid: [2021-01-01, 2022-12-31]",
        "test: [2017-01-01, 2020-08-01]":  "test: [2023-01-01, 2026-06-16]",
    }
    for yaml_name in ["conf_baseline.yaml", "conf_combined_factors.yaml",
                       "conf_combined_factors_sota_model.yaml"]:
        yaml_path = self.workspace_path / yaml_name
        if yaml_path.exists():
            content = yaml_path.read_text()
            for old, new in _cn_to_us.items():
                content = content.replace(old, new)
            yaml_path.write_text(content)

    # (b) Overwrite read_exp_res.py with a version that handles FAILED recorders.
    # RD-Agent's original selects the recorder with the latest end_time; FAILED
    # recorders have end_time set too, so it finds them, then crashes trying to
    # load portfolio_analysis/report_normal_1day.pkl which was never written.
    _robust_read_exp_res = '''\
import pickle
from pathlib import Path

import pandas as pd
import qlib
from mlflow.tracking import MlflowClient

qlib.init()
from qlib.workflow import R

experiments = R.list_experiments()

latest_finished = None
latest_any = None
for experiment in experiments:
    recorders = R.list_recorders(experiment_name=experiment)
    for recorder_id in recorders:
        if recorder_id is None:
            continue
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)
        end_time = recorder.info.get("end_time")
        if end_time is None:
            continue
        status = recorder.info.get("status", "")
        if latest_any is None or end_time > latest_any.info.get("end_time", ""):
            latest_any = recorder
        if status not in ("FAILED", 4):
            if latest_finished is None or end_time > latest_finished.info.get("end_time", ""):
                latest_finished = recorder

latest_recorder = latest_finished if latest_finished is not None else latest_any

if latest_recorder is None:
    print("No recorders found")
else:
    print(f"Latest recorder: {latest_recorder} status={latest_recorder.info.get('status')}")
    metrics = pd.Series(latest_recorder.list_metrics())
    output_path = Path(__file__).resolve().parent / "qlib_res.csv"
    metrics.to_csv(output_path)
    print(f"Metrics saved to {output_path}: {metrics.to_dict()}")

    try:
        ret_data_frame = latest_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        ret_data_frame.to_pickle("ret.pkl")
        print("ret.pkl saved from portfolio_analysis")
    except Exception as e:
        print(f"portfolio_analysis unavailable ({e}), writing dummy ret.pkl")
        pd.Series(dtype=float, name="return").to_pickle("ret.pkl")
'''
    res_path = self.workspace_path / "read_exp_res.py"
    if res_path.exists():
        res_path.write_text(_robust_read_exp_res)

    # (c) Run qrun locally; (d) disable caching to prevent cross-workspace pollution
    env = LocalEnv(conf=LocalConf(bin_path=_python_bin, default_entry=f"qrun {qlib_config_name}"))
    env.conf.enable_cache = False  # (d) avoids baseline IC leaking into combined-factors result
    env.prepare()

    # (c) MLFLOW_ALLOW_FILE_STORE=true is required by mlflow 2.x+ for file-based tracking
    _mlflow_env = {"PYTHONPATH": "./", "MLFLOW_ALLOW_FILE_STORE": "true", **run_env}

    execute_qlib_log = env.check_output(
        local_path=str(self.workspace_path),
        entry=f"qrun {qlib_config_name}",
        env=_mlflow_env,
    )

    execute_log = env.check_output(
        local_path=str(self.workspace_path),
        entry="python read_exp_res.py",
        env=_mlflow_env,
    )

    import pandas as pd
    quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
    if quantitative_backtesting_chart_path.exists():
        ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
        from rdagent.log import rdagent_logger as logger
        logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
    else:
        from rdagent.log import rdagent_logger as logger
        logger.error("No result file found.")
        return None, execute_qlib_log

    qlib_res_path = self.workspace_path / "qlib_res.csv"
    if qlib_res_path.exists():
        return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
    else:
        from rdagent.log import rdagent_logger as logger
        logger.error(f"File {qlib_res_path} does not exist.")
        return None, execute_qlib_log

_ws_mod.QlibFBWorkspace.execute = _local_workspace_execute


# ── Now safe to import the scenario and loop ───────────────────────────────────
import asyncio
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING

print("=" * 60)
print("RD-Agent fin_factor — autonomous alpha factor discovery")
print(f"Model : {os.environ.get('CHAT_MODEL', 'not set')}")
print(f"Config: {FACTOR_PROP_SETTING}")
print("=" * 60)

# To resume a previous session, replace FactorRDLoop(FACTOR_PROP_SETTING) with:
#   loop = FactorRDLoop.load("log/<session-dir>", checkout=True)
# loop_n counts *total* loops completed in the session, not additional loops.
loop = FactorRDLoop(FACTOR_PROP_SETTING)
asyncio.run(loop.run(loop_n=10))
