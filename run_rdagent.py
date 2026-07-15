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

# ── Windows: force Python UTF-8 Mode ──────────────────────────────────────────
# RD-Agent writes LLM-generated factor code via Path.write_text() and reads
# subprocess output in text mode. On Windows these default to cp1252, which
# raises UnicodeEncode/DecodeError on the non-Latin1 characters models routinely
# emit (e.g. '̄' combining macron for x̄). UTF-8 Mode makes all file and
# pipe I/O default to UTF-8. It must be enabled at interpreter start, so if we
# are not already in UTF-8 Mode, re-exec with -X utf8. PYTHONUTF8=1 is exported
# so multiprocessing-spawned workers also start in UTF-8 Mode (and therefore
# skip this block instead of re-exec'ing themselves).
if sys.platform == "win32" and not sys.flags.utf8_mode:
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.execv(sys.executable, [sys.executable, "-X", "utf8",
                              os.path.abspath(__file__), *sys.argv[1:]])

from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE any rdagent imports so pydantic-settings sees the values.
# override=True is required because a stale OS-level ANTHROPIC_API_KEY (User env var)
# would otherwise shadow the .env value — python-dotenv does not override existing
# environment variables by default, so .env must be made authoritative here.
load_dotenv(".env", override=True)


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
        'provider_uri: "~/.qlib/qlib_data/cn_data"': 'provider_uri: "~/.qlib/qlib_data/us_data_pit_full"',
        "region: cn":                    "region: us",
        "market: &market csi300":        "market: &market sp500pit",
        "benchmark: &benchmark SH000300": "benchmark: &benchmark SP500EW",
        "start_time: 2008-01-01":        "start_time: 2010-01-01",  # 16 GB RAM tier (was 2015 for 8 GB)
        # Multi-line backtest block MUST be replaced before the single-line
        # "end_time: 2020-08-01" entry below — otherwise that replacement
        # destroys this pattern before it can match.
        # HOLDOUT DISCIPLINE (2026-07-15): RD-Agent generation/selection must NOT see the
        # 2024-01-01 -> 2026-06 holdout. All windows below end at 2023-12-31 so its backtest
        # IC feedback is computed only on <=2023 data. The holdout is confirmed separately,
        # once, on the single pre-registered factor (see ext16/ext17).
        "backtest:\n        start_time: 2017-01-01\n        end_time: 2020-08-01":
            "backtest:\n        start_time: 2022-01-01\n        end_time: 2023-12-31",
        "end_time: 2020-08-01":          "end_time: 2023-12-31",
        "end_time: 2022-08-01":          "end_time: 2023-12-31",
        # fit window: 2015 for 16 GB RAM tier (was 2018 for 8 GB, per local_setup.md RAM table)
        "fit_start_time: 2008-01-01":    "fit_start_time: 2015-01-01",
        "fit_end_time: 2014-12-31":      "fit_end_time: 2021-12-31",
        "limit_threshold: 0.095":        "limit_threshold: null",  # CN circuit-breaker, not applicable to US
        "train: [2008-01-01, 2014-12-31]": "train: [2015-01-01, 2019-12-31]",
        "valid: [2015-01-01, 2016-12-31]": "valid: [2020-01-01, 2021-12-31]",
        "test: [2017-01-01, 2020-08-01]":  "test: [2022-01-01, 2023-12-31]",
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


# ── Patch 4 (Windows): run LocalEnv's code sandbox natively on Windows ─────────
#
# RD-Agent's LocalEnv.run / _run assume a POSIX host and fail on native Windows:
#   (a) volume mounts use os.symlink        -> WinError 1314 without privilege
#   (b) every entry is wrapped in
#         /bin/sh -c 'timeout --kill-after=10 N {entry}; ...'
#       but Popen(shell=True) on Windows uses cmd.exe, which has no /bin/sh or
#       GNU `timeout`.
#   (c) PATH is rebuilt by splitting/joining on ':' (env.py line ~524). On Windows
#       that splits "C:\..." at the drive colon AND overrides the real Windows PATH,
#       so `python` / `qrun` can no longer be found.
#   (d) live output uses select.poll(), which does not exist on Windows.
#
# Fixes below (win32 only): override run() to drop the POSIX sh/timeout wrapper,
# and override _run() to build PATH with os.pathsep and capture output via
# communicate() (no poll). Symlinks themselves are handled by enabling Windows
# Developer Mode (grants SeCreateSymbolicLinkPrivilege to non-admin users).
# The volume-map construction below is copied verbatim from LocalEnv._run so the
# mount semantics stay identical.
#
if sys.platform == "win32":
    import subprocess as _subprocess
    import rdagent.utils.env as _env_mod
    from rdagent.utils.env import LocalEnv as _LocalEnv

    def _win_local_run(self, entry=None, local_path=None, env=None,
                       running_extra_volume=_env_mod.MappingProxyType({}), **kwargs):
        # --- build volume map (verbatim from upstream LocalEnv._run) ---
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = _env_mod.T("scenarios.data_science.share:scen.cache_path").r()
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp
        assert local_path is not None, "local_path should not be None"
        volumes = _env_mod.normalize_volumes(volumes, local_path)

        created = []
        try:
            # symlink the volumes (works once Developer Mode is enabled)
            for real, link in volumes.items():
                link_path = Path(link)
                real_path = Path(real)
                if not link_path.parent.exists():
                    link_path.parent.mkdir(parents=True, exist_ok=True)
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(real_path)
                created.append(link_path)

            # Windows PATH: prepend the venv bin dir to the REAL PATH (os.pathsep),
            # never split on ':'.
            run_env = {**os.environ}
            if env:
                run_env.update({k: str(v) for k, v in env.items()})
            if self.conf.bin_path:
                run_env["PATH"] = self.conf.bin_path + os.pathsep + os.environ.get("PATH", "")

            if entry is None:
                entry = self.conf.default_entry
            cwd = Path(local_path).resolve() if local_path else None

            proc = _subprocess.Popen(
                entry, cwd=cwd, env=run_env,
                stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
                text=True, shell=True,
                encoding="utf-8", errors="replace",
            )
            timeout = self.conf.running_timeout_period
            try:
                out, err = proc.communicate(timeout=timeout)
            except _subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                combined = (out or "") + (err or "")
                combined += f"\n\nThe running time exceeds {timeout} seconds, so the process is killed."
                print(combined)
                return combined, 1
            combined = (out or "") + (err or "")
            print(combined)
            return combined, proc.returncode
        finally:
            for p in created:
                try:
                    if p.is_symlink():
                        p.unlink()
                    elif p.exists():
                        try:
                            p.unlink()
                        except (IsADirectoryError, PermissionError, OSError):
                            os.rmdir(p)
                except FileNotFoundError:
                    pass

    def _win_run(self, entry=None, local_path=".", env=None, **kwargs):
        running_extra_volume = kwargs.get("running_extra_volume", {})
        if entry is None:
            entry = self.conf.default_entry
        # Skip the POSIX '/bin/sh -c "timeout ... {entry}"' wrapper; run bare entry.
        if self.conf.enable_cache:
            return self.cached_run(entry, local_path, env, running_extra_volume)
        _retry = next(n for n in dir(self) if n.endswith("__run_with_retry"))
        return getattr(self, _retry)(entry, local_path, env, running_extra_volume)

    _LocalEnv._run = _win_local_run
    _LocalEnv.run = _win_run


# ── Now safe to import the scenario and loop ───────────────────────────────────
import asyncio
from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING

# Entry-point guard: on Windows, multiprocessing uses 'spawn', which re-imports this
# module in every worker. Without this guard the loop would relaunch recursively (fork
# bomb). On Linux ('fork') it is a harmless no-op. The module-level monkey-patches above
# are intentionally left unguarded so spawned workers inherit the patched environment.
if __name__ == "__main__":
    # Force UTF-8 stdout/stderr: banner and loguru logs contain non-cp1252 chars (—, ×, →)
    # that crash on the default Windows console codepage.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 60)
    print("RD-Agent fin_factor — autonomous alpha factor discovery")
    print(f"Model : {os.environ.get('CHAT_MODEL', 'not set')}")
    print(f"Config: {FACTOR_PROP_SETTING}")
    print("=" * 60)

    # To resume a previous session, replace FactorRDLoop(FACTOR_PROP_SETTING) with:
    #   loop = FactorRDLoop.load("log/<session-dir>", checkout=True)
    # loop_n counts *total* loops completed in the session, not additional loops.
    loop = FactorRDLoop(FACTOR_PROP_SETTING)
    loop_n = int(os.environ.get("RDAGENT_LOOP_N", "10"))
    asyncio.run(loop.run(loop_n=loop_n))
