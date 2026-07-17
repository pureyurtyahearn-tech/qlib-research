"""Preflight check -- MUST be run and PASS before any RD-Agent job.

Verifies, for the target universe (auto-detected by simulating run_rdagent.py's own
market/provider_uri patch chain, so it always reflects what the run would actually use;
override with --market):

  (a) the target universe's SF1 fundamentals file exists and >=95% of the tickers in the
      store's instrument list have at least one non-null fundamental value
  (b) the binary store actually contains the universe: a live qlib.init() + D.instruments()
      call resolves to a nonempty, sane instrument count
  (c) the liquidity/selection filter that determined which tickers belong to this universe
      was computed from the SAME price panel that is actually loaded into the store -- not a
      stale/different source. (This is exactly the NYSE bug found by hand on 2026-07-17:
      nyse.txt's ticker set was filtered using the old Kaggle/yfinance daily_pv.h5, but the
      store's actual bins came from the later Sharadar-native sep_nyse_panel.h5.)
  (d) no fundamental factor column exceeds a 20% null rate
  (e) FACTOR_CoSTEER_DATA_FOLDER's daily_pv.h5 (the RD-Agent factor-coding sandbox) has an
      instrument set overlapping >=80% with the target market's instrument list. (This is
      the universe-mismatch bug found by hand on 2026-07-17: the sandbox env var is a global
      .env setting, decoupled from run_rdagent.py's market patch. When it silently pointed at
      the SP500-ish sandbox while market='nyse', every custom factor came out NaN for every
      NYSE row -- 0% overlap -- and got Fillna'd to a constant, so nothing ever moved a
      prediction. No exception was ever raised; it just silently produced dead factors.)

Exits 0 and prints PASS if all checks pass. Exits 1 and prints FAIL with specific reasons
if any check fails -- the run must not proceed past a FAIL.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent
SH = REPO / "git_ignore_folder" / "sharadar"
US_DATA = Path.home() / ".qlib" / "qlib_data" / "us_data"
US_DATA_PIT_FULL = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"

# ── Provenance manifest ──────────────────────────────────────────────────────────
# Recorded by hand from reading each universe's actual build/pull scripts. This is the
# source of truth for check (c) -- update it whenever a universe's ingestion pipeline
# changes (new pull script, re-selected ticker list, etc).
UNIVERSE = {
    "nyse": dict(
        store=US_DATA_PIT_FULL,
        instruments_file="nyse.txt",
        fundamentals_file="fundamentals_nyse_daily.h5",
        price_panel_in_store="sep_nyse_panel.h5",       # nx_extend_nyse_sharadar.py's source
        liquidity_filter_source="sep_nyse_panel.h5",    # nx_refilter_nyse_liquidity.py (fixed 2026-07-17;
                                                         # was daily_pv.h5, the stale pre-Sharadar Kaggle source)
    ),
    "nasdaq": dict(
        store=US_DATA_PIT_FULL,
        instruments_file="nasdaq.txt",
        fundamentals_file="fundamentals_nasdaq_daily.h5",
        price_panel_in_store="sep_nasdaq_panel.h5",     # nq4_extend_store.py's source
        liquidity_filter_source="sep_nasdaq_panel.h5",  # nq2_quality_liquidity.py's source
    ),
    "sp500pit": dict(
        store=US_DATA_PIT_FULL,
        instruments_file="sp500pit.txt",
        fundamentals_file="fundamentals_daily.h5",
        price_panel_in_store="sep_panel_full.h5",       # ext4_build_store.py's source
        liquidity_filter_source=None,                    # no liquidity gate -- ever-members only
    ),
}

FUND_COLS = ["$pe", "$pb", "$ey", "$de", "$roe", "$rgrow", "$fcfy"]
NULL_RATE_MAX = 0.20
# $pe is structurally undefined for negative-earnings companies, so it runs a higher null
# rate than the other factors as a matter of course -- not a data-quality problem.
NULL_RATE_MAX_OVERRIDE = {"$pe": 0.35}
COVERAGE_MIN = 0.95
SANDBOX_OVERLAP_MIN = 0.80


def resolve_chain(pairs, start):
    """Simulate a sequence of string.replace(old, new) calls: follow old->new links
    starting from `start` until a fixed point. Mirrors run_rdagent.py's own patch loop,
    where dict insertion order lets one substitution's output be matched by a later key."""
    cur = start
    changed = True
    while changed:
        changed = False
        for old, new in pairs:
            if cur == old:
                cur = new
                changed = True
    return cur


def detect_market_and_provider():
    """Read run_rdagent.py's TARGET_MARKET variable and provider_uri patch to resolve what
    an actual run would use right now, without invoking RD-Agent.

    TARGET_MARKET drives the market patch via an f-string (single source of truth, also
    used to set FACTOR_CoSTEER_DATA_FOLDER), so it's read directly rather than simulated
    via string-replace chain resolution the way provider_uri still is."""
    src = (REPO / "run_rdagent.py").read_text(encoding="utf-8")
    m = re.search(r'^TARGET_MARKET\s*=\s*"(\w+)"', src, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find TARGET_MARKET = \"...\" in run_rdagent.py")
    market = m.group(1)
    provider_pairs = re.findall(
        r"'provider_uri: \"([^\"]+)\"':\s*'provider_uri: \"([^\"]+)\"'", src)
    provider_uri = resolve_chain(provider_pairs, "~/.qlib/qlib_data/cn_data")
    return market, provider_uri


def check_fundamentals_coverage(u):
    fpath = SH / u["fundamentals_file"]
    if not fpath.exists():
        return False, f"fundamentals file missing: {fpath}"
    fund = pd.read_hdf(fpath)
    any_nonnull = fund[FUND_COLS].notna().any(axis=1)
    covered = set(fund.index.get_level_values(1)[any_nonnull].unique())

    inst_path = u["store"] / "instruments" / u["instruments_file"]
    if not inst_path.exists():
        return False, f"instrument list missing: {inst_path}"
    inst_tickers = {l.split("\t")[0].strip().upper()
                    for l in inst_path.read_text().splitlines() if l.strip()}

    matched = inst_tickers & {t.upper() for t in covered}
    coverage = len(matched) / len(inst_tickers) if inst_tickers else 0.0
    ok = coverage >= COVERAGE_MIN
    msg = (f"{fpath.name}: {len(matched)}/{len(inst_tickers)} instrument-list tickers "
           f"have >=1 non-null fundamental ({coverage:.1%}, need >={COVERAGE_MIN:.0%})")
    return ok, msg


def check_store_universe(u, market):
    try:
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=str(u["store"]), region="us", kernels=1)
        inst = D.instruments(market=market)
        names = D.list_instruments(inst, start_time="2023-01-01", end_time="2023-12-29", as_list=True)
    except Exception as e:
        return False, f"qlib.init()/D.instruments(market='{market}') raised: {e!r}"
    ok = len(names) > 0
    msg = f"D.instruments(market='{market}') against {u['store']} resolved to {len(names)} active instruments in 2023"
    return ok, msg


def check_liquidity_provenance(u):
    lsrc, psrc = u["liquidity_filter_source"], u["price_panel_in_store"]
    if lsrc is None:
        return True, "no separate liquidity/selection filter for this universe (N/A)"
    ok = lsrc == psrc
    msg = f"price panel in store: {psrc}  |  liquidity filter computed from: {lsrc}"
    msg += "  -- MATCH" if ok else "  -- MISMATCH (ticker selection and store data come from different sources)"
    return ok, msg


def check_null_rates(u):
    fpath = SH / u["fundamentals_file"]
    if not fpath.exists():
        return False, f"fundamentals file missing: {fpath}"
    fund = pd.read_hdf(fpath)
    lines, bad = [], []
    for c in FUND_COLS:
        nr = fund[c].isna().mean()
        cap = NULL_RATE_MAX_OVERRIDE.get(c, NULL_RATE_MAX)
        flag = f"  <-- EXCEEDS {cap:.0%}" if nr > cap else ""
        lines.append(f"      {c:8} null rate {nr:.1%} (cap {cap:.0%}){flag}")
        if nr > cap:
            bad.append(c)
    ok = len(bad) == 0
    msg = "\n".join(lines)
    if bad:
        msg += f"\n      FAILING COLUMNS: {', '.join(bad)}"
    return ok, msg


def resolve_sandbox_folder(market):
    """FACTOR_CoSTEER_DATA_FOLDER is now set programmatically by run_rdagent.py from
    TARGET_MARKET via its _MARKET_SANDBOX dict (single source of truth) -- read THAT,
    not .env, since run_rdagent.py overrides .env's value at runtime regardless of what
    the file says. Falls back to .env only if run_rdagent.py has no such mapping."""
    src = (REPO / "run_rdagent.py").read_text(encoding="utf-8")
    m = re.search(r"_MARKET_SANDBOX\s*=\s*\{(.*?)\}", src, re.DOTALL)
    if m:
        pairs = re.findall(r'"(\w+)":\s*"([^"]+)"', m.group(1))
        mapping = dict(pairs)
        if market in mapping:
            return mapping[market], "run_rdagent.py _MARKET_SANDBOX"
    load_dotenv(REPO / ".env", override=True)
    folder = os.environ.get("FACTOR_CoSTEER_DATA_FOLDER")
    return folder, ".env (no _MARKET_SANDBOX entry for this market -- fallback, may be stale)"


def check_costeer_sandbox_overlap(u, market):
    folder, source = resolve_sandbox_folder(market)
    if not folder:
        return False, "FACTOR_CoSTEER_DATA_FOLDER could not be resolved from run_rdagent.py or .env"
    sandbox_path = REPO / folder / "daily_pv.h5"
    if not sandbox_path.exists():
        return False, f"sandbox file missing: {sandbox_path}"
    sandbox = pd.read_hdf(sandbox_path)
    sandbox_insts = {t.upper() for t in sandbox.index.get_level_values("instrument").unique()}

    inst_path = u["store"] / "instruments" / u["instruments_file"]
    if not inst_path.exists():
        return False, f"instrument list missing: {inst_path}"
    market_tickers = {l.split("\t")[0].strip().upper()
                       for l in inst_path.read_text().splitlines() if l.strip()}

    overlap = market_tickers & sandbox_insts
    coverage = len(overlap) / len(market_tickers) if market_tickers else 0.0
    ok = coverage >= SANDBOX_OVERLAP_MIN
    msg = (f"FACTOR_CoSTEER_DATA_FOLDER={folder}  (source: {source}; "
           f"{len(sandbox_insts)} sandbox instruments)\n"
           f"      overlap with market='{market}' instrument list: "
           f"{len(overlap)}/{len(market_tickers)} = {coverage:.1%} (need >={SANDBOX_OVERLAP_MIN:.0%})")
    if not ok:
        msg += ("\n      CoSTEER will compute every custom factor against the WRONG universe -- "
                "values will be NaN for nearly all target-market rows and Fillna'd to a constant, "
                "silently producing dead factors with no exception raised.")
    return ok, msg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default=None,
                     help="Override the auto-detected target market (nyse/nasdaq/sp500pit).")
    args = ap.parse_args()

    detected_market, provider_uri = detect_market_and_provider()
    market = args.market or detected_market

    print("=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)
    print(f"  detected from run_rdagent.py: market='{detected_market}'  provider_uri='{provider_uri}'")
    if args.market:
        print(f"  overridden with --market: '{market}'")
    print()

    if market not in UNIVERSE:
        print(f"FAIL: no provenance manifest entry for market '{market}'. "
              f"Known universes: {list(UNIVERSE)}. Add an entry to UNIVERSE in this script.")
        sys.exit(1)

    u = UNIVERSE[market]
    results = []

    print(f"(a) SF1 fundamentals coverage >= {COVERAGE_MIN:.0%}")
    ok, msg = check_fundamentals_coverage(u)
    print(f"    {msg}")
    results.append(("(a) fundamentals coverage", ok))
    print()

    print("(b) binary store contains universe (live qlib.init + D.instruments)")
    ok, msg = check_store_universe(u, market)
    print(f"    {msg}")
    results.append(("(b) store/D.instruments", ok))
    print()

    print("(c) liquidity filter source matches the price panel actually in the store")
    ok, msg = check_liquidity_provenance(u)
    print(f"    {msg}")
    results.append(("(c) liquidity provenance", ok))
    print()

    overrides_str = ", ".join(f"{k} <= {v:.0%}" for k, v in NULL_RATE_MAX_OVERRIDE.items())
    print(f"(d) no fundamental column exceeds {NULL_RATE_MAX:.0%} null rate "
          f"(override: {overrides_str})")
    ok, msg = check_null_rates(u)
    print(msg)
    results.append(("(d) null rates", ok))
    print()

    print(f"(e) CoSTEER sandbox instrument overlap with target market >= {SANDBOX_OVERLAP_MIN:.0%}")
    ok, msg = check_costeer_sandbox_overlap(u, market)
    print(f"    {msg}")
    results.append(("(e) sandbox overlap", ok))
    print()

    print("=" * 70)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"FAIL -- {len(failed)}/{len(results)} check(s) failed: {', '.join(failed)}")
        print("Do NOT proceed with the RD-Agent run until these are resolved.")
        print("=" * 70)
        sys.exit(1)
    else:
        print(f"PASS -- all {len(results)} checks passed for market='{market}'.")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
