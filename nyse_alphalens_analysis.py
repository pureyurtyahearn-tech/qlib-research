"""Alphalens tearsheet for the NYSE SOTA factor (loop 24's trained model output).

combined_factors_df.parquet has no single "score" column -- it holds the 33 raw
custom factor INPUTS (momentum_5d, earnings_yield, ..., value_momentum_interaction_rank).
The actual SOTA factor -- the thing that ranks BUY/SELL everywhere else in this repo
(nyse_daily_signal.py, nx_wf_walkforward.py, the paper tracker) -- is loop 24's trained
LightGBM model's PREDICTION on top of these 33 factors plus Alpha158's 20 base features.
Alphalens tests one factor at a time, so this script reconstructs that same historical
score panel (Alpha158 20 + these 33 custom columns, scored with the SAVED model --
no retraining, no walk-forward folds) and feeds the model's output to Alphalens as the
single factor under test. Confirmed with the user before building.

Date range: 2010-01-01 through the factor panel's last date (2026-06-29), matching
nx_wf_walkforward.py's validated window. The full custom-factor history goes back to
1999, but running Alphalens's quantile bucketing / IC / turnover machinery over ~27
years x 1100+ names adds a lot of runtime for no analytical benefit over the already-
validated 2010+ window.

Forward returns (1d/5d/10d) are NOT precomputed here -- Alphalens' own
get_clean_factor_and_forward_returns() computes them directly from a wide close-price
panel pulled from the qlib store. That's the standard, correct way to use this API
(passing precomputed returns instead of prices is not what it expects).
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.tears import create_full_tear_sheet

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
SOTA_WORKSPACE = "c3955146822249b6b195e5c4e084de5a"   # loop 24
OUT_DIR = Path("git_ignore_folder/alphalens_output")
START = "2010-01-01"
PERIODS = (1, 5, 10)
QUANTILES = 5

# Alpha158's 20 base features, exact expressions -- copied verbatim from
# nx_wf_walkforward.py (same set used to train/validate loop 24's model).
ALPHA158_EXPR = {
    'RESI5': 'Resi($close, 5)/$close',
    'WVMA5': 'Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)',
    'RSQR5': 'Rsquare($close, 5)',
    'KLEN': '($high-$low)/$open',
    'RSQR10': 'Rsquare($close, 10)',
    'CORR5': 'Corr($close, Log($volume+1), 5)',
    'CORD5': 'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)',
    'CORR10': 'Corr($close, Log($volume+1), 10)',
    'ROC60': 'Ref($close, 60)/$close',
    'RESI10': 'Resi($close, 10)/$close',
    'VSTD5': 'Std($volume, 5)/($volume+1e-12)',
    'RSQR60': 'Rsquare($close, 60)',
    'CORR60': 'Corr($close, Log($volume+1), 60)',
    'WVMA60': 'Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)',
    'STD5': 'Std($close, 5)/$close',
    'RSQR20': 'Rsquare($close, 20)',
    'CORD60': 'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)',
    'CORD10': 'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)',
    'CORR20': 'Corr($close, Log($volume+1), 20)',
    'KLOW': '(Less($open, $close)-$low)/$open',
}


def load_model():
    ws = SH.parent / "RD-Agent_workspace" / SOTA_WORKSPACE
    candidates = list(ws.glob("mlruns/*/*/artifacts/params.pkl"))
    if not candidates:
        raise SystemExit(f"FAIL: no saved model found under {ws}/mlruns/. "
                          f"Refusing to proceed without the real loop 24 model.")
    with open(candidates[0], "rb") as f:
        model = pickle.load(f)
    print(f"OK: model loaded from {candidates[0]}")
    print(f"    {type(model).__name__}, {model.model.num_trees()} trees, "
          f"{len(model.model.feature_name())} features")
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()

    custom = pd.read_parquet(
        f"git_ignore_folder/RD-Agent_workspace/{SOTA_WORKSPACE}/combined_factors_df.parquet")
    custom.columns = [c[1] for c in custom.columns]
    custom_cols = list(custom.columns)
    print(f"custom factors ({len(custom_cols)}) loaded from combined_factors_df.parquet")

    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(STORE), region="us", kernels=1)

    last_factor_date = custom.index.get_level_values("datetime").max()
    codes = D.list_instruments(D.instruments(market="nyse"),
                                start_time=START, end_time=str(last_factor_date.date()),
                                as_list=True)
    print(f"NYSE instruments, {START} -> {last_factor_date.date()}: {len(codes)}")

    # ---- Alpha158 base features over the same window ----
    names20 = list(ALPHA158_EXPR)
    alpha = D.features(codes, list(ALPHA158_EXPR.values()),
                        start_time=START, end_time=str(last_factor_date.date()))
    alpha.columns = names20
    alpha.index = alpha.index.set_names(["instrument", "datetime"])
    alpha = alpha.reset_index().set_index(["datetime", "instrument"]).sort_index()
    print(f"Alpha158 base panel: {alpha.shape}")

    # ---- merge with the 33 custom factors, score with the SAVED loop-24 model ----
    panel = alpha.join(custom, how="inner")
    panel = panel[panel.index.get_level_values("instrument").isin(set(codes))]
    feat_cols = names20 + custom_cols
    panel = panel.dropna(subset=feat_cols)
    print(f"merged scoring panel (post-dropna): {panel.shape}  ({len(feat_cols)} features)")

    X = panel[feat_cols]
    scores = model.model.predict(X)
    factor = pd.Series(scores, index=panel.index, name="factor")
    print(f"factor (model predicted_score) computed for {len(factor)} (date, instrument) rows")

    # ---- wide close-price panel for Alphalens' own forward-return computation ----
    # Extended past the last factor date to cover the longest lookahead period (10d).
    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    cal_ts = pd.DatetimeIndex(cal)
    last_idx = int(cal_ts.searchsorted(last_factor_date))
    price_end = cal_ts[min(last_idx + max(PERIODS) + 2, len(cal_ts) - 1)]
    print(f"price panel through {price_end.date()} "
          f"(factor data ends {last_factor_date.date()}, +{max(PERIODS)}d buffer)")

    px = D.features(codes, ["$close"], start_time=START, end_time=str(price_end.date()))
    px.columns = ["close"]
    px.index = px.index.set_names(["instrument", "datetime"])
    prices = px.reset_index().pivot(index="datetime", columns="instrument", values="close")
    print(f"price panel: {prices.shape}")

    # ---- sector groupby, from Sharadar TICKERS (same source/method as the earlier
    # BUY-list sector-breakdown task) ----
    tickers = pd.read_csv(SH / "tickers.csv", low_memory=False)
    sector_series = tickers.drop_duplicates("ticker").set_index("ticker")["sector"]
    inst_set = sorted(factor.index.get_level_values("instrument").unique())
    sector_map = sector_series.reindex(inst_set).dropna().to_dict()
    print(f"sector mapping: {len(sector_map)}/{len(inst_set)} instruments matched to a "
          f"Sharadar sector (unmatched names are dropped by Alphalens' groupby)")

    print("\nrunning get_clean_factor_and_forward_returns "
          f"(periods={PERIODS}, quantiles={QUANTILES}) ...")
    factor_data = get_clean_factor_and_forward_returns(
        factor, prices, groupby=sector_map, quantiles=QUANTILES, periods=PERIODS,
        max_loss=0.5,
    )
    print(f"clean factor_data: {factor_data.shape}")

    # alphalens' GridFigure explicitly plt.close()s each section's figure once it's
    # done with it (built for inline Jupyter display, not headless scripts) -- by the
    # time create_full_tear_sheet() returns, most figures are already gone from
    # plt.get_fignums(). Capture each Figure object at CREATION time instead, by
    # wrapping plt.figure(): a closed figure is still fully renderable/saveable as
    # long as something holds a reference to it.
    captured_figs = []
    _orig_figure = plt.figure

    def _capturing_figure(*args, **kwargs):
        fig = _orig_figure(*args, **kwargs)
        captured_figs.append(fig)
        return fig

    plt.figure = _capturing_figure
    print("\ngenerating full tear sheet (quantile stats + returns + information + "
          "turnover, broken out by sector) ...")
    try:
        create_full_tear_sheet(factor_data, long_short=True, group_neutral=False, by_group=True)
    finally:
        plt.figure = _orig_figure

    # create_full_tear_sheet() calls its four sections in a fixed order (quantile
    # stats table, returns, information, turnover), so position reliably identifies
    # each figure -- more readable than the first subplot's title (which understates
    # what's actually in a composite, multi-panel figure).
    section_labels = [
        "quantile_statistics_table",
        "returns_tearsheet_quantiles_cumulative",
        "returns_by_sector_all_groups",
        "information_coefficient_ts_hist_qq_bysector",
        "turnover_and_factor_rank_autocorrelation",
    ]
    print(f"\n{len(captured_figs)} figures generated, saving to {OUT_DIR}/")
    for i, fig in enumerate(captured_figs, start=1):
        label = section_labels[i - 1] if i - 1 < len(section_labels) else f"figure_{i}"
        out_path = OUT_DIR / f"{i:02d}_{label}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  saved {out_path.name}")

    print(f"\ndone. {len(captured_figs)} plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
