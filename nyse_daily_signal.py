"""Daily signal generator for the NYSE SOTA factor model (loop 24) -- step 1 of paper
trading. Reads only what's already on disk: no live data pulls, no broker connection.

  - loop 24's trained LGBModel, loaded from its saved mlruns artifact
  - loop 24's own combined_factors_df.parquet for the 33 accepted custom factors (the
    literal training data, not re-derived -- avoids any transcription mismatch)
  - Alpha158's 20 base features, computed from the qlib store (sep_nyse_panel.h5)

Scores the LATEST available date in the data (printed clearly -- this is the store's
last trading day, not necessarily today's calendar date), ranks by predicted score, and
tiers top-50/bottom-50/rest as BUY/SELL/HOLD.

FEATURE ORDER WARNING: the saved booster has no column names (Column_0..Column_52) --
order must exactly match training. qlib's FilterCol processor masks Alpha158's ~158
features by name WITHOUT reordering, so the 20 base features must be fed in Alpha158's
own internal generation order, not the order they're listed in run_rdagent.py's YAML
patch. Verified directly against Alpha158.get_feature_config() -- see ALPHA158_ORDER
below. Getting this wrong would silently misalign every prediction with no error raised.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
SOTA_WORKSPACE = "c3955146822249b6b195e5c4e084de5a"   # loop 24
N_TOP = 50
MIN_PRICE = 5.0   # names below this are excluded from BUY/SELL, forced to HOLD

# True Alpha158 internal generation order for our 20 base features (verified against
# Alpha158.get_feature_config(), NOT the col_list order in run_rdagent.py's YAML patch).
ALPHA158_ORDER = ['KLEN', 'KLOW', 'ROC60', 'STD5', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR60',
                  'RESI5', 'RESI10', 'CORR5', 'CORR10', 'CORR20', 'CORR60', 'CORD5',
                  'CORD10', 'CORD60', 'VSTD5', 'WVMA5', 'WVMA60']
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


def check_and_load_model():
    """Step 1, per instruction: verify the model is actually on disk and readable before
    building anything around it. Exits loudly if not -- does not proceed on a missing file."""
    ws = SH.parent / "RD-Agent_workspace" / SOTA_WORKSPACE
    if not ws.exists():
        print(f"FAIL: loop 24 workspace not found: {ws}")
        sys.exit(1)
    candidates = list(ws.glob("mlruns/*/*/artifacts/params.pkl"))
    if not candidates:
        print(f"FAIL: no saved model (params.pkl) found under {ws}/mlruns/. "
              f"Refusing to build a signal generator around a missing model file.")
        sys.exit(1)
    model_path = candidates[0]
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        booster = model.model
        n_feat = len(booster.feature_name())
        n_trees = booster.num_trees()
    except Exception as e:
        print(f"FAIL: model file exists at {model_path} but could not be loaded: {e!r}")
        sys.exit(1)
    print(f"OK: model loaded from {model_path}")
    print(f"    {type(model).__name__}, {n_trees} trees, {n_feat} features")
    if n_feat != 53:
        print(f"WARNING: expected 53 features (20 Alpha158 + 33 custom), got {n_feat}. "
              f"Feature-order assumptions below may not hold for this model -- verify before trusting output.")
    return model


def main():
    model = check_and_load_model()

    custom = pd.read_parquet(f"git_ignore_folder/RD-Agent_workspace/{SOTA_WORKSPACE}/combined_factors_df.parquet")
    custom.columns = [c[1] for c in custom.columns]
    custom_cols = list(custom.columns)
    print(f"custom factors ({len(custom_cols)}): loaded from loop 24's combined_factors_df.parquet")

    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(STORE), region="us", kernels=1)

    # Latest date from the STORE's own calendar (not the static sep_nyse_panel.h5 snapshot,
    # which is not touched by yfinance price extensions -- reading from that file would
    # silently report a stale date after the store itself has been extended).
    cal = [l.strip() for l in open(STORE / "calendars" / "day.txt") if l.strip()]
    latest_date = pd.Timestamp(cal[-1])
    print(f"\nlatest available date in the qlib store calendar: {latest_date.date()}")
    print("(this is the store's last trading day, not necessarily today's calendar date --"
          " no live data pull happens in THIS script; reflects whatever extension has already run)")

    codes = D.list_instruments(D.instruments(market="nyse"),
                               start_time=str(latest_date.date()), end_time=str(latest_date.date()),
                               as_list=True)
    print(f"active NYSE names (market='nyse' on {latest_date.date()}): {len(codes)}")

    # Alpha158 base features: query a lookback window (covers the 60d rolling ops),
    # keep only the latest row per instrument.
    lookback_start = (latest_date - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    alpha = D.features(codes, [ALPHA158_EXPR[n] for n in ALPHA158_ORDER],
                       start_time=lookback_start, end_time=str(latest_date.date()))
    alpha.columns = ALPHA158_ORDER
    alpha.index = alpha.index.set_names(["instrument", "datetime"])
    alpha = alpha.reset_index()
    alpha_latest = alpha.sort_values("datetime").groupby("instrument").tail(1).set_index("instrument")

    # Custom factors: latest row per instrument from loop 24's own factor panel. This file
    # is NOT touched by price-only extensions (e.g. nx_extend_nyse_yfinance.py) -- if the
    # store's price data is newer than this panel's last date, these 33 factors are stale
    # (last-known-value carried forward) even though price/Alpha158 below are current.
    custom_max_date = custom.index.get_level_values(0).max()
    if custom_max_date < latest_date:
        print(f"\nWARNING: custom-factor panel's last date is {custom_max_date.date()}, "
              f"{(latest_date - custom_max_date).days} days behind the store's {latest_date.date()}. "
              f"The 33 custom factors below use last-known (stale) values; only price and "
              f"Alpha158 base features are current as of {latest_date.date()}.")
    custom_reset = custom.reset_index()
    custom_reset.columns = ["datetime", "instrument"] + custom_cols
    custom_latest = (custom_reset[custom_reset["instrument"].isin(codes)]
                     .sort_values("datetime").groupby("instrument").tail(1).set_index("instrument"))

    # Current price (latest close per instrument), from the store -- not the static h5.
    price_df = D.features(codes, ["$close"], start_time=str(latest_date.date()),
                          end_time=str(latest_date.date()))
    price_df.columns = ["current_price"]
    price_df.index = price_df.index.set_names(["instrument", "datetime"])
    price_latest = price_df.reset_index().set_index("instrument")["current_price"]

    feat = alpha_latest[ALPHA158_ORDER].join(custom_latest[custom_cols], how="inner")
    feat = feat.join(price_latest, how="inner")
    feat = feat[feat["current_price"].notna()]
    print(f"\nnames with both features and a current price: {len(feat)}")

    X = feat[ALPHA158_ORDER + custom_cols]
    pred = model.model.predict(X)
    feat["predicted_score"] = pred

    feat.index = feat.index.rename("ticker")
    ranked = feat.sort_values("predicted_score", ascending=False).reset_index()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    n = len(ranked)

    # $5 price floor: names below it are ineligible for BUY/SELL regardless of model
    # score -- reclassified HOLD. Tiering (top/bottom N_TOP) is computed only among
    # eligible names, so a sub-$5 name at the top of the ranking doesn't bump an
    # otherwise-qualifying eligible name out of the BUY/SELL lists.
    eligible = ranked["current_price"] >= MIN_PRICE
    n_filtered = int((~eligible).sum())
    elig_idx = ranked.index[eligible]

    ranked["signal"] = "HOLD"
    ranked.loc[elig_idx[:N_TOP], "signal"] = "BUY"
    ranked.loc[elig_idx[-N_TOP:], "signal"] = "SELL"

    out = ranked[["ticker", "rank", "predicted_score", "current_price", "signal"]]
    date_str = latest_date.strftime("%Y%m%d")
    out_path = f"nyse_signal_{date_str}.csv"
    out.to_csv(out_path, index=False)

    buy_out = out[out.signal == "BUY"].sort_values("predicted_score", ascending=False)
    sell_out = out[out.signal == "SELL"].sort_values("predicted_score", ascending=True)

    print(f"\n{'='*60}")
    print(f"NYSE DAILY SIGNAL -- {latest_date.date()}")
    print(f"{'='*60}")
    print(f"names scored: {n}   BUY: {len(buy_out)}   "
          f"SELL: {len(sell_out)}   HOLD: {(out.signal=='HOLD').sum()}")
    print(f"filtered out by ${MIN_PRICE:.0f} price floor (forced HOLD): {n_filtered}")
    print(f"\nTop 10 BUY:")
    print(buy_out.head(10).to_string(index=False))
    print(f"\nTop 10 SELL (most bearish):")
    print(sell_out.head(10).to_string(index=False))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
