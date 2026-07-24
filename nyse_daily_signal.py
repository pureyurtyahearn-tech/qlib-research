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
MIN_PRICE = 5.0   # BUY-eligibility floor -- names below this are never tagged BUY
LONG_ONLY = True  # when True, suppress SELL signals entirely (BUY and HOLD only)
VIX_THRESHOLD = 25.0  # regime filter: VIX >= this suppresses all BUY signals for the day

# Sector filter, BUY-eligibility only (same treatment as MIN_PRICE). Per the Alphalens
# tearsheet (nyse_alphalens_analysis.py, 2010-2026): these three sectors show no usable
# signal in the quantile-return breakdown --
#   Energy: actively INVERTED -- Q5 (highest predicted score) is the most NEGATIVE
#     quantile across all three horizons (1D/5D/10D), the opposite of what the factor
#     is supposed to predict.
#   Communication Services: negative mean return across every quantile (1 through 5) --
#     no separation, and no quantile is actually profitable.
#   Consumer Defensive: no monotonic rank ordering between quantiles -- the factor
#     carries no information here.
# Industrials, Technology, Financial Services, and Consumer Cyclical showed clean
# monotonic Q1->Q5 separation and are retained. Sectors from Sharadar TICKERS.sector.
SECTOR_EXCLUDE = {"Energy", "Communication Services", "Consumer Defensive"}

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


def get_current_vix():
    """Single check before the model runs: latest VIX close via yfinance."""
    import yfinance as yf
    vix = yf.download("^VIX", period="5d", progress=False, auto_adjust=False)
    return float(vix["Close"].iloc[-1].iloc[0])


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
    current_vix = get_current_vix()
    regime_filter_active = current_vix >= VIX_THRESHOLD
    print(f"VIX check: {current_vix:.2f} (threshold {VIX_THRESHOLD:.0f})")
    if regime_filter_active:
        print("REGIME FILTER ACTIVE — VIX above 25, no new longs today.")

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

    # Sector lookup -- same source/method as the earlier BUY-list sector-concentration
    # check: Sharadar TICKERS.sector, deduplicated to one row per ticker.
    tickers_tbl = pd.read_csv(SH / "tickers.csv", low_memory=False)
    sector_map = tickers_tbl.drop_duplicates("ticker").set_index("ticker")["sector"]
    ranked["sector"] = ranked["ticker"].map(sector_map)

    # $5 price floor and sector exclusion both apply to BUY eligibility only: names
    # failing either are never tagged BUY (reclassified HOLD), regardless of model
    # score. BUY tiering (top N_TOP) is computed only among eligible names, so an
    # ineligible name at the top of the ranking doesn't bump an otherwise-qualifying
    # eligible name out of the BUY list. SELL tiering (bottom N_TOP by rank) is
    # unaffected by either filter -- moot while LONG_ONLY suppresses SELL entirely,
    # but kept correct in case LONG_ONLY is later set False.
    eligible_price = ranked["current_price"] >= MIN_PRICE
    eligible_sector = ~ranked["sector"].isin(SECTOR_EXCLUDE)
    eligible_buy = eligible_price & eligible_sector
    n_filtered_price = int((~eligible_price).sum())
    n_filtered_sector = int((eligible_price & ~eligible_sector).sum())
    elig_idx = ranked.index[eligible_buy]

    ranked["signal"] = "HOLD"
    if not regime_filter_active:
        ranked.loc[elig_idx[:N_TOP], "signal"] = "BUY"
        if not LONG_ONLY:
            ranked.loc[ranked.index[-N_TOP:], "signal"] = "SELL"

    out = ranked[["ticker", "rank", "predicted_score", "current_price", "sector", "signal"]]
    date_str = latest_date.strftime("%Y%m%d")
    out_path = f"nyse_signal_{date_str}.csv"
    out.to_csv(out_path, index=False)

    buy_out = out[out.signal == "BUY"].sort_values("predicted_score", ascending=False)
    sell_out = out[out.signal == "SELL"].sort_values("predicted_score", ascending=True)

    print(f"\n{'='*60}")
    print(f"NYSE DAILY SIGNAL -- {latest_date.date()}")
    print(f"{'='*60}")
    if LONG_ONLY:
        print("mode: LONG-ONLY -- SELL signals suppressed (all non-BUY names are HOLD)")
    if regime_filter_active:
        print(f"REGIME FILTER ACTIVE -- VIX {current_vix:.2f} >= {VIX_THRESHOLD:.0f}, all BUY signals suppressed today")
    print(f"names scored: {n}   BUY: {len(buy_out)}   "
          f"SELL: {len(sell_out)}   HOLD: {(out.signal=='HOLD').sum()}")
    print(f"filtered out of BUY by ${MIN_PRICE:.0f} price floor (forced HOLD): {n_filtered_price}")
    print(f"filtered out of BUY by excluded sector {sorted(SECTOR_EXCLUDE)} (forced HOLD): {n_filtered_sector}")
    print(f"\nTop 10 BUY:")
    print(buy_out.head(10).to_string(index=False))
    if LONG_ONLY:
        print(f"\n(SELL list suppressed -- LONG_ONLY=True)")
    else:
        print(f"\nTop 10 SELL (most bearish):")
        print(sell_out.head(10).to_string(index=False))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
