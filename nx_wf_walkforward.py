"""ROLLING WALK-FORWARD for the NYSE RD-Agent SOTA factor library — same standard as
wf1_walkforward.py's SP500 FCF-yield/ROE test, adapted for a multi-factor trained model
instead of a single sign-locked raw factor.

Why this matters: loop 24's +23.13% headline is a SINGLE backtest window (2022-01-01 to
2023-12-31, per run_rdagent.py's patched YAML), and it was arrived at by an LLM comparing
30 successive trials against that SAME window -- implicit selection bias. This script
retrains the identical feature set (Alpha158's 20 base features + the 33 accepted custom
factors from loops 1,3,4,6,10,13,17,24, pulled directly from loop 24's own
combined_factors_df.parquet -- not re-derived, the literal training data) on an EXPANDING
window ending before each test year, then scores that year out-of-sample. This turns one
cherry-picked window into ~17 independent OOS observations.

Reused UNCHANGED from the SP500 walk-forward (sn_common.py / ext6_momentum_full.py):
  - rank_ic, quintiles, phased_book, simulate, ann, ir, tstat
  - cost assumptions: 5bps open / 15bps close (OPEN_C, CLOSE_C)
  - K=50 top/bottom portfolio, 21-phase monthly rebalance
Adapted for NYSE:
  - universe/eligibility from D.instruments(market='nyse') x sep_nyse_panel.h5, not the
    SP500 PIT membership matrix (NYSE has no index-membership concept, same as the store)
  - signal = LGBM prediction (53 features) retrained each fold, not a single raw factor
  - LGBM hyperparams match run_rdagent.py's Alpha158 template exactly (same run trained on)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb
import qlib
from qlib.data import D
import sn_common as C
from ext6_momentum_full import simulate, ann, ir, tstat

STORE = Path.home() / ".qlib" / "qlib_data" / "us_data_pit_full"
SH = Path("git_ignore_folder/sharadar")
SOTA_WORKSPACE = "c3955146822249b6b195e5c4e084de5a"   # loop 24
FIRST_TEST, LAST = 2010, 2026
K = 50

# Alpha158's 20 base features, exact expressions (same as run_rdagent.py's FilterCol list)
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
LABEL_EXPR = 'Ref($close, -2) / Ref($close, -1) - 1'   # exact qrun training label

LGB_PARAMS = dict(objective="regression", loss="mse", colsample_bytree=0.8879,
                   learning_rate=0.2, subsample=0.8789, lambda_l1=205.6999,
                   lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=20,
                   verbosity=-1)


def main():
    qlib.init(provider_uri=str(STORE), region="us", kernels=1)
    codes = D.list_instruments(D.instruments(market="nyse"),
                               start_time="1999-01-01", end_time="2026-06-29", as_list=True)
    print(f"NYSE market: {len(codes)} instruments", flush=True)

    # ---- Alpha158 base features (raw, same 20 as the actual training run) ----
    names20 = list(ALPHA158_EXPR)
    alpha = D.features(codes, list(ALPHA158_EXPR.values()) + [LABEL_EXPR],
                       start_time="1999-01-01", end_time="2026-06-29")
    alpha.columns = names20 + ["label"]
    alpha.index = alpha.index.set_names(["instrument", "datetime"])
    alpha = alpha.reset_index().set_index(["datetime", "instrument"]).sort_index()
    print(f"Alpha158 base + label: {alpha.shape}", flush=True)

    # ---- 33 accepted custom factors, pulled directly from loop 24's own training data ----
    custom = pd.read_parquet(f"git_ignore_folder/RD-Agent_workspace/{SOTA_WORKSPACE}/combined_factors_df.parquet")
    custom.columns = [c[1] for c in custom.columns]
    custom_cols = list(custom.columns)
    print(f"custom SOTA factors ({len(custom_cols)}): {custom_cols}", flush=True)

    panel = alpha.join(custom, how="inner")
    panel = panel[panel.index.get_level_values("instrument").isin(set(codes))]
    feat_cols = names20 + custom_cols
    print(f"merged panel: {panel.shape}  ({len(feat_cols)} features)", flush=True)

    # ---- close/volume wide arrays for eligibility, daily returns, 21d fwd (IC diagnostic) ----
    px = pd.read_hdf(SH / "sep_nyse_panel.h5")
    close = px["$close"].unstack("ticker").sort_index()
    close = close[[c for c in close.columns if c in set(codes)]]
    retv_full = close.pct_change()
    fwd_full = (close.shift(-22) / close.shift(-1) - 1)
    elig_full = close.notna()

    dates = panel.index.get_level_values("datetime").unique().sort_values()
    years = dates.year

    rows = []
    for y in range(FIRST_TEST, LAST + 1):
        train = panel[panel.index.get_level_values("datetime") < f"{y}-01-01"]
        test_idx = panel.index.get_level_values("datetime")
        test = panel[(test_idx >= f"{y}-01-01") & (test_idx <= f"{y}-12-31")]
        train = train.dropna(subset=["label"])
        if len(train) < 10000 or len(test) < 1000:
            print(f"  {y}: skipped (train={len(train)}, test={len(test)} rows -- insufficient)")
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=200)
        model.fit(train[feat_cols], train["label"])
        pred = model.predict(test[feat_cols])
        pred_s = pd.Series(pred, index=test.index)
        sig_wide = pred_s.unstack("instrument").reindex(index=close.index, columns=close.columns)

        d = close.index[(close.index >= f"{y}-01-01") & (close.index <= f"{y}-12-31")]
        if len(d) < 120:
            continue
        elig = elig_full.loc[d].values
        retv = retv_full.loc[d].values
        fwd = fwd_full.loc[d].values
        ew = np.array([np.nanmean(np.where(elig[t], retv[t], np.nan)) for t in range(len(d))])

        sig = sig_wide.loc[d].shift(1).values     # shift(1): trade on t+1 using signal known at t
        ic_arr = C.rank_ic(sig_wide.loc[d].values, fwd, elig)
        ic = float(ic_arr.mean()) if len(ic_arr) else np.nan
        top_ex, top_t, top_turn, top_pp = C.phased_book(sig, retv, elig, ew, len(d), K, bottom=False)
        bot_ex, _, _, _ = C.phased_book(sig, retv, elig, ew, len(d), K, bottom=True)

        rec = dict(year=y, n_train=len(train), n_test_names=int(elig.any(axis=0).sum()),
                   ic=ic, ew=ann(ew), top_ex=top_ex, bot_ex=bot_ex, ls=top_ex - bot_ex,
                   turn=top_turn, phases_pos=top_pp)
        rows.append(rec)
        print(f"  {y}: train_rows={len(train):,}  names={rec['n_test_names']}  "
              f"IC {ic:+.4f}  EW {rec['ew']:+.1%}  top50_ex {top_ex:+.2%}  "
              f"LS {rec['ls']:+.2%}  ({100*top_pp:.0f}% phases>0)", flush=True)

    wf = pd.DataFrame(rows).set_index("year")
    wf.to_csv(SH / "nx_wf_walkforward.csv")

    print(f"\n{'='*72}\nNYSE WALK-FORWARD SUMMARY  ({len(wf)} OOS years, {FIRST_TEST}-{LAST})\n{'='*72}")
    ic, ex = wf["ic"], wf["top_ex"]
    t_ex = ex.mean() / ex.std() * np.sqrt(len(ex)) if ex.std() > 0 else 0
    print(f"  mean OOS IC          : {ic.mean():+.4f}   IC>0: {(ic>0).sum()}/{len(ic)} ({(ic>0).mean():.0%})")
    print(f"  mean top-50 net excess/yr : {ex.mean():+.2%}   median {ex.median():+.2%}")
    print(f"  hit rate (years excess>0) : {(ex>0).sum()}/{len(ex)} ({(ex>0).mean():.0%})")
    print(f"  t-stat (excess)      : {t_ex:+.2f}")
    print(f"  mean long-short/yr   : {wf['ls'].mean():+.2%}")
    print(f"\n  in-sample headline (loop 24, 2022-2023 backtest): IC=0.002652, ann.ret +23.13%, dd -21.92%")
    print(f"  OOS 2022: {wf.loc[2022] if 2022 in wf.index else 'n/a'}")
    print(f"  OOS 2023: {wf.loc[2023] if 2023 in wf.index else 'n/a'}")
    print(f"\nsaved nx_wf_walkforward.csv")


if __name__ == "__main__":
    main()
