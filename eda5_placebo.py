"""EDA part 5: placebo / noise-floor. Run random signals through the exact qlib label
(Ref($close,-2)/Ref($close,-1)-1) and backtest; compare to the real prediction's IC."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from qlib.backtest import backtest as normal_backtest

REAL_PRED = r"git_ignore_folder/RD-Agent_workspace/12b0eb7178a0495a980618bf8bac9a54/mlruns/657215356007540889/871f0a480c7547f5bd6dd434c863d6c7/artifacts/pred.pkl"

def daily_ic(sig, lab, rank=True):
    dfm = pd.DataFrame({"s": sig, "l": lab}).dropna()
    method = "spearman" if rank else "pearson"
    ic = dfm.groupby(level="datetime").apply(lambda g: g["s"].corr(g["l"], method=method) if len(g) > 3 else np.nan)
    return ic.mean()

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")

    pred = pd.read_pickle(REAL_PRED)
    pred = pred.iloc[:, 0] if pred.ndim > 1 else pred
    insts = list(pred.index.get_level_values("instrument").unique())
    print(f"signal shape: {pred.shape}, {len(insts)} instruments, "
          f"{pred.index.get_level_values('datetime').min().date()}..{pred.index.get_level_values('datetime').max().date()}")

    # exact qlib label: return from t+1 close to t+2 close (tradeable next day)
    lab = D.features(insts, ["Ref($close,-2)/Ref($close,-1)-1"],
                     start_time="2023-01-01", end_time="2026-06-16").iloc[:, 0]
    lab.index = lab.index.set_names(["instrument", "datetime"])
    lab = lab.reorder_levels(["datetime", "instrument"]).sort_index()

    # ---- null distribution: 20 random signals ----
    print("\n=== 5a. NULL: 20 random-noise signals, Rank IC vs qlib label ===")
    rng = np.random.default_rng(0)
    null_ic = []
    for k in range(20):
        rs = pd.Series(rng.standard_normal(len(pred)), index=pred.index)
        null_ic.append(daily_ic(rs, lab, rank=True))
    null_ic = np.array(null_ic)
    print(f"  null RankIC: mean={null_ic.mean():+.5f}  std={null_ic.std():.5f}  "
          f"min={null_ic.min():+.5f}  max={null_ic.max():+.5f}  max|IC|={np.abs(null_ic).max():.5f}")

    # ---- real prediction's IC vs same label ----
    real_ric = daily_ic(pred, lab, rank=True)
    real_ic  = daily_ic(pred, lab, rank=False)
    z = (real_ric - null_ic.mean()) / null_ic.std()
    print(f"\n=== 5b. REAL prediction vs same label ===")
    print(f"  real RankIC={real_ric:+.5f}   real IC={real_ic:+.5f}")
    print(f"  real RankIC is {z:+.1f} sigma above the null mean  (null std={null_ic.std():.5f})")

    # ---- one random signal through the full backtest ----
    print("\n=== 5c. random signal through the qlib backtest (portfolio) ===")
    rs = pd.Series(np.random.default_rng(42).standard_normal(len(pred)), index=pred.index).to_frame("score")
    strat = {"class":"TopkDropoutStrategy","module_path":"qlib.contrib.strategy","kwargs":{"signal":rs,"topk":50,"n_drop":5}}
    ex = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor","kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    pmd,_ = normal_backtest(executor=ex, strategy=strat, start_time="2023-01-01", end_time="2026-06-16",
                            account=100000000, benchmark="SPY",
                            exchange_kwargs={"limit_threshold":None,"deal_price":"close","open_cost":0.0005,"close_cost":0.0015,"min_cost":5})
    rep,_ = pmd["1day"]
    ann = (1+rep["return"]).prod()**(252/len(rep))-1
    bench_ann = (1+rep["bench"]).prod()**(252/len(rep))-1
    print(f"  random-portfolio ann return={ann:+.4f}  vs SPY ann={bench_ann:+.4f}  excess={ann-bench_ann:+.4f}")
    print(f"  turnover_sum={float(rep['total_turnover'].sum()):.3e}  nonzero-return days={int((rep['return']!=0).sum())}")

if __name__ == "__main__":
    main()
