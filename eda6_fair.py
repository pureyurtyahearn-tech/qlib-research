"""Fair-comparison backtests for a representative real prediction:
 A) long-only TopK50 excess vs SPY, RSP(ETF), and constructed equal-weight SP500
 B) market-neutral long-short (top50 - bottom50) = pure cross-sectional factor alpha."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from qlib.backtest import backtest as normal_backtest
from qlib.contrib.evaluate import risk_analysis

REAL_PRED = r"git_ignore_folder/RD-Agent_workspace/12b0eb7178a0495a980618bf8bac9a54/mlruns/657215356007540889/871f0a480c7547f5bd6dd434c863d6c7/artifacts/pred.pkl"
S, E = "2023-01-01", "2026-06-16"

def feat(insts, expr):
    f = D.features(insts, [expr], start_time=S, end_time=E).iloc[:, 0]
    f.index = f.index.set_names(["instrument", "datetime"])
    return f.reorder_levels(["datetime", "instrument"]).sort_index()

def ann(series):  # annualized compounded return from daily simple returns
    series = series.dropna()
    return (1+series).prod()**(252/len(series)) - 1

def ra(excess, label):
    r = risk_analysis(excess.dropna(), freq="day")["risk"]
    print(f"    vs {label:16} annRet={r['annualized_return']:+.4f}  IR={r['information_ratio']:+.3f}  maxDD={r['max_drawdown']:+.4f}")

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    pred = pd.read_pickle(REAL_PRED); pred = pred.iloc[:,0] if pred.ndim>1 else pred
    insts = list(pred.index.get_level_values("instrument").unique())
    sp500 = [l.split("\t")[0].strip() for l in open(Path.home()/".qlib"/"qlib_data"/"us_data"/"instruments"/"sp500.txt") if "2099-12-31" in l]

    # ---- A) long-only TopK50 backtest ----
    strat = {"class":"TopkDropoutStrategy","module_path":"qlib.contrib.strategy","kwargs":{"signal":pred.to_frame("score"),"topk":50,"n_drop":5}}
    ex = {"class":"SimulatorExecutor","module_path":"qlib.backtest.executor","kwargs":{"time_per_step":"day","generate_portfolio_metrics":True}}
    pmd,_ = normal_backtest(executor=ex, strategy=strat, start_time=S, end_time=E, account=100000000, benchmark="SPY",
                            exchange_kwargs={"limit_threshold":None,"deal_price":"close","open_cost":0.0005,"close_cost":0.0015,"min_cost":5})
    rep,_ = pmd["1day"]
    port = rep["return"]; spy = rep["bench"]
    dates = port.index
    # RSP daily return and equal-weight index daily return, aligned to backtest dates
    rsp = feat(["RSP"], "$close/Ref($close,1)-1").xs("RSP", level="instrument").reindex(dates)
    cons_ret = feat(sp500, "$close/Ref($close,1)-1")
    ew = cons_ret.groupby(level="datetime").mean().reindex(dates)

    print("=== A) LONG-ONLY TopK50 (representative pred, IC~0.005) ===")
    print(f"  absolute annualized returns:")
    print(f"    portfolio ={ann(port):+.4f}   SPY={ann(spy):+.4f}   RSP={ann(rsp):+.4f}   EW-SP500={ann(ew):+.4f}")
    print(f"  benchmark structural gap:  SPY-EW = {ann(spy)-ann(ew):+.4f}  (cap-weight vs equal-weight drag)")
    print(f"  excess-return metrics (annualized):")
    ra(port-spy, "SPY (cap-wt)")
    ra(port-rsp, "RSP (EW ETF)")
    ra(port-ew,  "EW-SP500 idx")

    # ---- B) market-neutral long-short ----
    print("\n=== B) MARKET-NEUTRAL LONG-SHORT (top50 - bottom50, equal-wt, daily) ===")
    fwd = feat(insts, "Ref($close,-2)/Ref($close,-1)-1")   # tradeable next-day fwd return
    sig = pred
    df = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
    def ls_day(g, k=50):
        if len(g) < 2*k: k = max(5, len(g)//5)
        g = g.sort_values("sig")
        return g["fwd"].iloc[-k:].mean() - g["fwd"].iloc[:k].mean()
    ls = df.groupby(level="datetime").apply(ls_day)
    long_only = df.groupby(level="datetime").apply(lambda g: g.sort_values("sig")["fwd"].iloc[-50:].mean())
    ew_fwd = df.groupby(level="datetime")["fwd"].mean()
    def stats(r, name):
        r = r.dropna()
        sharpe = r.mean()/r.std()*np.sqrt(252)
        cum = (1+r).cumprod(); dd = (cum/cum.cummax()-1).min()
        print(f"  {name:22} annRet={ann(r):+.4f}  Sharpe={sharpe:+.2f}  maxDD={dd:+.4f}  hit={100*(r>0).mean():.1f}%")
    stats(ls, "long-short (top-bottom)")
    stats(long_only-ew_fwd, "long-only minus EW")
    stats(ew_fwd, "equal-weight (ref)")

if __name__ == "__main__":
    main()
