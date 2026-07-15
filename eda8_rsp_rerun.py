"""Re-evaluate the 30-loop run's long-only Top-50 (n_drop=5) portfolios against RSP and
equal-weight, net of costs. Reuses each loop's saved report_normal_1day.pkl (portfolio is
benchmark-independent); only the excess is recomputed. No re-backtest."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis

S, E = "2023-01-01", "2026-06-16"
WSROOT = Path("git_ignore_folder/RD-Agent_workspace")

def feat1(inst, expr, dates):
    f = D.features([inst], [expr], start_time=S, end_time=E).iloc[:,0]
    f.index = f.index.droplevel("instrument") if "instrument" in f.index.names else f.index
    return f.reindex(dates)

def ann(r): r=r.dropna(); return (1+r).prod()**(252/len(r))-1

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    wss = [l.strip() for l in open("ws_order.txt") if l.strip()]

    sp500 = [l.split("\t")[0].strip() for l in open(Path.home()/".qlib"/"qlib_data"/"us_data"/"instruments"/"sp500.txt") if "2099-12-31" in l]
    consret = D.features(sp500, ["$close/Ref($close,1)-1"], start_time=S, end_time=E).iloc[:,0]
    consret.index = consret.index.set_names(["instrument","datetime"])
    ew_all = consret.groupby(level="datetime").mean()
    rsp_all = D.features(["RSP"], ["$close/Ref($close,1)-1"], start_time=S, end_time=E).iloc[:,0]
    rsp_all.index = rsp_all.index.get_level_values("datetime")

    rows=[]
    for i,w in enumerate(wss):
        rep_path = next(WSROOT.joinpath(w).rglob("report_normal_1day.pkl"))
        rep = pd.read_pickle(rep_path)
        gross = rep["return"]; net = rep["return"] - rep["cost"]; spy = rep["bench"]; dates = gross.index
        rsp = rsp_all.reindex(dates); ew = ew_all.reindex(dates)
        csv = pd.read_csv(next(WSROOT.joinpath(w).glob("qlib_res.csv")), index_col=0).iloc[:,0]
        ic = float(csv.get("IC", np.nan))
        def rr(ex):
            r=risk_analysis(ex.dropna(), freq="day")["risk"]; return r["annualized_return"], r["information_ratio"], r["max_drawdown"]
        # NET of cost is the real answer
        exR_g,_,_   = rr(gross-rsp)
        exR_n,iR_n,dR_n = rr(net-rsp)
        exE_n,iE_n,_ = rr(net-ew)
        exS_n,iS_n,_ = rr(net-spy)
        rows.append(dict(ic=ic, portG=ann(gross), portN=ann(net), rsp=ann(rsp), ew=ann(ew), spy=ann(spy),
                         cost=ann(gross)-ann(net), exR_g=exR_g, exR_n=exR_n, irR_n=iR_n, ddR_n=dR_n,
                         exE_n=exE_n, irE_n=iE_n, exS_n=exS_n))
    df=pd.DataFrame(rows)

    print(f"benchmarks (annualized): SPY={df['spy'].iloc[0]:+.4f}  RSP={df['rsp'].iloc[0]:+.4f}  EW={df['ew'].iloc[0]:+.4f}")
    print(f"avg cost drag (gross-net portfolio return): {df['cost'].mean():.4f}/yr")
    print(f"\n{'bt':>3}{'IC':>8}{'portNet':>9}{'exRSP_g':>9}{'exRSP_net':>10}{'IR_RSP_n':>9}{'exEW_net':>9}")
    for i,r in df.iterrows():
        print(f"{i:>3}{r.ic:>8.4f}{r.portN:>9.4f}{r.exR_g:>+9.3f}{r.exR_n:>+10.3f}{r.irR_n:>+9.2f}{r.exE_n:>+9.3f}")

    def summ(col,name):
        v=df[col]; print(f"  {name:20} mean={v.mean():+.4f} std={v.std():.4f} min={v.min():+.4f} max={v.max():+.4f} %>0={100*(v>0).mean():.0f}%")
    print("\n=== NET-OF-COST excess distribution across 30 loops ===")
    summ("exS_n","net excess vs SPY"); summ("exR_n","net excess vs RSP"); summ("exE_n","net excess vs EW")
    summ("irR_n","net IR vs RSP")
    v=df["exR_n"].values; r=np.corrcoef(np.arange(len(v)),v)[0,1]
    print(f"\n  net-excess-vs-RSP trend: corr(loop,excess)={r:+.2f}  first10avg={v[:10].mean():+.4f}  last10avg={v[-10:].mean():+.4f}")
    print(f"  loops beating RSP NET of cost: {int((df['exR_n']>0).sum())}/30   beating EW NET: {int((df['exE_n']>0).sum())}/30")

if __name__ == "__main__":
    main()
