"""Mean-reversion vs momentum: IC, noise floor, IC decay across holding horizons,
and signal persistence (autocorrelation -> turnover). SP500 universe, 2023-01..2026-06."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from qlib.data import D

S, E = "2023-01-01", "2026-06-16"
FDIR = Path("git_ignore_folder/_mr_factors")
MEANREV = ["composite_reversal_zscore","reversal_20d","price_zscore_20d","rsi_14d","rsi_7d",
           "rsi_14d_centered","williams_r_10d","stochastic_k_14d","z_ema_dev_5d",
           "volatility_normalized_reversal_1d","ts_percentile_rank_5d_return_20d",
           "price_channel_position_20d","kaufman_efficiency_ratio_10d"]
MOMENTUM = ["vw_momentum_5d","vol_adj_momentum_10d","price_trend_slope_10d","obv_momentum_20d",
            "vpt_momentum_10d","PriceToHigh20","VolumeWeightedMom10"]
HORIZONS = [1,2,3,5,10,21]

def ric(sig_w, fwd_w):
    """mean daily cross-sectional Spearman IC"""
    out=[]
    for t in sig_w.index:
        a, b = sig_w.loc[t], fwd_w.loc[t]
        m = a.notna() & b.notna()
        if m.sum() > 20:
            out.append(a[m].corr(b[m], method="spearman"))
    return float(np.nanmean(out))

def main():
    import qlib
    qlib.init(provider_uri=str(Path.home()/".qlib"/"qlib_data"/"us_data"), region="us")
    sp = [l.split("\t")[0].strip() for l in open(Path.home()/".qlib"/"qlib_data"/"us_data"/"instruments"/"sp500.txt") if "2099-12-31" in l]

    # forward returns: enter at t+1 close, exit at t+1+h close
    fwd = {}
    for h in HORIZONS:
        f = D.features(sp, [f"Ref($close,-{1+h})/Ref($close,-1)-1"], start_time=S, end_time=E).iloc[:,0]
        f.index = f.index.set_names(["instrument","datetime"])
        fwd[h] = f.reorder_levels(["datetime","instrument"]).unstack("instrument").sort_index()
    dates = fwd[1].index

    def load(name):
        p = FDIR/f"{name}.h5"
        if not p.exists(): return None
        s = pd.read_hdf(p).iloc[:,0]
        w = s.unstack("instrument")
        cols = [c for c in w.columns if c in fwd[1].columns]
        return w.reindex(index=dates)[cols]

    # ---- noise floor ----
    rng = np.random.default_rng(0)
    null = [ric(pd.DataFrame(rng.standard_normal(fwd[1].shape), index=fwd[1].index, columns=fwd[1].columns), fwd[1]) for _ in range(15)]
    nstd = float(np.std(null))
    print(f"NOISE FLOOR (15 random signals): RankIC mean={np.mean(null):+.5f}  std={nstd:.5f}\n")

    rows=[]
    for fam, names in [("MEANREV",MEANREV), ("MOMENTUM",MOMENTUM)]:
        for n in names:
            w = load(n)
            if w is None: continue
            ics = {h: ric(w, fwd[h]) for h in HORIZONS}
            # signal persistence: cross-sectional rank autocorr at lag k
            ac = {}
            for k in [1,5,21]:
                v=[]
                for i in range(k, len(w), 5):
                    a,b = w.iloc[i], w.iloc[i-k]
                    m=a.notna()&b.notna()
                    if m.sum()>20: v.append(a[m].corr(b[m], method="spearman"))
                ac[k]=float(np.nanmean(v))
            rows.append(dict(fam=fam, name=n, **{f"ic{h}":ics[h] for h in HORIZONS},
                             ac1=ac[1], ac5=ac[5], ac21=ac[21], z=ics[1]/nstd))
    df=pd.DataFrame(rows)

    print(f"{'factor':36}{'IC h=1':>8}{'h=2':>8}{'h=3':>8}{'h=5':>8}{'h=10':>8}{'h=21':>8}{'|z| h1':>8}{'AC(1)':>7}{'AC(5)':>7}{'AC(21)':>8}")
    for fam in ["MEANREV","MOMENTUM"]:
        print(f"--- {fam} ---")
        for _,r in df[df.fam==fam].iterrows():
            print(f"{r['name']:36}{r.ic1:>+8.4f}{r.ic2:>+8.4f}{r.ic3:>+8.4f}{r.ic5:>+8.4f}{r.ic10:>+8.4f}{r.ic21:>+8.4f}"
                  f"{abs(r.z):>8.1f}{r.ac1:>+7.2f}{r.ac5:>+7.2f}{r.ac21:>+8.2f}")

    print("\n=== FAMILY AVERAGES (|IC| — sign-agnostic, MR factors are negatively signed by construction) ===")
    print(f"{'family':10}{'|IC| h=1':>10}{'h=2':>8}{'h=3':>8}{'h=5':>8}{'h=10':>8}{'h=21':>8}{'AC(1)':>8}{'AC(5)':>8}{'AC(21)':>8}")
    for fam in ["MEANREV","MOMENTUM"]:
        d=df[df.fam==fam]
        print(f"{fam:10}{d.ic1.abs().mean():>10.4f}{d.ic2.abs().mean():>8.4f}{d.ic3.abs().mean():>8.4f}"
              f"{d.ic5.abs().mean():>8.4f}{d.ic10.abs().mean():>8.4f}{d.ic21.abs().mean():>8.4f}"
              f"{d.ac1.abs().mean():>8.2f}{d.ac5.abs().mean():>8.2f}{d.ac21.abs().mean():>8.2f}")
    print("\n=== IC RETENTION (|IC| at horizon h) / (|IC| at h=1)  -- higher = decays SLOWER ===")
    print(f"{'family':10}{'h=2':>8}{'h=3':>8}{'h=5':>8}{'h=10':>8}{'h=21':>8}")
    for fam in ["MEANREV","MOMENTUM"]:
        d=df[df.fam==fam]; base=d.ic1.abs().mean()
        print(f"{fam:10}" + "".join(f"{d[f'ic{h}'].abs().mean()/base:>8.2f}" for h in [2,3,5,10,21]))
    df.to_csv("git_ignore_folder/_mr_factors/summary.csv", index=False)

if __name__ == "__main__":
    main()
