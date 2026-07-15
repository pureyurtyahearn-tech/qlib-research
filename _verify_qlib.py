from pathlib import Path

def main():
    import qlib
    from qlib.data import D
    # kernels=1 -> single process; avoids Windows spawn overhead on low free RAM
    qlib.init(provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "us_data"),
              region="us", kernels=1)

    insts = D.instruments(market="sp500")
    inst_list = D.list_instruments(insts, start_time="2023-01-01",
                                   end_time="2026-06-16", as_list=True)
    print("sp500 instruments active in window:", len(inst_list), flush=True)

    df = D.features(inst_list, ["$open", "$high", "$low", "$close", "$volume"],
                    start_time="2023-01-01", end_time="2026-06-16")
    print("rows:", len(df), flush=True)
    print("date span:", df.index.get_level_values(1).min(), "->",
          df.index.get_level_values(1).max(), flush=True)
    print("instruments returned:", df.index.get_level_values(0).nunique(), flush=True)
    post = df.loc[(slice(None), slice("2025-01-01", None)), "$close"]
    print("NaN frac in $close (2025+):", round(float(post.isna().mean()), 4), flush=True)
    print("OK - heavy query succeeded", flush=True)

if __name__ == "__main__":
    main()
