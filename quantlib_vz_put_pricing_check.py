"""QuantLib sanity check: Black-Scholes-Merton fair value for a short VZ put.

One-off pricing check on a specific trade -- not part of the NYSE factor pipeline.
Prices the VZ Aug 21 2026 $43 European put (sold for $0.75) using QuantLib's
AnalyticEuropeanEngine (Black-Scholes-Merton with a continuous dividend yield) and
compares the theoretical value against the actual sale price.

CAVEAT: volatility is a single flat number (28.1%, VZ's ATM IV from a TWS screenshot),
applied uniformly across strikes. The $43 strike is ~2.2% out-of-the-money for a put
(spot $43.97 > strike $43), and equity puts typically carry volatility skew -- OTM
puts usually trade at HIGHER IV than ATM, not the same. If the true market IV for the
$43 strike specifically is above 28.1%, the real theoretical value is probably somewhat
above this script's output. Treat this as a quick sanity check, not a precise skew-
consistent valuation -- rerun with the actual $43-strike IV from the option chain for
a tighter number.
"""
import QuantLib as ql

# ---- inputs ----
SPOT = 43.97            # VZ current price
STRIKE = 43.00
RISK_FREE_RATE = 0.0525  # approximate US risk-free rate
DIVIDEND_YIELD = 0.065   # VZ dividend yield
VOLATILITY = 0.281       # ATM IV from TWS screenshot
DAYS_TO_EXPIRY = 28      # Aug 21 2026 expiry
SOLD_FOR = 0.75


def main():
    print("QuantLib version:", ql.__version__)

    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()

    today = ql.Date(24, 7, 2026)
    ql.Settings.instance().evaluationDate = today
    expiry = today + DAYS_TO_EXPIRY
    print(f"today: {today}  expiry: {expiry}")

    payoff = ql.PlainVanillaPayoff(ql.Option.Put, STRIKE)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(SPOT))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, RISK_FREE_RATE, day_count))
    div_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, DIVIDEND_YIELD, day_count))
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, VOLATILITY, day_count))

    process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    npv = option.NPV()
    delta = option.delta()
    gamma = option.gamma()
    theta = option.theta()            # per year
    theta_daily = theta / 365.0       # per calendar day
    vega = option.vega()              # per 1.00 = 100 vol points

    print(f"\nTheoretical fair value (NPV): {npv:.4f}")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta (per year): {theta:.4f}")
    print(f"Theta (per calendar day): {theta_daily:.4f}")
    print(f"Vega (per 1.00 = 100 vol pts): {vega:.4f}  ->  per 1 vol point (1%): {vega / 100:.4f}")

    diff = SOLD_FOR - npv
    pct = diff / npv * 100
    print(f"\nSold for: {SOLD_FOR}")
    print(f"Model fair value: {npv:.4f}")
    print(f"Difference (sold - model): {diff:+.4f}  ({pct:+.1f}% relative to model)")


if __name__ == "__main__":
    main()
