#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/greeks_taylor.hpp>

using namespace p4mfe;
using namespace p4mfe::ch06;
using ch03::OptionType;

// ---------------------------------------------------------------- 6.8.1 ---
TEST(Ch06Atm, ZeroRates) {
  const double spot = 40;
  for (double maturity : {0.25, 0.5, 1.0}) {
    for (double sigma : {0.1, 0.2, 0.35}) {
      const double call = ch03::BsCall(spot, spot, maturity, sigma, 0.0);
      const double approx = AtmPriceApprox(spot, maturity, sigma);
      EXPECT_NEAR(approx / call - 1, 0.0, sigma * sigma * maturity / 20);
      EXPECT_NEAR(ch03::BsPut(spot, spot, maturity, sigma, 0.0), call, 1e-12);
      EXPECT_NEAR(AtmImpliedVolApprox(call, spot, maturity) / sigma - 1, 0.0,
                  sigma * sigma * maturity / 20);
    }
  }
  // 1/sqrt(2 pi) ~ 0.4.
  EXPECT_NEAR(AtmPriceApprox(100, 1, 0.2), 0.4 * 0.2 * 100, 0.03);
}

TEST(Ch06Atm, WithRatesAndDividends) {
  const double spot = 50, maturity = 0.5, sigma = 0.25;
  for (double r : {0.0, 0.02, 0.05}) {
    for (double q : {0.0, 0.01, 0.03}) {
      const double call = ch03::BsCall(spot, spot, maturity, sigma, r, q);
      const double put = ch03::BsPut(spot, spot, maturity, sigma, r, q);
      // Errors: O((sigma^2 T) sigma sqrt(T)) + O(((r-q)^2/sigma + (r+q)^2)
      // T^{3/2}).
      const double tol = 0.015 * AtmPriceApprox(spot, maturity, sigma);
      EXPECT_NEAR(AtmCallApprox(spot, maturity, sigma, r, q), call, tol)
          << r << ' ' << q;
      EXPECT_NEAR(AtmPutApprox(spot, maturity, sigma, r, q), put, tol)
          << r << ' ' << q;
      // Put-Call parity to first order in r, q.
      EXPECT_NEAR(AtmCallApprox(spot, maturity, sigma, r, q) -
                      AtmPutApprox(spot, maturity, sigma, r, q),
                  (r - q) * maturity * spot, 1e-12);
    }
  }
  // Rates help: the rate-corrected formula beats the r = q = 0 one when r > 0.
  const double call = ch03::BsCall(spot, spot, maturity, sigma, 0.05, 0.0);
  EXPECT_LT(std::abs(AtmCallApprox(spot, maturity, sigma, 0.05) - call),
            std::abs(AtmPriceApprox(spot, maturity, sigma) - call) / 5);
}

TEST(Ch06Atm, Greeks) {
  const double spot = 60, maturity = 0.25, sigma = 0.2;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const auto g = ch03::BsGreeks(type, spot, spot, maturity, sigma, 0.0);
    const auto a = AtmGreeksApprox(type, spot, maturity, sigma);
    EXPECT_NEAR(a.delta, g.delta, 1e-4);
    EXPECT_NEAR(a.gamma / g.gamma, 1.0, 2e-3);
    EXPECT_NEAR(a.vega / g.vega, 1.0, 2e-3);
    EXPECT_NEAR(a.theta / g.theta, 1.0, 2e-3);
    EXPECT_NEAR(a.rho, g.rho, 1e-3);
  }
}

// ---------------------------------------------------------------- 6.8.2 ---
TEST(Ch06Atm, Derivation) {
  // N_lin is N's tangent at 0.
  for (double x : {-0.05, 0.01, 0.1})
    EXPECT_NEAR(NormCdfLinear(x), norm_cdf(x), std::pow(std::abs(x), 3) / 10);
  // Linearizing N, then expanding exponentials, then dropping quadratic rate
  // terms.
  const double spot = 100, maturity = 0.5, sigma = 0.3, r = 0.03, q = 0.01;
  const double exact = ch03::BsCall(spot, spot, maturity, sigma, r, q);
  const double lin = AtmCallLinearized(spot, maturity, sigma, r, q);
  EXPECT_NEAR(lin, exact, 0.01 * exact);
  EXPECT_NEAR(AtmCallApprox(spot, maturity, sigma, r, q), lin, 0.01 * exact);
  EXPECT_NEAR(AtmCallLinearized(spot, maturity, sigma, 0, 0),
              AtmPriceApprox(spot, maturity, sigma), 1e-12);
}

// ---------------------------------------------------------------- 6.8.3 ---
TEST(Ch06Atm, Precision) {
  for (double maturity : {1.0 / 12, 0.5, 1.0, 3.0}) {
    for (double sigma : {0.1, 0.3, 0.6}) {
      const double e = AtmRelativeError(maturity, sigma);
      const double b = AtmRelativeErrorBound(maturity, sigma);
      EXPECT_GT(e, 0.0);       // always overestimates
      EXPECT_LE(e, b);         // by at most sigma^2 T / 24
      EXPECT_GT(e, 0.75 * b);  // and about that much
      EXPECT_NEAR(AtmPriceExact(80, maturity, sigma),
                  ch03::BsCall(80, 80, maturity, sigma, 0.0), 1e-12);
    }
  }
  // Ratio tends to 1 as sigma sqrt(T) -> 0.
  EXPECT_NEAR(AtmRelativeError(0.01, 0.1) / AtmRelativeErrorBound(0.01, 0.1),
              1.0, 1e-4);
}

// ------------------------------------------------------------------ 6.9 ---
TEST(Ch06GreeksTaylor, OptionChangeApproximation) {
  const double spot = 50, strike = 52, maturity = 0.5, sigma = 0.3, r = 0.04,
               q = 0.01;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const double value =
        ch03::BsPrice(type, spot, strike, maturity, sigma, r, q);
    const auto g = ch03::BsGreeks(type, spot, strike, maturity, sigma, r, q);
    // One trading day, a one-standard-deviation move, small vol and rate moves.
    const double dt = 1.0 / 252, d_spot = sigma * spot * std::sqrt(dt),
                 ds = 0.005, dr = 0.001;
    const double exact = ch03::BsPrice(type, spot + d_spot, strike,
                                       maturity - dt, sigma + ds, r + dr, q) -
                         value;
    const double full = OptionChangeTaylor(g, d_spot, dt, ds, dr);
    EXPECT_LT(std::abs(full - exact), 0.01 * std::abs(exact));
    EXPECT_LT(std::abs(OptionChangeDeltaGamma(g, d_spot) + g.theta * dt +
                       g.vega * ds + g.rho * dr - full),
              1e-14);
    // The Gamma term matters: Delta alone is worse than Delta-Gamma (spot move
    // only).
    const double d_value_spot =
        ch03::BsPrice(type, spot + 3 * d_spot, strike, maturity, sigma, r, q) -
        value;
    EXPECT_LT(std::abs(OptionChangeDeltaGamma(g, 3 * d_spot) - d_value_spot),
              std::abs(OptionChangeDelta(g, 3 * d_spot) - d_value_spot) / 5);
    EXPECT_NEAR(ThetaFromPde(value, spot, g.delta, g.gamma, sigma, r, q),
                g.theta, 1e-10);
  }
}

TEST(Ch06GreeksTaylor, DeltaNeutralGammaPnl) {
  // Long one call, short Delta shares (q = 0).
  const double spot = 100, strike = 100, maturity = 0.75, sigma = 0.25,
               r = 0.03;
  const double call = ch03::BsCall(spot, strike, maturity, sigma, r);
  const auto g =
      ch03::BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r);
  const double portfolio_value = call - g.delta * spot;
  const double dt = 1.0 / 252;
  const auto actual = [&](double d_spot) {
    return (ch03::BsCall(spot + d_spot, strike, maturity - dt, sigma, r) -
            call) -
           g.delta * d_spot;
  };
  // The breakeven move is dS^2 = sigma^2 S^2 dt.
  const double be = sigma * spot * std::sqrt(dt);
  for (double m : {0.0, 0.5, 1.0, 2.0}) {
    const double approx = DeltaNeutralChange(portfolio_value, spot, g.gamma,
                                             sigma, r, m * be, dt);
    EXPECT_NEAR(approx, actual(m * be), 5e-3) << m;  // third-order terms
  }
  EXPECT_LT(DeltaNeutralChange(portfolio_value, spot, g.gamma, sigma, r,
                               0.5 * be, dt),
            r * portfolio_value * dt);
  EXPECT_GT(DeltaNeutralChange(portfolio_value, spot, g.gamma, sigma, r,
                               2.0 * be, dt),
            r * portfolio_value * dt);
}
