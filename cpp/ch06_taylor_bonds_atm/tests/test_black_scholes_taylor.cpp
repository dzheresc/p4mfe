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
  const double S = 40;
  for (double T : {0.25, 0.5, 1.0}) {
    for (double sigma : {0.1, 0.2, 0.35}) {
      const double C = ch03::bs_call(S, S, T, sigma, 0.0);
      const double approx = atm_price_approx(S, T, sigma);
      EXPECT_NEAR(approx / C - 1, 0.0, sigma * sigma * T / 20);
      EXPECT_NEAR(ch03::bs_put(S, S, T, sigma, 0.0), C, 1e-12);
      EXPECT_NEAR(atm_implied_vol_approx(C, S, T) / sigma - 1, 0.0,
                  sigma * sigma * T / 20);
    }
  }
  // 1/sqrt(2 pi) ~ 0.4.
  EXPECT_NEAR(atm_price_approx(100, 1, 0.2), 0.4 * 0.2 * 100, 0.03);
}

TEST(Ch06Atm, WithRatesAndDividends) {
  const double S = 50, T = 0.5, sigma = 0.25;
  for (double r : {0.0, 0.02, 0.05}) {
    for (double q : {0.0, 0.01, 0.03}) {
      const double C = ch03::bs_call(S, S, T, sigma, r, q);
      const double P = ch03::bs_put(S, S, T, sigma, r, q);
      // Errors: O((sigma^2 T) sigma sqrt(T)) + O(((r-q)^2/sigma + (r+q)^2)
      // T^{3/2}).
      const double tol = 0.015 * atm_price_approx(S, T, sigma);
      EXPECT_NEAR(atm_call_approx(S, T, sigma, r, q), C, tol) << r << ' ' << q;
      EXPECT_NEAR(atm_put_approx(S, T, sigma, r, q), P, tol) << r << ' ' << q;
      // Put-Call parity to first order in r, q.
      EXPECT_NEAR(atm_call_approx(S, T, sigma, r, q) -
                      atm_put_approx(S, T, sigma, r, q),
                  (r - q) * T * S, 1e-12);
    }
  }
  // Rates help: the rate-corrected formula beats the r = q = 0 one when r > 0.
  const double C = ch03::bs_call(S, S, T, sigma, 0.05, 0.0);
  EXPECT_LT(std::abs(atm_call_approx(S, T, sigma, 0.05) - C),
            std::abs(atm_price_approx(S, T, sigma) - C) / 5);
}

TEST(Ch06Atm, Greeks) {
  const double S = 60, T = 0.25, sigma = 0.2;
  for (OptionType type : {OptionType::Call, OptionType::Put}) {
    const auto g = ch03::bs_greeks(type, S, S, T, sigma, 0.0);
    const auto a = atm_greeks_approx(type, S, T, sigma);
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
    EXPECT_NEAR(norm_cdf_linear(x), norm_cdf(x), std::pow(std::abs(x), 3) / 10);
  // Linearizing N, then expanding exponentials, then dropping quadratic rate
  // terms.
  const double S = 100, T = 0.5, sigma = 0.3, r = 0.03, q = 0.01;
  const double exact = ch03::bs_call(S, S, T, sigma, r, q);
  const double lin = atm_call_linearized(S, T, sigma, r, q);
  EXPECT_NEAR(lin, exact, 0.01 * exact);
  EXPECT_NEAR(atm_call_approx(S, T, sigma, r, q), lin, 0.01 * exact);
  EXPECT_NEAR(atm_call_linearized(S, T, sigma, 0, 0),
              atm_price_approx(S, T, sigma), 1e-12);
}

// ---------------------------------------------------------------- 6.8.3 ---
TEST(Ch06Atm, Precision) {
  for (double T : {1.0 / 12, 0.5, 1.0, 3.0}) {
    for (double sigma : {0.1, 0.3, 0.6}) {
      const double e = atm_relative_error(T, sigma);
      const double b = atm_relative_error_bound(T, sigma);
      EXPECT_GT(e, 0.0);       // always overestimates
      EXPECT_LE(e, b);         // by at most sigma^2 T / 24
      EXPECT_GT(e, 0.75 * b);  // and about that much
      EXPECT_NEAR(atm_price_exact(80, T, sigma),
                  ch03::bs_call(80, 80, T, sigma, 0.0), 1e-12);
    }
  }
  // Ratio tends to 1 as sigma sqrt(T) -> 0.
  EXPECT_NEAR(
      atm_relative_error(0.01, 0.1) / atm_relative_error_bound(0.01, 0.1), 1.0,
      1e-4);
}

// ------------------------------------------------------------------ 6.9 ---
TEST(Ch06GreeksTaylor, OptionChangeApproximation) {
  const double S = 50, K = 52, T = 0.5, sigma = 0.3, r = 0.04, q = 0.01;
  for (OptionType type : {OptionType::Call, OptionType::Put}) {
    const double V = ch03::bs_price(type, S, K, T, sigma, r, q);
    const auto g = ch03::bs_greeks(type, S, K, T, sigma, r, q);
    // One trading day, a one-standard-deviation move, small vol and rate moves.
    const double dt = 1.0 / 252, dS = sigma * S * std::sqrt(dt), ds = 0.005,
                 dr = 0.001;
    const double exact =
        ch03::bs_price(type, S + dS, K, T - dt, sigma + ds, r + dr, q) - V;
    const double full = option_change_taylor(g, dS, dt, ds, dr);
    EXPECT_LT(std::abs(full - exact), 0.01 * std::abs(exact));
    EXPECT_LT(std::abs(option_change_delta_gamma(g, dS) + g.theta * dt +
                       g.vega * ds + g.rho * dr - full),
              1e-14);
    // The Gamma term matters: Delta alone is worse than Delta-Gamma (spot move
    // only).
    const double dV_S = ch03::bs_price(type, S + 3 * dS, K, T, sigma, r, q) - V;
    EXPECT_LT(std::abs(option_change_delta_gamma(g, 3 * dS) - dV_S),
              std::abs(option_change_delta(g, 3 * dS) - dV_S) / 5);
    EXPECT_NEAR(theta_from_pde(V, S, g.delta, g.gamma, sigma, r, q), g.theta,
                1e-10);
  }
}

TEST(Ch06GreeksTaylor, DeltaNeutralGammaPnl) {
  // Long one call, short Delta shares (q = 0).
  const double S = 100, K = 100, T = 0.75, sigma = 0.25, r = 0.03;
  const double C = ch03::bs_call(S, K, T, sigma, r);
  const auto g = ch03::bs_greeks(OptionType::Call, S, K, T, sigma, r);
  const double Pi = C - g.delta * S;
  const double dt = 1.0 / 252;
  const auto actual = [&](double dS) {
    return (ch03::bs_call(S + dS, K, T - dt, sigma, r) - C) - g.delta * dS;
  };
  // The breakeven move is dS^2 = sigma^2 S^2 dt.
  const double be = sigma * S * std::sqrt(dt);
  for (double m : {0.0, 0.5, 1.0, 2.0}) {
    const double approx =
        delta_neutral_change(Pi, S, g.gamma, sigma, r, m * be, dt);
    EXPECT_NEAR(approx, actual(m * be), 5e-3) << m;  // third-order terms
  }
  EXPECT_LT(delta_neutral_change(Pi, S, g.gamma, sigma, r, 0.5 * be, dt),
            r * Pi * dt);
  EXPECT_GT(delta_neutral_change(Pi, S, g.gamma, sigma, r, 2.0 * be, dt),
            r * Pi * dt);
}
