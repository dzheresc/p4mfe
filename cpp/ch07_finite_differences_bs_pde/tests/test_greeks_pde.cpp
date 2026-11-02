#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/greeks_taylor.hpp>
#include <p4mfe/ch07/bs_pde.hpp>
#include <p4mfe/ch07/fd_greeks.hpp>

using namespace p4mfe;
using namespace p4mfe::ch07;

// ---------------------------------------------------------------- 7.3 -----
TEST(Ch07FdGreeks, MatchClosedForm) {
  const double S = 42, K = 40, T = 0.75, sigma = 0.3, r = 0.04, q = 0.01;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const auto exact = ch03::bs_greeks(type, S, K, T, sigma, r, q);
    const auto c = bs_fd_greeks(type, S, K, T, sigma, r, q);
    EXPECT_NEAR(c.delta, exact.delta, 1e-6);
    EXPECT_NEAR(c.gamma, exact.gamma, 1e-6);
    EXPECT_NEAR(c.vega, exact.vega, 1e-6);
    EXPECT_NEAR(c.theta, exact.theta, 1e-6);
    EXPECT_NEAR(c.rho, exact.rho, 1e-6);
    // One-sided differences are O(h): much less accurate with the same steps.
    const FdSteps big{0.5, 0.01, 0.01, 0.01};
    const auto fwd =
        bs_fd_greeks(type, S, K, T, sigma, r, q, big, FdScheme::Forward);
    const auto cen =
        bs_fd_greeks(type, S, K, T, sigma, r, q, big, FdScheme::Central);
    EXPECT_GT(std::abs(fwd.delta - exact.delta),
              10 * std::abs(cen.delta - exact.delta));
    EXPECT_GT(std::abs(fwd.vega - exact.vega),
              10 * std::abs(cen.vega - exact.vega));
    // Forward Delta error ~ dS/2 * Gamma.
    EXPECT_NEAR(fwd.delta - exact.delta, 0.5 * big.dS * exact.gamma, 2e-4);
  }
}

TEST(Ch07FdGreeks, AnyPricer) {
  // A pricer with no closed-form Greeks in this library: a cash-or-nothing
  // call, e^(-rT) N(d2) (chapter 4). Its Delta is e^(-rT) phi(d2) / (S sigma
  // sqrt(T)).
  const double K = 100, q = 0.0;
  const auto digital = [&](double S, double T, double sigma, double r) {
    return std::exp(-r * T) * norm_cdf(ch03::bs_d1_d2(S, K, T, sigma, r, q).d2);
  };
  const double S = 95, T = 0.5, sigma = 0.25, r = 0.03;
  const auto g = fd_greeks(digital, S, T, sigma, r);
  const double d2 = ch03::bs_d1_d2(S, K, T, sigma, r).d2;
  EXPECT_NEAR(g.delta,
              std::exp(-r * T) * norm_pdf(d2) / (S * sigma * std::sqrt(T)),
              1e-7);
  // The digital also satisfies the Black-Scholes PDE.
  EXPECT_NEAR(pde_residual(digital(S, T, sigma, r), g, S, sigma, r), 0.0, 1e-6);
}

// -------------------------------------------------------------- 7.3.1 -----
TEST(Ch07FdGreeks, NumericalAccuracy) {
  const double S = 100, K = 100, T = 1, sigma = 0.2, r = 0.05;
  const auto exact = ch03::bs_greeks(OptionType::kCall, S, K, T, sigma, r);
  const auto call_S = [&](double s) {
    return ch03::bs_call(s, K, T, sigma, r);
  };
  // Gamma by the second difference: O(h^2) for large h, but round-off of
  // order eps C / h^2 ruins it for tiny h.
  const auto rows = fd_error_table(call_S, S, exact.gamma, FdScheme::Central,
                                   1.0, 9, 10.0, 2);
  EXPECT_NEAR(rows[0].error / rows[1].error, 100.0, 5.0);
  EXPECT_LT(std::abs(rows[2].error), 1e-7);  // h = 0.01
  EXPECT_GT(std::abs(rows[8].error),
            1e3 * std::abs(rows[2].error));  // h = 1e-8
}

// ---------------------------------------------------------------- 7.4 -----
TEST(Ch07BsPde, SolutionsAndNonSolutions) {
  const double sigma = 0.3, r = 0.05, q = 0.02, K = 50, T = 1.0;
  for (double S : {30.0, 50.0, 70.0}) {
    for (double t : {0.0, 0.4}) {
      for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
        const auto V = [&](double s, double u) {
          return ch03::bs_price(type, s, K, T - u, sigma, r, q);
        };
        EXPECT_NEAR(bs_pde_residual_fd(V, S, t, sigma, r, q), 0.0, 1e-5);
      }
      const auto fwd = [&](double s, double u) {
        return forward_contract_value(s, u, K, T, r, q);
      };
      EXPECT_NEAR(bs_pde_residual_fd(fwd, S, t, sigma, r, q), 0.0, 1e-6);
      for (double a : {-1.0, 0.0, 1.0, 2.0, 3.5}) {
        const auto V = [&](double s, double u) {
          return power_solution(a, s, u, sigma, r, q);
        };
        EXPECT_NEAR(bs_pde_residual_fd(V, S, t, sigma, r, q) / V(S, t), 0.0,
                    1e-6)
            << a;
      }
      // S^2 alone is not a solution, and the stock without dividend
      // reinvestment is not one when q > 0.
      EXPECT_GT(std::abs(bs_pde_residual_fd(
                    [](double s, double) { return s * s; }, S, t, sigma, r, q)),
                1.0);
      EXPECT_NEAR(bs_pde_residual_fd([](double s, double) { return s; }, S, t,
                                     sigma, r, q),
                  -q * S, 1e-6);
    }
  }
  EXPECT_DOUBLE_EQ(power_solution_rate(0, sigma, r, q), r);
  EXPECT_DOUBLE_EQ(power_solution_rate(1, sigma, r, q), q);
}

// -------------------------------------------------------------- 7.4.2 -----
TEST(Ch07BsPde, PdeAndGreeks) {
  const double S = 60, sigma = 0.25, r = 0.04, q = 0.01;
  // A portfolio: long a 3m 55 call, short two 6m 65 puts, short 0.3 shares.
  const auto c = ch03::bs_greeks(OptionType::kCall, S, 55, 0.25, sigma, r, q);
  const auto p = ch03::bs_greeks(OptionType::kPut, S, 65, 0.5, sigma, r, q);
  const Greeks share{1.0, 0.0, 0.0, 0.0, 0.0};  // V = S
  const double V = ch03::bs_call(S, 55, 0.25, sigma, r, q) -
                   2 * ch03::bs_put(S, 65, 0.5, sigma, r, q) - 0.3 * S;
  const auto g = combine_greeks({1.0, -2.0, -0.3}, {c, p, share});
  // Options solve the PDE; a share held without reinvesting dividends
  // leaves -q S, so the short 0.3 shares leave +0.3 q S.
  EXPECT_NEAR(pde_residual(V, g, S, sigma, r, q), 0.3 * q * S, 1e-10);

  // Delta-neutral (q = 0): Theta = r V - 1/2 sigma^2 S^2 Gamma, chapter 6's
  // theta_from_pde with Delta = 0.
  const auto c0 = ch03::bs_greeks(OptionType::kCall, S, 55, 0.25, sigma, r);
  const double V0 = ch03::bs_call(S, 55, 0.25, sigma, r) - c0.delta * S;
  const auto h = combine_greeks({1.0, -c0.delta}, {c0, share});
  EXPECT_NEAR(h.delta, 0.0, 1e-15);
  EXPECT_NEAR(h.theta, ch06::theta_from_pde(V0, S, 0.0, h.gamma, sigma, r),
              1e-10);
  EXPECT_LT(h.theta, 0.0);  // long Gamma pays through time decay
}

// -------------------------------------------------------------- 7.4.1 -----
TEST(Ch07BsPde, DeltaHedging) {
  const double S0 = 100, K = 105, T = 0.5, sigma = 0.2, r = 0.03, q = 0.01;
  std::mt19937_64 rng(1);
  const auto path = gbm_path(S0, T, sigma, 0.1, q, 50, rng);
  ASSERT_EQ(path.size(), 51u);
  EXPECT_DOUBLE_EQ(path[0], S0);

  // Error shrinks like 1/sqrt(n) and its mean is ~0, for any drift.
  for (double mu : {-0.1, 0.03, 0.2}) {
    const auto s25 = delta_hedge_stats(OptionType::kCall, S0, K, T, sigma, r, q,
                                       mu, 25, 2000, 7);
    const auto s400 = delta_hedge_stats(OptionType::kCall, S0, K, T, sigma, r,
                                        q, mu, 400, 2000, 7);
    EXPECT_NEAR(s25.std_dev / s400.std_dev, 4.0, 0.8) << mu;
    EXPECT_LT(std::abs(s400.mean), 4 * s400.std_dev / std::sqrt(2000.0) + 0.01)
        << mu;
    EXPECT_LT(s400.std_dev, 0.1 * ch03::bs_call(S0, K, T, sigma, r, q));
  }
  const auto put = delta_hedge_stats(OptionType::kPut, S0, K, T, sigma, r, q,
                                     0.05, 200, 1000, 3);
  EXPECT_LT(std::abs(put.mean), 0.05);
}
