#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch07/exercises/ex01_02_difference_orders.hpp>
#include <p4mfe/ch07/exercises/ex06_butterfly_spread.hpp>
#include <p4mfe/ch07/exercises/ex07_bull_spread.hpp>
#include <p4mfe/ch07/exercises/ex08_butterfly_delta.hpp>
#include <p4mfe/ch07/finite_differences.hpp>

using namespace p4mfe;
using namespace p4mfe::ch07;
using namespace p4mfe::ch07::ex;

TEST(Ch07Ex06, ButterflyValueIsSumOfParts) {
  const SpreadMarket m;
  const double x = 2.0;
  const double parts =
      ex_call(m.K - x, m) - 2.0 * ex_call(m.K, m) + ex_call(m.K + x, m);
  EXPECT_DOUBLE_EQ(
      ex06_butterfly_value([&](double k) { return ex_call(k, m); }, m.K, x),
      parts);
  EXPECT_GT(parts, 0.0);  // the tent payoff is >= 0
  // Payoff is the tent (x - |S - K|)^+.
  for (double S : {40.0, 48.0, 49.0, 50.0, 50.5, 52.0, 60.0})
    EXPECT_NEAR(ex06_butterfly_payoff(S, m.K, x),
                std::max(x - std::abs(S - m.K), 0.0), 1e-12)
        << S;
}

TEST(Ch07Ex06, ScaledButterflyTendsToCkk) {
  const SpreadMarket m;
  const double ckk = ex06_call_kk(m);
  // C_KK from the closed form agrees with a difference of -C_K = e^(-rT) N(d2).
  const auto minus_ck = [&](double k) {
    SpreadMarket mk = m;
    mk.K = k;
    return ex07_cash_or_nothing(mk);
  };
  EXPECT_NEAR(-central_difference(minus_ck, m.K, 1e-3), ckk, 1e-8);
  // Second order convergence in x.
  const auto err = [&](double x) { return ex06_scaled_butterfly(x, m) - ckk; };
  EXPECT_NEAR(ex_observed_order(err, 1.0), 2.0, 0.1);
  EXPECT_LT(std::abs(err(0.05)), 1e-6);
}

TEST(Ch07Ex06, ArrowDebreuPayoffLimit) {
  const double K = 50.0;
  for (double x : {1.0, 0.1, 1e-3, 1e-6})
    EXPECT_NEAR(ex06_arrow_debreu_payoff(K, K, x), 1.0, 1e-12);
  for (double S : {49.0, 49.99, 50.001, 51.0}) {
    EXPECT_GT(ex06_arrow_debreu_payoff(S, K, 2.0), 0.0);
    EXPECT_EQ(ex06_arrow_debreu_payoff(S, K, 1e-4), 0.0) << S;  // x < |S - K|
  }
  // Payoffs are in [0, 1].
  for (double S = 45.0; S <= 55.0; S += 0.25) {
    const double p = ex06_arrow_debreu_payoff(S, K, 3.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
  }
}

TEST(Ch07Ex07, ScaledBullTendsToCashOrNothing) {
  const SpreadMarket m;
  const double limit = ex07_cash_or_nothing(m);
  EXPECT_NEAR(
      -central_difference([&](double k) { return ex_call(k, m); }, m.K, 1e-3),
      limit, 1e-8);
  const auto err = [&](double x) { return ex07_scaled_bull(x, m) - limit; };
  EXPECT_NEAR(ex_observed_order(err, 0.2), 1.0, 0.1);
  EXPECT_NEAR(err(0.01) / (-0.5 * 0.01 * ex06_call_kk(m)), 1.0, 0.02);
  // Increasing to the limit as x decreases (C is convex in K).
  double prev = ex07_scaled_bull(4.0, m);
  for (double x : {2.0, 1.0, 0.5, 0.1}) {
    const double v = ex07_scaled_bull(x, m);
    EXPECT_GT(v, prev);
    EXPECT_LT(v, limit);
    prev = v;
  }
}

TEST(Ch07Ex07, DigitalPayoffLimit) {
  const double K = 50.0;
  for (double S : {30.0, 49.9, 50.0})
    EXPECT_EQ(ex07_digital_payoff(S, K, 1e-3), 0.0) << S;
  for (double S : {50.01, 51.0, 80.0})
    EXPECT_NEAR(ex07_digital_payoff(S, K, 1e-3), 1.0, 1e-9) << S;
  EXPECT_NEAR(ex07_digital_payoff(50.5, K, 2.0), 0.25, 1e-12);  // on the ramp
}

TEST(Ch07Ex08, NascentDelta) {
  const double K = 50.0;
  // int f = 1 for every x.
  for (double x : {5.0, 1.0, 0.1})
    EXPECT_NEAR(ex08_integral([](double) { return 1.0; }, K, x), 1.0, 1e-12);
  // Smooth g: int f g = g(K) + g''(K) x^2/12 + O(x^4).
  const auto g = [](double S) { return std::cos(S / 7.0); };
  const double g2 = -std::cos(K / 7.0) / 49.0;
  for (double x : {2.0, 1.0, 0.5}) {
    const double I = ex08_integral(g, K, x);
    EXPECT_NEAR((I - g(K)) / (x * x / 12.0 * g2), 1.0, 0.01) << x;
  }
  // Merely continuous g (a kink at 50.3): the limit is still g(K).
  const auto gk = [](double S) { return std::abs(S - 50.3) + std::sqrt(S); };
  for (double x : {1.0, 0.1, 1e-3})
    EXPECT_LE(std::abs(ex08_integral(gk, K, x) - gk(K)), 2.0 * x) << x;
  EXPECT_NEAR(ex08_integral(gk, K, 1e-4), gk(K), 1e-4);
}
