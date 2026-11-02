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
  const double parts = ExCall(m.strike - x, m) - 2.0 * ExCall(m.strike, m) +
                       ExCall(m.strike + x, m);
  EXPECT_DOUBLE_EQ(
      Ex06ButterflyValue([&](double k) { return ExCall(k, m); }, m.strike, x),
      parts);
  EXPECT_GT(parts, 0.0);  // the tent payoff is >= 0
  // Payoff is the tent (x - |S - K|)^+.
  for (double spot : {40.0, 48.0, 49.0, 50.0, 50.5, 52.0, 60.0})
    EXPECT_NEAR(Ex06ButterflyPayoff(spot, m.strike, x),
                std::max(x - std::abs(spot - m.strike), 0.0), 1e-12)
        << spot;
}

TEST(Ch07Ex06, ScaledButterflyTendsToCkk) {
  const SpreadMarket m;
  const double ckk = Ex06CallKk(m);
  // C_KK from the closed form agrees with a difference of -C_K = e^(-rT) N(d2).
  const auto minus_ck = [&](double k) {
    SpreadMarket mk = m;
    mk.strike = k;
    return Ex07CashOrNothing(mk);
  };
  EXPECT_NEAR(-CentralDifference(minus_ck, m.strike, 1e-3), ckk, 1e-8);
  // Second order convergence in x.
  const auto err = [&](double x) { return Ex06ScaledButterfly(x, m) - ckk; };
  EXPECT_NEAR(ExObservedOrder(err, 1.0), 2.0, 0.1);
  EXPECT_LT(std::abs(err(0.05)), 1e-6);
}

TEST(Ch07Ex06, ArrowDebreuPayoffLimit) {
  const double strike = 50.0;
  for (double x : {1.0, 0.1, 1e-3, 1e-6})
    EXPECT_NEAR(Ex06ArrowDebreuPayoff(strike, strike, x), 1.0, 1e-12);
  for (double spot : {49.0, 49.99, 50.001, 51.0}) {
    EXPECT_GT(Ex06ArrowDebreuPayoff(spot, strike, 2.0), 0.0);
    EXPECT_EQ(Ex06ArrowDebreuPayoff(spot, strike, 1e-4), 0.0)
        << spot;  // x < |S - K|
  }
  // Payoffs are in [0, 1].
  for (double spot = 45.0; spot <= 55.0; spot += 0.25) {
    const double p = Ex06ArrowDebreuPayoff(spot, strike, 3.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
  }
}

TEST(Ch07Ex07, ScaledBullTendsToCashOrNothing) {
  const SpreadMarket m;
  const double limit = Ex07CashOrNothing(m);
  EXPECT_NEAR(-CentralDifference([&](double k) { return ExCall(k, m); },
                                 m.strike, 1e-3),
              limit, 1e-8);
  const auto err = [&](double x) { return Ex07ScaledBull(x, m) - limit; };
  EXPECT_NEAR(ExObservedOrder(err, 0.2), 1.0, 0.1);
  EXPECT_NEAR(err(0.01) / (-0.5 * 0.01 * Ex06CallKk(m)), 1.0, 0.02);
  // Increasing to the limit as x decreases (C is convex in K).
  double prev = Ex07ScaledBull(4.0, m);
  for (double x : {2.0, 1.0, 0.5, 0.1}) {
    const double v = Ex07ScaledBull(x, m);
    EXPECT_GT(v, prev);
    EXPECT_LT(v, limit);
    prev = v;
  }
}

TEST(Ch07Ex07, DigitalPayoffLimit) {
  const double strike = 50.0;
  for (double spot : {30.0, 49.9, 50.0})
    EXPECT_EQ(Ex07DigitalPayoff(spot, strike, 1e-3), 0.0) << spot;
  for (double spot : {50.01, 51.0, 80.0})
    EXPECT_NEAR(Ex07DigitalPayoff(spot, strike, 1e-3), 1.0, 1e-9) << spot;
  EXPECT_NEAR(Ex07DigitalPayoff(50.5, strike, 2.0), 0.25,
              1e-12);  // on the ramp
}

TEST(Ch07Ex08, NascentDelta) {
  const double strike = 50.0;
  // int f = 1 for every x.
  for (double x : {5.0, 1.0, 0.1})
    EXPECT_NEAR(Ex08Integral([](double) { return 1.0; }, strike, x), 1.0,
                1e-12);
  // Smooth g: int f g = g(K) + g''(K) x^2/12 + O(x^4).
  const auto g = [](double spot) { return std::cos(spot / 7.0); };
  const double g2 = -std::cos(strike / 7.0) / 49.0;
  for (double x : {2.0, 1.0, 0.5}) {
    const double integral = Ex08Integral(g, strike, x);
    EXPECT_NEAR((integral - g(strike)) / (x * x / 12.0 * g2), 1.0, 0.01) << x;
  }
  // Merely continuous g (a kink at 50.3): the limit is still g(K).
  const auto gk = [](double spot) {
    return std::abs(spot - 50.3) + std::sqrt(spot);
  };
  for (double x : {1.0, 0.1, 1e-3})
    EXPECT_LE(std::abs(Ex08Integral(gk, strike, x) - gk(strike)), 2.0 * x) << x;
  EXPECT_NEAR(Ex08Integral(gk, strike, 1e-4), gk(strike), 1e-4);
}
