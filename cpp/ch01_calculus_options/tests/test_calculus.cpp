#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch01/calculus.hpp>

using namespace p4mfe::ch01;

TEST(Ch01Derivative, ProductAndChainRule) {
  // (x^2 e^x)' = (x^2 + 2x) e^x ; (sin(x^2))' = 2x cos(x^2)
  const double x = 0.7;
  EXPECT_NEAR(derivative([](double t) { return t * t * std::exp(t); }, x),
              (x * x + 2 * x) * std::exp(x), 1e-9);
  EXPECT_NEAR(derivative([](double t) { return std::sin(t * t); }, x),
              2 * x * std::cos(x * x), 1e-9);
}

TEST(Ch01Derivative, Second) {
  EXPECT_NEAR(second_derivative([](double t) { return std::log(t); }, 2.0),
              -0.25, 1e-6);
}

TEST(Ch01Riemann, ConvergesToIntegral) {
  // integral_0^1 x e^x dx = 1 (integration by parts)
  const auto f = [](double x) { return x * std::exp(x); };
  EXPECT_NEAR(riemann_sum(f, 0, 1, 1000, RiemannRule::Midpoint), 1.0, 1e-6);
  EXPECT_NEAR(riemann_sum(f, 0, 1, 100000, RiemannRule::Left), 1.0, 1e-4);
  EXPECT_NEAR(riemann_sum(f, 0, 1, 100000, RiemannRule::Right), 1.0, 1e-4);
  // Left sums of an increasing function underestimate, right sums overestimate.
  EXPECT_LT(riemann_sum(f, 0, 1, 10, RiemannRule::Left), 1.0);
  EXPECT_GT(riemann_sum(f, 0, 1, 10, RiemannRule::Right), 1.0);
}

TEST(Ch01Leibniz, VariableLimitsAndIntegrand) {
  // I(t) = integral_0^t e^(-t x) dx = (1 - e^(-t^2)) / t
  // I'(t) = (2 t^2 e^(-t^2) - 1 + e^(-t^2)) / t^2
  const double t = 1.3;
  const double expected =
      (2 * t * t * std::exp(-t * t) - 1 + std::exp(-t * t)) / (t * t);
  const double got = integral_derivative(
      [](double x, double s) { return std::exp(-s * x); },
      [](double) { return 0.0; }, [](double s) { return s; }, t);
  EXPECT_NEAR(got, expected, 1e-6);
}

TEST(Ch01Limits, DiscreteCompoundingConvergesToContinuous) {
  const double r = 0.05, T = 2.0;
  const double limit = continuous_compounding_factor(r, T);
  double prev_err = std::abs(compounding_factor(r, T, 1) - limit);
  for (double n : {2.0, 12.0, 365.0, 1e6}) {
    const double err = std::abs(compounding_factor(r, T, n) - limit);
    EXPECT_LT(err, prev_err) << n;
    prev_err = err;
  }
  EXPECT_NEAR(compounding_factor(r, T, 1e6), limit, 1e-8);
}

TEST(Ch01Lhopital, FirstAndSecondOrder) {
  EXPECT_NEAR(lhopital_limit([](double x) { return std::sin(x); },
                             [](double x) { return x; }, 0.0),
              1.0, 1e-9);
  EXPECT_NEAR(lhopital_limit([](double x) { return 1 - std::cos(x); },
                             [](double x) { return x * x; }, 0.0),
              0.5, 1e-6);
  EXPECT_NEAR(lhopital_limit([](double x) { return std::log(x); },
                             [](double x) { return x - 1; }, 1.0),
              1.0, 1e-9);
}
