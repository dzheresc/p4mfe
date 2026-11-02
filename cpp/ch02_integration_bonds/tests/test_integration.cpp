#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <stdexcept>

using namespace p4mfe::ch02;

namespace {
const auto kGauss = [](double x) { return std::exp(-x * x); };
// integral_0^2 e^(-x^2) dx = sqrt(pi)/2 erf(2), the book's 2.4.2 example.
const double kGauss02 = 0.5 * std::sqrt(std::numbers::pi) * std::erf(2.0);
}  // namespace

TEST(Ch02Rules, ExactForLowDegreePolynomials) {
  // Midpoint and trapezoidal are exact for lines, Simpson for cubics.
  const auto line = [](double x) { return 3 * x - 1; };
  const auto cubic = [](double x) { return x * x * x - 2 * x * x + x + 5; };
  EXPECT_NEAR(MidpointRule(line, 0, 2, 1), 4.0, 1e-14);
  EXPECT_NEAR(TrapezoidalRule(line, 0, 2, 1), 4.0, 1e-14);
  // integral_0^2 cubic = 4 - 16/3 + 2 + 10 = 32/3
  EXPECT_NEAR(SimpsonRule(cubic, 0, 2, 1), 32.0 / 3.0, 1e-13);
}

TEST(Ch02Rules, SimpsonIsWeightedTrapezoidalAndMidpoint) {
  for (std::size_t n : {1u, 4u, 17u})
    EXPECT_NEAR(
        SimpsonRule(kGauss, 0, 2, n),
        (TrapezoidalRule(kGauss, 0, 2, n) + 2 * MidpointRule(kGauss, 0, 2, n)) /
            3,
        1e-14);
}

TEST(Ch02Rules, MidpointAndTrapezoidalBracketConvexIntegrand) {
  // For convex f, midpoint underestimates and trapezoidal overestimates.
  const auto f = [](double x) { return std::exp(x); };
  const double exact = std::exp(1.0) - 1.0;
  EXPECT_LT(MidpointRule(f, 0, 1, 8), exact);
  EXPECT_GT(TrapezoidalRule(f, 0, 1, 8), exact);
  // ... and the midpoint error is about half the trapezoidal one.
  EXPECT_NEAR((exact - MidpointRule(f, 0, 1, 64)) /
                  (TrapezoidalRule(f, 0, 1, 64) - exact),
              0.5, 1e-3);
}

TEST(Ch02Convergence, ObservedOrders) {
  EXPECT_NEAR(ObservedOrder(kGauss, 0, 2, kGauss02, 32, Rule::kMidpoint), 2.0,
              0.05);
  EXPECT_NEAR(ObservedOrder(kGauss, 0, 2, kGauss02, 32, Rule::kTrapezoidal),
              2.0, 0.05);
  EXPECT_NEAR(ObservedOrder(kGauss, 0, 2, kGauss02, 16, Rule::kSimpson), 4.0,
              0.1);
}

TEST(Ch02Convergence, ConcreteExampleToTolerance) {
  // 2.4.2: integral_0^2 e^(-x^2) dx to tolerance 1e-7 with each rule.
  for (Rule rule : {Rule::kMidpoint, Rule::kTrapezoidal, Rule::kSimpson}) {
    const auto res = IntegrateToTolerance(kGauss, 0, 2, 1e-7, rule);
    EXPECT_NEAR(res.value, kGauss02, 1e-7);
  }
  // Simpson needs far fewer intervals than the second-order rules.
  EXPECT_LT(
      IntegrateToTolerance(kGauss, 0, 2, 1e-7, Rule::kSimpson).intervals,
      IntegrateToTolerance(kGauss, 0, 2, 1e-7, Rule::kMidpoint).intervals / 8);
}

TEST(Ch02Improper, InfiniteLimits) {
  EXPECT_NEAR(
      ImproperIntegral([](double x) { return 1 / (x * x); }, 1, kInfinity)
          .value,
      1.0, 1e-12);
  EXPECT_NEAR(
      ImproperIntegral([](double x) { return std::exp(-x); }, 0, kInfinity)
          .value,
      1.0, 1e-8);
  EXPECT_NEAR(ImproperIntegral([](double x) { return std::exp(-x * x / 2); },
                               -kInfinity, kInfinity)
                  .value,
              std::sqrt(2 * std::numbers::pi), 1e-8);
  EXPECT_NEAR(
      ImproperIntegral([](double x) { return std::exp(x); }, -kInfinity, 0)
          .value,
      1.0, 1e-8);
  EXPECT_NEAR(ImproperIntegral([](double x) { return 1 / (1 + x * x); },
                               -kInfinity, kInfinity)
                  .value,
              std::numbers::pi, 1e-8);
  EXPECT_THROW((void)ImproperIntegral(kGauss, 1, 0), std::invalid_argument);
}

TEST(Ch02Improper, GammaFunction) {
  EXPECT_NEAR(GammaFunction(1), 1.0, 1e-8);
  EXPECT_NEAR(GammaFunction(5), 24.0, 1e-7);
  EXPECT_NEAR(GammaFunction(2.5), std::tgamma(2.5), 1e-8);
  // Gamma(x + 1) = x Gamma(x)
  EXPECT_NEAR(GammaFunction(4.3), 3.3 * GammaFunction(3.3), 1e-7);
  // 0 < x < 2, x != 1: integrated after the substitution t = s^k.
  EXPECT_NEAR(GammaFunction(0.5), std::sqrt(std::numbers::pi), 1e-8);
  EXPECT_NEAR(GammaFunction(1.5), 0.5 * std::sqrt(std::numbers::pi), 1e-8);
  EXPECT_NEAR(GammaFunction(0.3), std::tgamma(0.3), 1e-7);
}

TEST(Ch02DifferentiateImproper, UnderTheIntegralSign) {
  // d/dt integral_0^inf e^(-t x) dx = d/dt (1/t) = -1/t^2
  const double t = 1.7;
  EXPECT_NEAR(
      ImproperIntegralDerivative(
          [](double x, double s) { return std::exp(-s * x); }, 0, kInfinity, t),
      -1 / (t * t), 1e-7);
  // F(t) = integral_0^inf e^(-x^2) cos(t x) dx = sqrt(pi)/2 e^(-t^2/4)
  const double expected =
      -t / 2 * std::sqrt(std::numbers::pi) / 2 * std::exp(-t * t / 4);
  EXPECT_NEAR(
      ImproperIntegralDerivative(
          [](double x, double s) { return std::exp(-x * x) * std::cos(s * x); },
          0, kInfinity, t),
      expected, 1e-7);
}
