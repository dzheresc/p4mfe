#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch08/exercises/ex01_gamma_maximum.hpp>
#include <p4mfe/ch08/exercises/ex02_03_double_integrals.hpp>
#include <p4mfe/ch08/exercises/ex04_05_polar_areas.hpp>
#include <p4mfe/ch08/exercises/ex06_07_exp_inequalities.hpp>
#include <p4mfe/ch08/exercises/ex08_gronwall.hpp>

using namespace p4mfe;
using namespace p4mfe::ch08::ex;

TEST(Ch08Ex01, GammaRisesThenFalls) {
  for (const Ex01Market m : {Ex01Market{}, Ex01Market{100.0, 2.0, 0.5, 0.02},
                             Ex01Market{40.0, 0.1, 0.15, 0.0}}) {
    const double s = Ex01Argmax(m);
    EXPECT_NEAR(Ex01Gamma(s, m), Ex01MaxGamma(m), 1e-12 * Ex01MaxGamma(m));
    // d1(S*) = -sigma sqrt(T), and the log-derivative changes sign there.
    EXPECT_NEAR(ch03::BsD1D2(s, m.strike, m.maturity, m.sigma, m.r).d1,
                -m.sigma * std::sqrt(m.maturity), 1e-12);
    // Strictly increasing on a grid below S*, strictly decreasing above.
    double prev = 0.0;
    for (double spot = 0.2 * s; spot <= s; spot += 0.01 * s) {
      const double g = Ex01Gamma(spot, m);
      EXPECT_GT(g, prev) << spot;
      EXPECT_GT(Ex01LogDerivative(spot, m), 0.0);
      prev = g;
    }
    prev = Ex01Gamma(s, m);
    for (double spot = 1.01 * s; spot <= 5.0 * s; spot += 0.01 * s) {
      const double g = Ex01Gamma(spot, m);
      EXPECT_LT(g, prev) << spot;
      EXPECT_LT(Ex01LogDerivative(spot, m), 0.0);
      prev = g;
    }
    // Global maximum on a wide grid.
    for (double spot = 1e-3 * m.strike; spot < 1e3 * m.strike; spot *= 1.05)
      EXPECT_LE(Ex01Gamma(spot, m), Ex01MaxGamma(m));
    // Limits at 0 and infinity.
    EXPECT_LT(Ex01Gamma(1e-3 * m.strike, m), 1e-6 * Ex01MaxGamma(m));
    EXPECT_LT(Ex01Gamma(1e3 * m.strike, m), 1e-6 * Ex01MaxGamma(m));
  }
}

TEST(Ch08Ex02, IntegralIsZero) {
  EXPECT_NEAR(Ex02Direct(), kEx02Exact, 1e-6);
  EXPECT_NEAR(Ex02ChangeOfVariables(), kEx02Exact, 1e-12);
  // The quadrature resolves the integrand: |x - y|/(x + y) integrates to 1/4.
  EXPECT_NEAR(Ex02AbsDirect(400), kEx02AbsExact, 1e-3);
}

TEST(Ch08Ex03, ParabolaAndLine) {
  // Simpson is exact for the cubic inner integrand and the quadratic in y.
  EXPECT_NEAR(Ex03Numeric(), kEx03Exact, 1e-10);
  for (double x : {1.0, 1.3, 1.7, 2.0}) {
    const double lo = (x + 1.0) * (x + 1.0), hi = 5.0 * x - 1.0;
    EXPECT_NEAR((x * x * hi - hi * hi) - (x * x * lo - lo * lo), Ex03Inner(x),
                1e-12);
  }
  EXPECT_NEAR(ch02::SimpsonRule(Ex03Inner, 1.0, 2.0, 2), -1.75, 1e-14);
}

TEST(Ch08Ex04, DiskArea) {
  for (double radius : {0.5, 1.0, 3.0}) {
    EXPECT_NEAR(Ex04DiskAreaPolar(radius), std::numbers::pi * radius * radius,
                1e-10 * radius * radius);
    EXPECT_NEAR(Ex04DiskAreaCartesian(radius),
                std::numbers::pi * radius * radius, 1e-4 * radius * radius);
  }
}

TEST(Ch08Ex05, EllipseArea) {
  for (auto [a, b] :
       {std::pair{1.0, 1.0}, std::pair{3.0, 0.5}, std::pair{0.2, 7.0}}) {
    EXPECT_NEAR(Ex05EllipseArea(a, b), Ex05Exact(a, b), 1e-6 * Ex05Exact(a, b));
    EXPECT_NEAR(
        ch08::JacobianDeterminant(
            [&](double r, double t) { return Ex05Map(a, b, r, t); }, 0.7, 1.1),
        a * b * 0.7, 1e-7 * a * b);
  }
}

TEST(Ch08Ex06, EPowPiIsLarger) {
  EXPECT_GT(Ex06EPowPi(), Ex06PiPowE());
  EXPECT_NEAR(Ex06EPowPi(), 23.140693, 1e-6);
  EXPECT_NEAR(Ex06PiPowE(), 22.459158, 1e-6);
  EXPECT_LT(Ex06F(std::numbers::pi), Ex06F(std::numbers::e));
  for (double x = 0.5; x < 20.0; x += 0.05)
    if (std::abs(x - std::numbers::e) > 1e-9) {
      EXPECT_LT(Ex06F(x), 1.0 / std::numbers::e);
    }
}

TEST(Ch08Ex07, ConvexityOfExp) {
  for (double x = -5.0; x <= 5.0; x += 0.25)
    for (double y = -5.0; y <= 5.0; y += 0.25) {
      EXPECT_GE(Ex07Lhs(x, y), Ex07Rhs(x, y) * (1.0 - 1e-15));
      EXPECT_NEAR(Ex07Lhs(x, y) - Ex07Rhs(x, y), Ex07Gap(x, y),
                  1e-13 * Ex07Lhs(x, y));
    }
  EXPECT_EQ(Ex07Gap(1.3, 1.3), 0.0);
  EXPECT_GT(Ex07Gap(1.3, 1.3 + 1e-6), 0.0);
}

TEST(Ch08Ex08, Gronwall) {
  const double bound = 2.0, x_end = 6.0;
  // u = M e^(V/2): the hypothesis holds strictly.
  const auto c1 = Ex08ComputeCheck(
      [&](double x) { return bound * std::exp(0.5 * Ex08Integral(x)); },
      Ex08Integrand, bound, x_end);
  // The gaps are 0 at x = 0 (u(0) = M) and negative after.
  EXPECT_LE(c1.hypothesis_gap, 0.0);
  EXPECT_LE(c1.conclusion_gap, 1e-12);
  EXPECT_LT(c1.g_increase, 1e-6);
  // u = M (1 + V/2): M + integral u v = M (1 + V/2)^2 >= u.
  const auto c2 = Ex08ComputeCheck(
      [&](double x) { return bound * (1.0 + 0.5 * Ex08Integral(x)); },
      Ex08Integrand, bound, x_end);
  EXPECT_LE(c2.hypothesis_gap, 1e-9);
  EXPECT_LE(c2.conclusion_gap, 1e-12);
  EXPECT_LT(c2.g_increase, 1e-6);
  // u = M e^V: equality, so the bound is sharp (the gaps are the O(h^2)
  // error of the trapezoidal integrals, h = 0.003).
  const auto c3 = Ex08ComputeCheck(
      [&](double x) { return bound * std::exp(Ex08Integral(x)); },
      Ex08Integrand, bound, x_end);
  EXPECT_NEAR(c3.conclusion_gap, 0.0, 1e-5);
  EXPECT_LT(std::abs(c3.hypothesis_gap),
            1e-3 * bound * std::exp(Ex08Integral(x_end)));
  EXPECT_LT(c3.g_increase, 1e-5);
}
