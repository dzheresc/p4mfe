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
    const double s = ex01_argmax(m);
    EXPECT_NEAR(ex01_gamma(s, m), ex01_max_gamma(m), 1e-12 * ex01_max_gamma(m));
    // d1(S*) = -sigma sqrt(T), and the log-derivative changes sign there.
    EXPECT_NEAR(ch03::bs_d1_d2(s, m.K, m.T, m.sigma, m.r).d1,
                -m.sigma * std::sqrt(m.T), 1e-12);
    // Strictly increasing on a grid below S*, strictly decreasing above.
    double prev = 0.0;
    for (double S = 0.2 * s; S <= s; S += 0.01 * s) {
      const double g = ex01_gamma(S, m);
      EXPECT_GT(g, prev) << S;
      EXPECT_GT(ex01_log_derivative(S, m), 0.0);
      prev = g;
    }
    prev = ex01_gamma(s, m);
    for (double S = 1.01 * s; S <= 5.0 * s; S += 0.01 * s) {
      const double g = ex01_gamma(S, m);
      EXPECT_LT(g, prev) << S;
      EXPECT_LT(ex01_log_derivative(S, m), 0.0);
      prev = g;
    }
    // Global maximum on a wide grid.
    for (double S = 1e-3 * m.K; S < 1e3 * m.K; S *= 1.05)
      EXPECT_LE(ex01_gamma(S, m), ex01_max_gamma(m));
    // Limits at 0 and infinity.
    EXPECT_LT(ex01_gamma(1e-3 * m.K, m), 1e-6 * ex01_max_gamma(m));
    EXPECT_LT(ex01_gamma(1e3 * m.K, m), 1e-6 * ex01_max_gamma(m));
  }
}

TEST(Ch08Ex02, IntegralIsZero) {
  EXPECT_NEAR(ex02_direct(), ex02_exact, 1e-6);
  EXPECT_NEAR(ex02_change_of_variables(), ex02_exact, 1e-12);
  // The quadrature resolves the integrand: |x - y|/(x + y) integrates to 1/4.
  EXPECT_NEAR(ex02_abs_direct(400), ex02_abs_exact, 1e-3);
}

TEST(Ch08Ex03, ParabolaAndLine) {
  // Simpson is exact for the cubic inner integrand and the quadratic in y.
  EXPECT_NEAR(ex03_numeric(), ex03_exact, 1e-10);
  for (double x : {1.0, 1.3, 1.7, 2.0}) {
    const double lo = (x + 1.0) * (x + 1.0), hi = 5.0 * x - 1.0;
    EXPECT_NEAR((x * x * hi - hi * hi) - (x * x * lo - lo * lo), ex03_inner(x),
                1e-12);
  }
  EXPECT_NEAR(ch02::SimpsonRule(ex03_inner, 1.0, 2.0, 2), -1.75, 1e-14);
}

TEST(Ch08Ex04, DiskArea) {
  for (double R : {0.5, 1.0, 3.0}) {
    EXPECT_NEAR(ex04_disk_area_polar(R), std::numbers::pi * R * R,
                1e-10 * R * R);
    EXPECT_NEAR(ex04_disk_area_cartesian(R), std::numbers::pi * R * R,
                1e-4 * R * R);
  }
}

TEST(Ch08Ex05, EllipseArea) {
  for (auto [a, b] :
       {std::pair{1.0, 1.0}, std::pair{3.0, 0.5}, std::pair{0.2, 7.0}}) {
    EXPECT_NEAR(ex05_ellipse_area(a, b), ex05_exact(a, b),
                1e-6 * ex05_exact(a, b));
    EXPECT_NEAR(
        ch08::jacobian_determinant(
            [&](double r, double t) { return ex05_map(a, b, r, t); }, 0.7, 1.1),
        a * b * 0.7, 1e-7 * a * b);
  }
}

TEST(Ch08Ex06, EPowPiIsLarger) {
  EXPECT_GT(ex06_e_pow_pi(), ex06_pi_pow_e());
  EXPECT_NEAR(ex06_e_pow_pi(), 23.140693, 1e-6);
  EXPECT_NEAR(ex06_pi_pow_e(), 22.459158, 1e-6);
  EXPECT_LT(ex06_f(std::numbers::pi), ex06_f(std::numbers::e));
  for (double x = 0.5; x < 20.0; x += 0.05)
    if (std::abs(x - std::numbers::e) > 1e-9) {
      EXPECT_LT(ex06_f(x), 1.0 / std::numbers::e);
    }
}

TEST(Ch08Ex07, ConvexityOfExp) {
  for (double x = -5.0; x <= 5.0; x += 0.25)
    for (double y = -5.0; y <= 5.0; y += 0.25) {
      EXPECT_GE(ex07_lhs(x, y), ex07_rhs(x, y) * (1.0 - 1e-15));
      EXPECT_NEAR(ex07_lhs(x, y) - ex07_rhs(x, y), ex07_gap(x, y),
                  1e-13 * ex07_lhs(x, y));
    }
  EXPECT_EQ(ex07_gap(1.3, 1.3), 0.0);
  EXPECT_GT(ex07_gap(1.3, 1.3 + 1e-6), 0.0);
}

TEST(Ch08Ex08, Gronwall) {
  const double M = 2.0, X = 6.0;
  // u = M e^(V/2): the hypothesis holds strictly.
  const auto c1 = ex08_check(
      [&](double x) { return M * std::exp(0.5 * ex08_V(x)); }, ex08_v, M, X);
  // The gaps are 0 at x = 0 (u(0) = M) and negative after.
  EXPECT_LE(c1.hypothesis_gap, 0.0);
  EXPECT_LE(c1.conclusion_gap, 1e-12);
  EXPECT_LT(c1.g_increase, 1e-6);
  // u = M (1 + V/2): M + integral u v = M (1 + V/2)^2 >= u.
  const auto c2 = ex08_check(
      [&](double x) { return M * (1.0 + 0.5 * ex08_V(x)); }, ex08_v, M, X);
  EXPECT_LE(c2.hypothesis_gap, 1e-9);
  EXPECT_LE(c2.conclusion_gap, 1e-12);
  EXPECT_LT(c2.g_increase, 1e-6);
  // u = M e^V: equality, so the bound is sharp (the gaps are the O(h^2)
  // error of the trapezoidal integrals, h = 0.003).
  const auto c3 = ex08_check([&](double x) { return M * std::exp(ex08_V(x)); },
                             ex08_v, M, X);
  EXPECT_NEAR(c3.conclusion_gap, 0.0, 1e-5);
  EXPECT_LT(std::abs(c3.hypothesis_gap), 1e-3 * M * std::exp(ex08_V(X)));
  EXPECT_LT(c3.g_increase, 1e-5);
}
