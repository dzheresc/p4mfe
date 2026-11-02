#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch06/exercises/ex01_sqrt_taylor.hpp>
#include <p4mfe/ch06/exercises/ex02_exp_six_digits.hpp>
#include <p4mfe/ch06/exercises/ex03_big_o.hpp>
#include <p4mfe/ch06/exercises/ex04_exp_composite_taylor.hpp>
#include <p4mfe/ch06/exercises/ex05_even_series.hpp>
#include <p4mfe/ch06/exercises/ex06_e_bounds.hpp>
#include <p4mfe/ch06/exercises/ex07_log_ratio_series.hpp>
#include <p4mfe/ch06/exercises/ex08_cosh_series.hpp>
#include <p4mfe/ch06/exercises/ex09_log_series_convergence.hpp>

using namespace p4mfe;
using namespace p4mfe::ch06;
using namespace p4mfe::ch06::ex;

TEST(Ch06Ex01, CubicTaylorOfSqrt) {
  // The derivatives at 0 match finite differences of sqrt(1 + x).
  const auto f = [](double x) { return std::sqrt(1.0 + x); };
  EXPECT_NEAR(ch01::derivative(f, 0.0), ex01_derivatives_at_zero[1], 1e-8);
  EXPECT_NEAR(ch01::second_derivative(f, 0.0), ex01_derivatives_at_zero[2],
              1e-5);
  for (double x : {-0.5, -0.1, 0.0, 0.2, 0.7}) {
    EXPECT_NEAR(ex01_cubic(x), ex01_cubic_from_derivatives(x), 1e-15);
    EXPECT_NEAR(ex01_cubic(x), binomial_series(x, 0.5, 3), 1e-15);
  }
  // The error is -(5/128) x^4 + O(x^5).
  for (double x : {1e-2, -1e-2, 1e-3}) {
    const double err = f(x) - ex01_cubic(x);
    EXPECT_NEAR(err / std::pow(x, 4), ex01_error_coefficient,
                0.1 * std::abs(ex01_error_coefficient));
  }
}

TEST(Ch06Ex02, ExpQuarterSixDigits) {
  const auto r = ex02_solve();
  EXPECT_EQ(r.n, 5u);
  EXPECT_NEAR(r.value, 1.2840250651, 1e-10);
  EXPECT_LT(r.bound, 0.5e-6);
  EXPECT_GE(ex02_error_bound(0.25, 4), 0.5e-6);  // n = 4 would not do
  // The bound holds and the answer is correct to six decimals.
  EXPECT_LT(std::exp(0.25) - r.value, r.bound);
  EXPECT_GT(std::exp(0.25) - r.value, 0.0);
  EXPECT_NEAR(std::round(r.value * 1e6), std::round(std::exp(0.25) * 1e6), 0.5);
}

TEST(Ch06Ex03, DifferenceIsBigOOfXSquared) {
  for (double x = -0.5; x <= 0.5; x += 0.01) {
    const double d = ex03_difference(x);
    EXPECT_NEAR(d, std::exp(-x) - 1.0 / (1.0 + x), 1e-14);
    EXPECT_LE(std::abs(d), ex03_big_o_constant * x * x + 1e-16);
  }
  // d / x^2 -> -1/2, and the next term is (5/6) x^3.
  for (double x : {1e-2, -1e-2, 1e-3, 1e-4}) {
    EXPECT_NEAR(ex03_difference(x) / (x * x), -0.5, 2.0 * std::abs(x));
    EXPECT_NEAR((ex03_difference(x) - ex03_two_terms(x)) / std::pow(x, 4), 0.0,
                1.0);
  }
}

TEST(Ch06Ex04, QuadraticTaylorOfExpG) {
  const auto c2 = ex04_part_ii();
  const auto c3 = ex04_part_iii();
  const double e = std::numbers::e;
  for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(c2[i], c3[i], 1e-14);
  EXPECT_NEAR(c2[0], e, 1e-15);
  EXPECT_NEAR(c2[1], 2.0 * e, 1e-14);
  EXPECT_NEAR(c2[2], 3.0 * e, 1e-14);
  // Error O(x^3) against e^((x+1)^2).
  const auto f = [](double x) { return std::exp((x + 1.0) * (x + 1.0)); };
  for (double x : {1e-2, -1e-2, 2e-3}) {
    const double err = f(x) - ex04_eval(c2, x);
    EXPECT_LT(std::abs(err), 20.0 * std::abs(x * x * x));
  }
  // (i) for another g: g(x) = sin(x) + 0.5 (g(0) = 0.5, g' = 1, g'' = 0),
  // checked against finite-difference derivatives of e^g.
  const auto h = [](double x) { return std::exp(std::sin(x) + 0.5); };
  const auto q = ex04_quadratic(0.5, 1.0, 0.0);
  EXPECT_NEAR(q[1], ch01::derivative(h, 0.0), 1e-8);
  EXPECT_NEAR(2.0 * q[2], ch01::second_derivative(h, 0.0), 1e-5);
  const auto l = ex04_linear(0.5, 1.0);
  EXPECT_NEAR(l[0], q[0], 1e-15);
  EXPECT_NEAR(l[1], q[1], 1e-15);
}

TEST(Ch06Ex05, EvenSeries) {
  for (double x : {-0.9, -0.5, 0.1, 0.6, 0.8}) {
    EXPECT_NEAR(ex05_log_series(x, 400), std::log1p(-x * x), 1e-12);
    EXPECT_NEAR(ex05_inverse_series(x, 400), 1.0 / (1.0 - x * x), 1e-12);
    // Direct sums of the stated series.
    double s = 0.0, g = 0.0;
    for (unsigned m = 0; m <= 20; ++m) {
      s += ex05_log_coefficient(m) * std::pow(x, m);
      g += ex05_inverse_coefficient(m) * std::pow(x, m);
    }
    EXPECT_NEAR(s, ex05_log_series(x, 10), 1e-14);
    EXPECT_NEAR(g, ex05_inverse_series(x, 10), 1e-13);
  }
}

TEST(Ch06Ex06, BoundsForE) {
  for (double x = 1.0; x <= 1000.0; x *= 1.1) {
    EXPECT_LT(ex06_log_gap_lower(x), 0.0) << x;
    EXPECT_GT(ex06_log_gap_upper(x), 0.0) << x;
    EXPECT_LT(ex06_lower(x), std::numbers::e * (1.0 + 1e-15));
    EXPECT_GT(ex06_upper(x), std::numbers::e * (1.0 - 1e-15));
    // Sharper than the hint's (1 + 1/x)^x and (1 + 1/x)^(x+1).
    EXPECT_GT(ex06_lower_exponent(x), x);
    EXPECT_LT(ex06_upper_exponent(x), x + 1.0);
  }
  // Rates: the gaps are about -t^3/24 and t^2/12, t = 1/x.
  for (double x : {100.0, 1000.0}) {
    const double t = 1.0 / x;
    EXPECT_NEAR(ex06_log_gap_lower(x) / (t * t * t), -1.0 / 24.0, 0.05 / 24.0);
    EXPECT_NEAR(ex06_log_gap_upper(x) / (t * t), 1.0 / 12.0, 0.05 / 12.0);
  }
  // The proof's auxiliary functions: zero at 0, closed-form derivatives match,
  // positive.
  EXPECT_EQ(ex06_h(0.0), 0.0);
  EXPECT_EQ(ex06_k(0.0), 0.0);
  for (double t = 0.05; t <= 1.0; t += 0.05) {
    EXPECT_NEAR(ch01::derivative(ex06_h, t), ex06_h_prime(t), 1e-8);
    EXPECT_NEAR(ch01::derivative(ex06_k, t), ex06_k_prime(t), 1e-8);
    EXPECT_GT(ex06_h(t), 0.0);
    EXPECT_GT(ex06_k(t), 0.0);
  }
}

TEST(Ch06Ex07, LogRatioSeries) {
  for (double x : {-0.95, -0.5, 0.0, 0.3, 0.9}) {
    EXPECT_NEAR(ex07_series(x, 2000), ex07_function(x), 1e-12);
    EXPECT_NEAR(ex07_series(x, 2000), 2.0 * std::atanh(x), 1e-12);
    EXPECT_NEAR(ex07_series(x, 10), ex07_series_from_log1p(x, 10), 1e-14);
  }
  EXPECT_EQ(ex07_radius, 1.0);
  // Outside the radius the partial sums blow up.
  EXPECT_GT(ex07_series(1.1, 200), 1e6);
}

TEST(Ch06Ex08, CoshSeries) {
  for (double x : {0.0, 0.5, 1.0, 2.0, -3.0}) {
    const double f = ex08_function(x);
    EXPECT_NEAR(ex08_series(x, 60), f, 1e-13 * f);
    EXPECT_NEAR(ex08_series(x, 8), ex08_series_from_exp(x, 8), 1e-12 * f);
    EXPECT_NEAR(f, 0.5 * (std::exp(x * x) + std::exp(-x * x)), 1e-14 * f);
  }
  EXPECT_TRUE(std::isinf(ex08_radius));
}

TEST(Ch06Ex09, LogSeriesConvergesOnTheWholeInterval) {
  for (double x : {-0.9, -0.6, -0.3, 0.3, 0.6, 0.9}) {
    for (unsigned n : {1u, 2u, 5u, 10u}) {
      const double err = ex09_error(x, n);
      // (i) the integral form of the remainder
      EXPECT_NEAR(ex09_error_integral(x, n), err, 1e-10) << x << " " << n;
      // (ii)/(iii) the bounds
      EXPECT_LE(std::abs(err), ex09_bound(x, n) + 1e-14);
      if (x > 0) {
        EXPECT_LE(std::abs(err), ex09_middle_bound(x, n) + 1e-12);
        EXPECT_LE(ex09_middle_bound(x, n), ex09_bound(x, n) + 1e-12);
      } else {
        EXPECT_NEAR(ex09_error_negative(x, n), std::abs(err), 1e-10);
      }
    }
    // (6.128): the error goes to zero.
    EXPECT_LT(std::abs(ex09_error(x, 600)), 1e-12);
    EXPECT_LT(ex09_bound(x, 600), 1e-12);
  }
}
