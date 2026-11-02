#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/exercises/ex01_06_antiderivatives.hpp>

using namespace p4mfe::ch01;
using namespace p4mfe::ch01::ex;

namespace {
// F(b) - F(a) must match the integral of f over [a, b].
template <typename F, typename Fp>
void expect_antiderivative(F&& F_, Fp&& f, double a, double b,
                           double tol = 1e-6) {
  EXPECT_NEAR(F_(b) - F_(a), riemann_sum(f, a, b, 20000),
              tol * std::max(1.0, std::abs(F_(b) - F_(a))));
  for (double x : {a, 0.5 * (a + b), b})
    EXPECT_NEAR(derivative(F_, x), f(x), 1e-6 * std::max(1.0, std::abs(f(x))));
}
}  // namespace

TEST(Ch01Ex01, LnAntiderivative) {
  expect_antiderivative(
      ex01_ln_antiderivative, [](double x) { return std::log(x); }, 0.5, 4.0);
  EXPECT_DOUBLE_EQ(ex01_ln_antiderivative(std::exp(1.0)), 0.0);  // e*1 - e
}

TEST(Ch01Ex02, InvXLnX) {
  const auto f = [](double x) { return 1.0 / (x * std::log(x)); };
  expect_antiderivative(ex02_inv_x_ln_x_antiderivative, f, 1.5, 10.0);
  expect_antiderivative(ex02_inv_x_ln_x_antiderivative, f, 0.1,
                        0.9);  // the (0, 1) branch
}

TEST(Ch01Ex03, TanAndArctan) {
  for (double x : {-1.2, -0.3, 0.0, 0.7, 1.4}) {
    EXPECT_NEAR(derivative([](double y) { return std::tan(y); }, x),
                ex03_tan_derivative(x), 1e-6 * ex03_tan_derivative(x));
    EXPECT_NEAR(ex03_tan_derivative(x), 1.0 + std::tan(x) * std::tan(x),
                1e-12 * ex03_tan_derivative(x));
  }
  expect_antiderivative(ex03_inv_1_plus_x2_antiderivative,
                        ex03_arctan_derivative, -3.0, 5.0);
  EXPECT_NEAR(riemann_sum(ex03_arctan_derivative, 0.0, 1.0, 10000),
              std::numbers::pi / 4.0, 1e-9);
}

TEST(Ch01Ex04, XPowNLn) {
  for (double n : {-3.0, -1.0, -0.5, 0.0, 1.0, 2.0, 4.5}) {
    expect_antiderivative(
        [n](double x) { return ex04_x_pow_n_ln_antiderivative(n, x); },
        [n](double x) { return std::pow(x, n) * std::log(x); }, 0.5, 3.0);
  }
  // n = 0 is exercise 1.
  EXPECT_DOUBLE_EQ(ex04_x_pow_n_ln_antiderivative(0.0, 2.0),
                   ex01_ln_antiderivative(2.0));
}

TEST(Ch01Ex05, XPowNExp) {
  for (unsigned n = 0; n <= 6; ++n) {
    expect_antiderivative(
        [n](double x) { return ex05_x_pow_n_exp_antiderivative(n, x); },
        [n](double x) { return std::pow(x, n) * std::exp(x); }, -2.0, 2.0);
  }
  const double x = 1.3;
  EXPECT_NEAR(ex05_x_pow_n_exp_antiderivative(2, x),
              (x * x - 2 * x + 2) * std::exp(x), 1e-14);
  // I_n(0) = (-1)^n n!.
  EXPECT_DOUBLE_EQ(ex05_x_pow_n_exp_antiderivative(4, 0.0), 24.0);
  EXPECT_DOUBLE_EQ(ex05_x_pow_n_exp_antiderivative(3, 0.0), -6.0);
}

TEST(Ch01Ex06, LnPowN) {
  for (unsigned n = 0; n <= 5; ++n) {
    expect_antiderivative(
        [n](double x) { return ex06_ln_pow_n_antiderivative(n, x); },
        [n](double x) { return std::pow(std::log(x), n); }, 0.5, 5.0);
  }
  EXPECT_NEAR(ex06_ln_pow_n_antiderivative(1, 2.0), ex01_ln_antiderivative(2.0),
              1e-14);
  EXPECT_THROW((void)ex06_ln_pow_n_antiderivative(2, -1.0), std::domain_error);
}
