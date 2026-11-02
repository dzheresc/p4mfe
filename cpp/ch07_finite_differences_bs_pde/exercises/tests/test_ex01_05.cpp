#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch07/exercises/ex01_02_difference_orders.hpp>
#include <p4mfe/ch07/exercises/ex03_04_stencils.hpp>
#include <p4mfe/ch07/exercises/ex05_nonsymmetric_stencils.hpp>

using namespace p4mfe;
using namespace p4mfe::ch07;
using namespace p4mfe::ch07::ex;

// Steps are kept large enough (h >= 0.05) that truncation dominates round-off
// on any IEEE double machine, so the observed orders are stable.

TEST(Ch07Ex01, CentralDifferenceIsFourthOrderAtZero) {
  EXPECT_NEAR(ex_observed_order(ex01_error, 0.1), 4.0, 0.1);
  EXPECT_NEAR(ex_observed_order(ex01_error, 0.05), 4.0, 0.05);
  EXPECT_NEAR(ex01_error(0.05) / std::pow(0.05, 4), ex01_leading_coefficient,
              0.01);
  // At a = 1, where f''' != 0, the same scheme is only second order.
  const auto err1 = [](double h) {
    const double exact =
        (1.0 + 3.0 - 6.0) * std::exp(1.0);  // (x^3 + 3x^2 - 6) e^x
    return central_difference(ex01_f, 1.0, h) - exact;
  };
  EXPECT_NEAR(ex_observed_order(err1, 0.05), 2.0, 0.1);
}

TEST(Ch07Ex02, ForwardDifferenceOfNIsSecondOrderAtZero) {
  EXPECT_NEAR(ex_observed_order(ex02_error, 0.1), 2.0, 0.05);
  EXPECT_NEAR(ex02_error(0.05) / (0.05 * 0.05), ex02_leading_coefficient,
              1e-3 * 0.07);
  // At x = 1 (N''(1) != 0) the forward difference is first order.
  const auto err1 = [](double h) {
    return forward_difference([](double x) { return norm_cdf(x); }, 1.0, h) -
           norm_pdf(1.0);
  };
  EXPECT_NEAR(ex_observed_order(err1, 0.05), 1.0, 0.1);
}

TEST(Ch07Ex03, OneSidedSecondOrder) {
  const double a = 0.3;
  const auto f = [](double x) { return std::exp(x); };
  const auto err = [&](double h) {
    return ex03_one_sided(f, a, h) - std::exp(a);
  };
  EXPECT_NEAR(ex_observed_order(err, 0.1), 2.0, 0.1);
  EXPECT_NEAR(err(0.05) / ex03_leading_error(std::exp(a), 0.05), 1.0, 0.05);
  // Exact for quadratics; for a cubic the error is exactly -h^2 f'''/3.
  const auto quad = [](double x) { return 2.0 * x * x - 3.0 * x + 1.0; };
  EXPECT_NEAR(ex03_one_sided(quad, a, 0.25), 4.0 * a - 3.0, 1e-12);
  const auto cube = [](double x) { return x * x * x; };
  EXPECT_NEAR(ex03_one_sided(cube, a, 0.25) - 3.0 * a * a,
              ex03_leading_error(6.0, 0.25), 1e-12);
  // Backward version with -h.
  EXPECT_NEAR(ex03_one_sided(quad, a, -0.25), 4.0 * a - 3.0, 1e-12);
}

TEST(Ch07Ex04, FourthDerivativeSecondOrder) {
  const double a = 0.7;
  const auto f = [](double x) { return std::sin(x); };
  const auto err = [&](double h) {
    return ex04_fourth_derivative(f, a, h) - std::sin(a);
  };
  EXPECT_NEAR(ex_observed_order(err, 0.2), 2.0, 0.05);
  EXPECT_NEAR(err(0.1) / ex04_leading_error(-std::sin(a), 0.1), 1.0, 0.01);
  // Exact for polynomials of degree 5: x^5 has f^(4) = 120 x.
  const auto p5 = [](double x) { return x * x * x * x * x - x * x; };
  EXPECT_NEAR(ex04_fourth_derivative(p5, a, 0.5), 120.0 * a, 1e-10);
  // Degree 6: error exactly h^2/6 * 720.
  const auto p6 = [](double x) { return x * x * x * x * x * x; };
  EXPECT_NEAR(ex04_fourth_derivative(p6, a, 0.5) - 360.0 * a * a,
              ex04_leading_error(720.0, 0.5), 1e-9);
}

TEST(Ch07Ex05, PointsAMinusHAndAPlus2H) {
  const double a = 0.4;
  const auto f = [](double x) { return std::exp(x); };
  // (i) second order, error h^2/3 f'''
  const auto e1 = [&](double h) {
    return ex05_first_derivative_h(f, a, h) - std::exp(a);
  };
  EXPECT_NEAR(ex_observed_order(e1, 0.1), 2.0, 0.1);
  EXPECT_NEAR(e1(0.05) / (0.05 * 0.05 / 3.0 * std::exp(a)), 1.0, 0.05);
  // (ii) first order, error h/3 f'''
  const auto e2 = [&](double h) {
    return ex05_second_derivative_h(f, a, h) - std::exp(a);
  };
  EXPECT_NEAR(ex_observed_order(e2, 0.05), 1.0, 0.05);
  EXPECT_NEAR(e2(0.01) / (0.01 / 3.0 * std::exp(a)), 1.0, 0.02);
  // The special cases agree with the general formulas (beta = a - h, gamma = a
  // + 2h).
  for (double h : {0.3, 0.1, 0.02}) {
    EXPECT_NEAR(ex05_first_derivative_h(f, a, h),
                ex05_first_derivative(f, a, a - h, a + 2 * h), 1e-9);
    EXPECT_NEAR(ex05_second_derivative_h(f, a, h),
                ex05_second_derivative(f, a, a - h, a + 2 * h), 1e-7);
  }
  // No second order f'' from these points: for f = (x - a)^3 the error is
  // exactly 2h.
  const auto cube = [a](double x) { return (x - a) * (x - a) * (x - a); };
  for (double h : {0.5, 0.1, 0.01})
    EXPECT_NEAR(ex05_second_derivative_h(cube, a, h), 2.0 * h, 1e-9);
  // Both formulas are exact for quadratics.
  const auto quad = [](double x) { return 3.0 * x * x - x + 2.0; };
  EXPECT_NEAR(ex05_first_derivative_h(quad, a, 0.25), 6.0 * a - 1.0, 1e-12);
  EXPECT_NEAR(ex05_second_derivative_h(quad, a, 0.25), 6.0, 1e-11);
}

TEST(Ch07Ex05, GeneralBetaGamma) {
  const double a = 1.0;
  const auto f = [](double x) { return std::log(1.0 + x * x); };
  // f''' at 1 for log(1 + x^2): f' = 2x/(1+x^2), f'' = 2(1-x^2)/(1+x^2)^2,
  // f''' = 4x(x^2 - 3)/(1+x^2)^3 = -1 at x = 1.
  const double f1 = 1.0, f2 = 0.0, f3 = -1.0;
  for (double C : {0.5, 2.0, 3.0}) {
    // (iii) residual O(k^2), (iv) O(k), k = gamma - a.
    const auto e1 = [&](double k) {
      return ex05_first_derivative(f, a, a - C * k, a + k) - f1;
    };
    const auto e2 = [&](double k) {
      return ex05_second_derivative(f, a, a - C * k, a + k) - f2;
    };
    EXPECT_NEAR(ex_observed_order(e1, 0.02), 2.0, 0.1) << C;
    EXPECT_NEAR(ex_observed_order(e2, 0.02), 1.0, 0.1) << C;
    const double k = 0.005;
    EXPECT_NEAR(e1(k) / ex05_first_leading_error(a, a - C * k, a + k, f3), 1.0,
                0.05)
        << C;
    EXPECT_NEAR(e2(k) / ex05_second_leading_error(a, a - C * k, a + k, f3), 1.0,
                0.05)
        << C;
  }
  // C = 1 (symmetric): both are the central differences, second order.
  const auto e2s = [&](double k) {
    return ex05_second_derivative(f, a, a - k, a + k) - f2;
  };
  EXPECT_NEAR(ex_observed_order(e2s, 0.05), 2.0, 0.1);
  EXPECT_NEAR(ex05_second_derivative(f, a, a - 0.1, a + 0.1),
              second_central_difference(f, a, 0.1), 1e-12);
  EXPECT_NEAR(ex05_first_derivative(f, a, a - 0.1, a + 0.1),
              central_difference(f, a, 0.1), 1e-12);
}
