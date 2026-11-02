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
  EXPECT_NEAR(ExObservedOrder(Ex01Error, 0.1), 4.0, 0.1);
  EXPECT_NEAR(ExObservedOrder(Ex01Error, 0.05), 4.0, 0.05);
  EXPECT_NEAR(Ex01Error(0.05) / std::pow(0.05, 4), kEx01LeadingCoefficient,
              0.01);
  // At a = 1, where f''' != 0, the same scheme is only second order.
  const auto err1 = [](double h) {
    const double exact =
        (1.0 + 3.0 - 6.0) * std::exp(1.0);  // (x^3 + 3x^2 - 6) e^x
    return CentralDifference(Ex01F, 1.0, h) - exact;
  };
  EXPECT_NEAR(ExObservedOrder(err1, 0.05), 2.0, 0.1);
}

TEST(Ch07Ex02, ForwardDifferenceOfNIsSecondOrderAtZero) {
  EXPECT_NEAR(ExObservedOrder(Ex02Error, 0.1), 2.0, 0.05);
  EXPECT_NEAR(Ex02Error(0.05) / (0.05 * 0.05), kEx02LeadingCoefficient,
              1e-3 * 0.07);
  // At x = 1 (N''(1) != 0) the forward difference is first order.
  const auto err1 = [](double h) {
    return ForwardDifference([](double x) { return NormCdf(x); }, 1.0, h) -
           NormPdf(1.0);
  };
  EXPECT_NEAR(ExObservedOrder(err1, 0.05), 1.0, 0.1);
}

TEST(Ch07Ex03, OneSidedSecondOrder) {
  const double a = 0.3;
  const auto f = [](double x) { return std::exp(x); };
  const auto err = [&](double h) {
    return Ex03OneSided(f, a, h) - std::exp(a);
  };
  EXPECT_NEAR(ExObservedOrder(err, 0.1), 2.0, 0.1);
  EXPECT_NEAR(err(0.05) / Ex03LeadingError(std::exp(a), 0.05), 1.0, 0.05);
  // Exact for quadratics; for a cubic the error is exactly -h^2 f'''/3.
  const auto quad = [](double x) { return 2.0 * x * x - 3.0 * x + 1.0; };
  EXPECT_NEAR(Ex03OneSided(quad, a, 0.25), 4.0 * a - 3.0, 1e-12);
  const auto cube = [](double x) { return x * x * x; };
  EXPECT_NEAR(Ex03OneSided(cube, a, 0.25) - 3.0 * a * a,
              Ex03LeadingError(6.0, 0.25), 1e-12);
  // Backward version with -h.
  EXPECT_NEAR(Ex03OneSided(quad, a, -0.25), 4.0 * a - 3.0, 1e-12);
}

TEST(Ch07Ex04, FourthDerivativeSecondOrder) {
  const double a = 0.7;
  const auto f = [](double x) { return std::sin(x); };
  const auto err = [&](double h) {
    return Ex04FourthDerivative(f, a, h) - std::sin(a);
  };
  EXPECT_NEAR(ExObservedOrder(err, 0.2), 2.0, 0.05);
  EXPECT_NEAR(err(0.1) / Ex04LeadingError(-std::sin(a), 0.1), 1.0, 0.01);
  // Exact for polynomials of degree 5: x^5 has f^(4) = 120 x.
  const auto p5 = [](double x) { return x * x * x * x * x - x * x; };
  EXPECT_NEAR(Ex04FourthDerivative(p5, a, 0.5), 120.0 * a, 1e-10);
  // Degree 6: error exactly h^2/6 * 720.
  const auto p6 = [](double x) { return x * x * x * x * x * x; };
  EXPECT_NEAR(Ex04FourthDerivative(p6, a, 0.5) - 360.0 * a * a,
              Ex04LeadingError(720.0, 0.5), 1e-9);
}

TEST(Ch07Ex05, PointsAMinusHAndAPlus2H) {
  const double a = 0.4;
  const auto f = [](double x) { return std::exp(x); };
  // (i) second order, error h^2/3 f'''
  const auto e1 = [&](double h) {
    return Ex05FirstDerivativeH(f, a, h) - std::exp(a);
  };
  EXPECT_NEAR(ExObservedOrder(e1, 0.1), 2.0, 0.1);
  EXPECT_NEAR(e1(0.05) / (0.05 * 0.05 / 3.0 * std::exp(a)), 1.0, 0.05);
  // (ii) first order, error h/3 f'''
  const auto e2 = [&](double h) {
    return Ex05SecondDerivativeH(f, a, h) - std::exp(a);
  };
  EXPECT_NEAR(ExObservedOrder(e2, 0.05), 1.0, 0.05);
  EXPECT_NEAR(e2(0.01) / (0.01 / 3.0 * std::exp(a)), 1.0, 0.02);
  // The special cases agree with the general formulas (beta = a - h, gamma = a
  // + 2h).
  for (double h : {0.3, 0.1, 0.02}) {
    EXPECT_NEAR(Ex05FirstDerivativeH(f, a, h),
                Ex05FirstDerivative(f, a, a - h, a + 2 * h), 1e-9);
    EXPECT_NEAR(Ex05SecondDerivativeH(f, a, h),
                Ex05SecondDerivative(f, a, a - h, a + 2 * h), 1e-7);
  }
  // No second order f'' from these points: for f = (x - a)^3 the error is
  // exactly 2h.
  const auto cube = [a](double x) { return (x - a) * (x - a) * (x - a); };
  for (double h : {0.5, 0.1, 0.01})
    EXPECT_NEAR(Ex05SecondDerivativeH(cube, a, h), 2.0 * h, 1e-9);
  // Both formulas are exact for quadratics.
  const auto quad = [](double x) { return 3.0 * x * x - x + 2.0; };
  EXPECT_NEAR(Ex05FirstDerivativeH(quad, a, 0.25), 6.0 * a - 1.0, 1e-12);
  EXPECT_NEAR(Ex05SecondDerivativeH(quad, a, 0.25), 6.0, 1e-11);
}

TEST(Ch07Ex05, GeneralBetaGamma) {
  const double a = 1.0;
  const auto f = [](double x) { return std::log(1.0 + x * x); };
  // f''' at 1 for log(1 + x^2): f' = 2x/(1+x^2), f'' = 2(1-x^2)/(1+x^2)^2,
  // f''' = 4x(x^2 - 3)/(1+x^2)^3 = -1 at x = 1.
  const double f1 = 1.0, f2 = 0.0, f3 = -1.0;
  for (double c : {0.5, 2.0, 3.0}) {
    // (iii) residual O(k^2), (iv) O(k), k = gamma - a.
    const auto e1 = [&](double k) {
      return Ex05FirstDerivative(f, a, a - c * k, a + k) - f1;
    };
    const auto e2 = [&](double k) {
      return Ex05SecondDerivative(f, a, a - c * k, a + k) - f2;
    };
    EXPECT_NEAR(ExObservedOrder(e1, 0.02), 2.0, 0.1) << c;
    EXPECT_NEAR(ExObservedOrder(e2, 0.02), 1.0, 0.1) << c;
    const double k = 0.005;
    EXPECT_NEAR(e1(k) / Ex05FirstLeadingError(a, a - c * k, a + k, f3), 1.0,
                0.05)
        << c;
    EXPECT_NEAR(e2(k) / Ex05SecondLeadingError(a, a - c * k, a + k, f3), 1.0,
                0.05)
        << c;
  }
  // C = 1 (symmetric): both are the central differences, second order.
  const auto e2s = [&](double k) {
    return Ex05SecondDerivative(f, a, a - k, a + k) - f2;
  };
  EXPECT_NEAR(ExObservedOrder(e2s, 0.05), 2.0, 0.1);
  EXPECT_NEAR(Ex05SecondDerivative(f, a, a - 0.1, a + 0.1),
              SecondCentralDifference(f, a, 0.1), 1e-12);
  EXPECT_NEAR(Ex05FirstDerivative(f, a, a - 0.1, a + 0.1),
              CentralDifference(f, a, 0.1), 1e-12);
}
