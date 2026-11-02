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
void ExpectAntiderivative(F&& antiderivative, Fp&& f, double a, double b,
                          double tol = 1e-6) {
  EXPECT_NEAR(
      antiderivative(b) - antiderivative(a), RiemannSum(f, a, b, 20000),
      tol * std::max(1.0, std::abs(antiderivative(b) - antiderivative(a))));
  for (double x : {a, 0.5 * (a + b), b})
    EXPECT_NEAR(Derivative(antiderivative, x), f(x),
                1e-6 * std::max(1.0, std::abs(f(x))));
}
}  // namespace

TEST(Ch01Ex01, LnAntiderivative) {
  ExpectAntiderivative(
      Ex01LnAntiderivative, [](double x) { return std::log(x); }, 0.5, 4.0);
  EXPECT_DOUBLE_EQ(Ex01LnAntiderivative(std::exp(1.0)), 0.0);  // e*1 - e
}

TEST(Ch01Ex02, InvXLnX) {
  const auto f = [](double x) { return 1.0 / (x * std::log(x)); };
  ExpectAntiderivative(Ex02InvXLnXAntiderivative, f, 1.5, 10.0);
  ExpectAntiderivative(Ex02InvXLnXAntiderivative, f, 0.1,
                       0.9);  // the (0, 1) branch
}

TEST(Ch01Ex03, TanAndArctan) {
  for (double x : {-1.2, -0.3, 0.0, 0.7, 1.4}) {
    EXPECT_NEAR(Derivative([](double y) { return std::tan(y); }, x),
                Ex03TanDerivative(x), 1e-6 * Ex03TanDerivative(x));
    EXPECT_NEAR(Ex03TanDerivative(x), 1.0 + std::tan(x) * std::tan(x),
                1e-12 * Ex03TanDerivative(x));
  }
  ExpectAntiderivative(Ex03Inv1PlusX2Antiderivative, Ex03ArctanDerivative, -3.0,
                       5.0);
  EXPECT_NEAR(RiemannSum(Ex03ArctanDerivative, 0.0, 1.0, 10000),
              std::numbers::pi / 4.0, 1e-9);
}

TEST(Ch01Ex04, XPowNLn) {
  for (double n : {-3.0, -1.0, -0.5, 0.0, 1.0, 2.0, 4.5}) {
    ExpectAntiderivative(
        [n](double x) { return Ex04XPowNLnAntiderivative(n, x); },
        [n](double x) { return std::pow(x, n) * std::log(x); }, 0.5, 3.0);
  }
  // n = 0 is exercise 1.
  EXPECT_DOUBLE_EQ(Ex04XPowNLnAntiderivative(0.0, 2.0),
                   Ex01LnAntiderivative(2.0));
}

TEST(Ch01Ex05, XPowNExp) {
  for (unsigned n = 0; n <= 6; ++n) {
    ExpectAntiderivative(
        [n](double x) { return Ex05XPowNExpAntiderivative(n, x); },
        [n](double x) { return std::pow(x, n) * std::exp(x); }, -2.0, 2.0);
  }
  const double x = 1.3;
  EXPECT_NEAR(Ex05XPowNExpAntiderivative(2, x),
              (x * x - 2 * x + 2) * std::exp(x), 1e-14);
  // I_n(0) = (-1)^n n!.
  EXPECT_DOUBLE_EQ(Ex05XPowNExpAntiderivative(4, 0.0), 24.0);
  EXPECT_DOUBLE_EQ(Ex05XPowNExpAntiderivative(3, 0.0), -6.0);
}

TEST(Ch01Ex06, LnPowN) {
  for (unsigned n = 0; n <= 5; ++n) {
    ExpectAntiderivative(
        [n](double x) { return Ex06LnPowNAntiderivative(n, x); },
        [n](double x) { return std::pow(std::log(x), n); }, 0.5, 5.0);
  }
  EXPECT_NEAR(Ex06LnPowNAntiderivative(1, 2.0), Ex01LnAntiderivative(2.0),
              1e-14);
  EXPECT_THROW((void)Ex06LnPowNAntiderivative(2, -1.0), std::domain_error);
}
