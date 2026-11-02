#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/exercises/ex01_02_improper_integrals.hpp>
#include <p4mfe/ch02/exercises/ex03_04_quadrature.hpp>
#include <p4mfe/ch02/exercises/ex05_06_integral_derivatives.hpp>
#include <p4mfe/ch02/exercises/ex07_simpson_identity.hpp>

using namespace p4mfe::ch02;
using namespace p4mfe::ch02::ex;

TEST(Ch02Ex01, CubicTermOfNormalCdf) {
  EXPECT_NEAR(kEx01Limit, -0.0664903800669, 1e-12);
  // The quotient is limit + x^2 / (40 sqrt(2 pi)) + O(x^4).
  for (double x : {0.5, 0.1, 1e-2, 1e-4, 1e-8, -1e-3})
    EXPECT_NEAR(Ex01Quotient(x), kEx01Limit, x * x / 40.0 + 1e-16) << x;
  EXPECT_NEAR(Ex01Quotient(1e-8), kEx01Limit, 1e-15);
  // The series branch agrees with norm_cdf where cancellation is mild.
  const double x = 0.9;
  const double naive =
      (p4mfe::NormCdf(x) - 0.5 - x / std::sqrt(2 * std::numbers::pi)) /
      (x * x * x);
  EXPECT_NEAR(Ex01Quotient(x), naive, 1e-13);
  EXPECT_NEAR(Ex01Quotient(1.0 - 1e-12), Ex01Quotient(1.0), 1e-10);
}

TEST(Ch02Ex02, GammaWellDefined) {
  for (double a : {0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0}) {
    EXPECT_LT(Ex02LeftPiece(a), Ex02LeftBound(a)) << a;
    EXPECT_NEAR(Ex02Gamma(a), std::tgamma(a), 1e-8 * std::tgamma(a)) << a;
  }
}

TEST(Ch02Ex02, RecursionAndFactorial) {
  EXPECT_NEAR(Ex02Gamma(1.0), 1.0, 1e-9);
  for (double a : {1.25, 2.5, 3.0, 4.75})
    EXPECT_NEAR(Ex02Gamma(a), (a - 1) * Ex02Gamma(a - 1), 1e-7) << a;
  double factorial = 1.0;
  for (int n = 1; n <= 8; ++n) {
    EXPECT_NEAR(Ex02Gamma(n), factorial, 1e-9 * factorial) << n;
    factorial *= n;
  }
}

TEST(Ch02Ex03, ThreeRulesToTolerance) {
  const double exact = Ex03Exact();
  EXPECT_NEAR(exact, 0.4083702471769958, 1e-14);
  const auto m = Ex03Approx(Rule::kMidpoint);
  const auto t = Ex03Approx(Rule::kTrapezoidal);
  const auto s = Ex03Approx(Rule::kSimpson);
  EXPECT_EQ(m.intervals, 256u);
  EXPECT_EQ(t.intervals, 512u);
  EXPECT_EQ(s.intervals, 16u);
  EXPECT_NEAR(m.value, 0.408369962, 1e-9);
  EXPECT_NEAR(t.value, 0.408370390, 1e-9);
  EXPECT_NEAR(s.value, 0.408370194, 1e-9);
  EXPECT_LT(m.value, exact);
  EXPECT_GT(t.value, exact);
  for (const auto& r : {m, t, s}) EXPECT_LT(std::abs(r.value - exact), 1e-6);
}

TEST(Ch02Ex04, MidpointToTolerance) {
  const double integral = Ex04Exact();
  EXPECT_NEAR(integral, 0.1791711722673056, 1e-15);
  // Exact value against an independent computation: x = u^2 makes the integrand
  // smooth.
  EXPECT_NEAR(
      integral,
      IntegrateToTolerance(
          [](double u) { return 2 * std::pow(u, 6) / (1 + std::pow(u, 4)); },
          0.0, 1.0, 1e-14)
          .value,
      1e-13);
  const auto m = Ex04Midpoint();
  EXPECT_EQ(m.intervals, 512u);
  EXPECT_NEAR(m.value, integral, 1e-6);
}

TEST(Ch02Ex04, FourthDerivativeUnbounded) {
  // Closed form against finite differences of f at an interior point.
  const double x = 0.4, h = 1e-2;
  const double fd = (Ex04F(x + 2 * h) - 4 * Ex04F(x + h) + 6 * Ex04F(x) -
                     4 * Ex04F(x - h) + Ex04F(x - 2 * h)) /
                    std::pow(h, 4);
  EXPECT_NEAR(Ex04F4(x), fd, 1e-3 * std::abs(fd) + 1e-2);
  // f''''(x) ~ -(15/16) x^(-3/2) as x -> 0+.
  for (double z : {1e-2, 1e-4, 1e-6, 1e-8})
    EXPECT_NEAR(Ex04F4(z) / (-0.9375 * std::pow(z, -1.5)), 1.0, 20 * z) << z;
  EXPECT_LT(Ex04F4(1e-8), -1e11);
}

TEST(Ch02Ex04, SimpsonOrderThreeAndAHalf) {
  const auto rows = Ex04SimpsonTable();
  ASSERT_EQ(rows.size(), 7u);
  EXPECT_EQ(rows.front().n, 4u);
  EXPECT_EQ(rows.back().n, 256u);
  EXPECT_NEAR(rows.front().error, 1.607e-5, 1e-8);
  EXPECT_NEAR(rows.back().error, 6.91e-12, 1e-13);
  for (std::size_t i = 2; i < rows.size(); ++i) {
    const double order = std::log2(rows[i - 1].error / rows[i].error);
    EXPECT_NEAR(order, kEx04SimpsonOrder, 0.05) << rows[i].n;
  }
  // Compare: a smooth integrand on the same interval converges at order 4.
  const auto smooth = [](double x) { return x * x * x * x * std::exp(x); };
  EXPECT_NEAR(ObservedOrder(smooth, 0.0, 1.0, std::numbers::e * 9.0 - 24.0, 64,
                            Rule::kSimpson),
              4.0, 0.01);
}

TEST(Ch02Ex05, CallGamma) {
  const Ex05Params p{100.0, 0.5, 0.3, 0.04};
  for (double x : {60.0, 90.0, 100.0, 115.0, 160.0}) {
    const double h = 1e-4 * x;
    const double fd = (Ex05G(p, x + h) - Ex05G(p, x - h)) / (2 * h);
    EXPECT_NEAR(Ex05GPrime(p, x), fd, 1e-9) << x;
  }
  // g(x) is also the improper integral itself.
  const double x = 105.0;
  const double gi = ImproperIntegral([](double y) { return p4mfe::NormPdf(y); },
                                     -kInfinity, Ex05B(p, x), 1e-11)
                        .value;
  EXPECT_NEAR(gi, Ex05G(p, x), 1e-9);
}

TEST(Ch02Ex06, SecondDerivativeIsH) {
  // Standard normal h: closed form, numerical integral, and finite differences.
  for (double t : {-2.0, -0.5, 0.0, 0.7, 2.5}) {
    EXPECT_NEAR(Ex06G([](double x) { return p4mfe::NormPdf(x); }, t),
                Ex06GNormal(t), 1e-9)
        << t;
    const double h = 1e-3;
    const double g1 = (Ex06GNormal(t + h) - Ex06GNormal(t - h)) / (2 * h);
    const double g2 =
        (Ex06GNormal(t + h) - 2 * Ex06GNormal(t) + Ex06GNormal(t - h)) /
        (h * h);
    EXPECT_NEAR(g1, Ex06GNormalPrime(t), 1e-7) << t;
    EXPECT_NEAR(g2, p4mfe::NormPdf(t), 1e-6) << t;
  }
  // A non-symmetric h with only polynomial decay: h(x) = 1/(1+x^2)^2.
  const auto hh = [](double x) { return 1.0 / ((1 + x * x) * (1 + x * x)); };
  const double t = 0.3, h = 1e-2;
  const double g2 = (Ex06G(hh, t + h, 1e-12) - 2 * Ex06G(hh, t, 1e-12) +
                     Ex06G(hh, t - h, 1e-12)) /
                    (h * h);
  EXPECT_NEAR(g2, hh(t), 1e-4);
}

TEST(Ch02Ex06, CallConvexityInStrikeIsDensity) {
  const Ex06Market m{100.0, 1.0, 0.25, 0.03};
  for (double strike : {70.0, 95.0, 100.0, 120.0, 150.0}) {
    const double h = 1e-2 * strike;
    const double c2 = (Ex06Call(m, strike + h) - 2 * Ex06Call(m, strike) +
                       Ex06Call(m, strike - h)) /
                      (h * h);
    EXPECT_NEAR(c2, Ex06DiscountedDensity(m, strike), 1e-5) << strike;
  }
  // And C(K) is the integral of exercise 6 with h = e^(-rT) f.
  const double strike = 110.0;
  const double call =
      Ex06G([&](double x) { return x > 0 ? Ex06DiscountedDensity(m, x) : 0.0; },
            strike, 1e-11);
  EXPECT_NEAR(call, Ex06Call(m, strike), 1e-7);
}

TEST(Ch02Ex07, SimpsonIsTwoThirdsMidpointPlusOneThirdTrapezoid) {
  const auto fs = {+[](double x) { return std::exp(-x * x); },
                   +[](double x) { return std::sqrt(x) * std::exp(-x); },
                   +[](double x) { return std::sin(5 * x) / (1 + x); }};
  for (auto f : fs)
    for (std::size_t n : {1u, 2u, 3u, 7u, 64u})
      EXPECT_NEAR(Ex07SimpsonFromMt(f, 0.5, 2.5, n),
                  SimpsonRule(f, 0.5, 2.5, n), 1e-14)
          << n;
}
