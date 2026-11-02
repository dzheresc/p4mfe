#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch06/taylor.hpp>
#include <p4mfe/ch06/taylor_series.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>
#include <vector>

using namespace p4mfe;
using namespace p4mfe::ch06;

// ------------------------------------------------------------------ 6.1 ---
TEST(Ch06Taylor, PolynomialOfExpAndRemainderBound) {
  // All derivatives of e^x at 0 are 1.
  for (unsigned n : {1u, 2u, 4u, 8u}) {
    const std::vector<double> d(n + 1, 1.0);
    for (double x : {-0.7, 0.3, 1.0}) {
      const double err = std::abs(std::exp(x) - TaylorPolynomial(d, 0.0, x));
      // |f^(n+1)| <= e^max(0, x) between 0 and x.
      EXPECT_LE(err,
                TaylorRemainderBound(std::exp(std::max(0.0, x)), 0.0, x, n) *
                    (1 + 1e-12));
    }
  }
}

TEST(Ch06Taylor, PolynomialAroundNonzeroPoint) {
  // ln x around a = 1: derivatives 0, 1, -1, 2, -6.
  const std::vector<double> d{0.0, 1.0, -1.0, 2.0, -6.0};
  EXPECT_NEAR(TaylorPolynomial(d, 1.0, 1.1),
              0.1 - 0.01 / 2 + 0.001 / 3 - 0.0001 / 4, 1e-15);
  EXPECT_THROW((void)TaylorPolynomial({}, 0.0, 1.0), std::invalid_argument);
}

TEST(Ch06Taylor, IntegralRemainderIsExact) {
  // f = sin, n = 3: f^(4) = sin.
  const auto sinf = [](double t) { return std::sin(t); };
  for (double a : {0.0, 0.5}) {
    for (double x : {-0.8, 0.2, 1.5}) {
      const std::vector<double> d{std::sin(a), std::cos(a), -std::sin(a),
                                  -std::cos(a)};
      const double pct_return = TaylorRemainderIntegral(sinf, a, x, 3);
      EXPECT_NEAR(TaylorPolynomial(d, a, x) + pct_return, std::sin(x), 1e-11)
          << a << ' ' << x;
    }
  }
}

TEST(Ch06Taylor, LinearAndQuadraticErrorOrders) {
  const auto f = [](double x) { return std::exp(x) * std::cos(x); };
  const double a = 0.4;
  // Error of the linear approximation is O(h^2), the quadratic O(h^3):
  // halving h divides them by ~4 and ~8.
  const auto e1 = [&](double h) {
    return std::abs(f(a + h) - TaylorLinear(f, a, a + h));
  };
  const auto e2 = [&](double h) {
    return std::abs(f(a + h) - TaylorQuadratic(f, a, a + h));
  };
  EXPECT_NEAR(std::log2(e1(0.02) / e1(0.01)), 2.0, 0.1);
  EXPECT_NEAR(std::log2(e2(0.02) / e2(0.01)), 3.0, 0.1);
}

// ------------------------------------------------------------------ 6.2 ---
TEST(Ch06Taylor, MultivariableQuadratic) {
  const auto f = [](const Vector& v) {
    return std::exp(v[0] - v[1]) + v[0] * v[1] * v[2] * v[2];
  };
  const Vector a{0.2, -0.1, 0.5};
  const Vector h{1e-2, -2e-2, 1.5e-2};
  Vector x = a, x2 = a;
  for (std::size_t i = 0; i < 3; ++i) {
    x[i] += h[i];
    x2[i] += 0.5 * h[i];
  }
  const double e_lin = std::abs(f(x) - TaylorLinearNd(f, a, x));
  const double e_quad = std::abs(f(x) - TaylorQuadraticNd(f, a, x));
  const double e_quad2 = std::abs(f(x2) - TaylorQuadraticNd(f, a, x2));
  EXPECT_LT(e_quad, e_lin / 20);
  EXPECT_NEAR(std::log2(e_quad / e_quad2), 3.0, 0.3);
  // A quadratic function equals its second-order expansion.
  const auto q = [](const Vector& v) {
    return 1 + 2 * v[0] - v[1] + 3 * v[0] * v[1] + v[1] * v[1];
  };
  EXPECT_NEAR(TaylorQuadraticNd(q, {0.0, 0.0}, {0.7, -1.3}), q({0.7, -1.3}),
              1e-6);
}

// ---------------------------------------------------------------- 6.2.1 ---
TEST(Ch06Taylor, TwoVariables) {
  // f(x, y) = e^x sin y at (0, pi/4): closed-form partials.
  const auto f = [](double x, double y) { return std::exp(x) * std::sin(y); };
  const double s = std::sin(std::numbers::pi / 4);
  const auto d = Partials2d(f, 0.0, std::numbers::pi / 4);
  EXPECT_NEAR(d.f, s, 1e-15);
  EXPECT_NEAR(d.fx, s, 1e-9);
  EXPECT_NEAR(d.fy, s, 1e-9);
  EXPECT_NEAR(d.fxx, s, 1e-6);
  EXPECT_NEAR(d.fxy, s, 1e-6);
  EXPECT_NEAR(d.fyy, -s, 1e-6);
  const double dx = 0.05, dy = -0.03;
  const double exact = f(dx, std::numbers::pi / 4 + dy);
  EXPECT_LT(std::abs(exact - TaylorQuadratic2d(d, dx, dy)), 1e-4);
  EXPECT_LT(std::abs(exact - TaylorQuadratic2d(d, dx, dy)),
            std::abs(exact - TaylorLinear2d(d, dx, dy)) / 10);
}

// ------------------------------------------------------------- 6.3, 6.3.1 ---
TEST(Ch06TaylorSeries, ClassicalExpansionsConverge) {
  for (double x : {-0.9, -0.3, 0.2, 0.75}) {
    EXPECT_NEAR(ExpSeries(x, 25), std::exp(x), 1e-15);
    EXPECT_NEAR(Log1pSeries(x, 2000), std::log1p(x),
                2e-12 + std::pow(std::abs(x), 2000));
    EXPECT_NEAR(GeometricSeries(x, 500), 1 / (1 - x), 1e-12);
    EXPECT_NEAR(BinomialSeries(x, 0.5, 500), std::sqrt(1 + x), 1e-12);
    EXPECT_NEAR(BinomialSeries(x, -2.0, 800), 1 / ((1 + x) * (1 + x)), 1e-9);
    EXPECT_NEAR(SinSeries(x, 12), std::sin(x), 1e-15);
    EXPECT_NEAR(CosSeries(x, 12), std::cos(x), 1e-15);
  }
  // e^x, sin, cos converge for every x.
  EXPECT_NEAR(ExpSeries(10.0, 60) / std::exp(10.0), 1.0, 1e-14);
  EXPECT_NEAR(SinSeries(6.0, 30), std::sin(6.0), 1e-12);
  // Integer exponents: the binomial series terminates.
  EXPECT_DOUBLE_EQ(BinomialSeries(3.0, 3.0, 3), 64.0);
  EXPECT_DOUBLE_EQ(BinomialSeries(3.0, 3.0, 50), 64.0);
  // ln 2 = 1 - 1/2 + 1/3 - ...: slow, error < 1/(n+1).
  EXPECT_NEAR(Log1pSeries(1.0, 1000), std::log(2.0), 1.0 / 1001);
  EXPECT_THROW((void)Log1pSeries(1.5, 10), std::domain_error);
}

TEST(Ch06TaylorSeries, NormalCdfSeries) {
  for (double x : {-2.0, -0.5, 0.0, 0.1, 1.0, 3.0})
    EXPECT_NEAR(NormCdfSeries(x, 60), norm_cdf(x), 1e-14) << x;
  // Truncation error shrinks with n.
  EXPECT_GT(std::abs(NormCdfSeries(1.5, 3) - norm_cdf(1.5)),
            std::abs(NormCdfSeries(1.5, 6) - norm_cdf(1.5)));
}
