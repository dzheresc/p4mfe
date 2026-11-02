#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch06/taylor_series.hpp>
#include <p4mfe/ch10/big_o.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <p4mfe/ch10/stirling.hpp>
#include <p4mfe/ch10/sums.hpp>
#include <p4mfe/ch10/taylor_convergence.hpp>
#include <vector>

using namespace p4mfe;
using namespace p4mfe::ch10;

// ----------------------------------------------------------------- 10.5 ---
TEST(Ch10BigO, SmallH) {
  // e^h - 1 - h = O(h^2): observed power 2, ratio to h -> 0 (o(h)).
  const auto r = [](double h) { return std::exp(h) - 1.0 - h; };
  EXPECT_NEAR(observed_power(r, 1e-3), 2.0, 1e-3);
  const std::vector<double> hs{1e-1, 1e-2, 1e-3, 1e-4};
  const auto ratios_h = order_ratios(r, [](double h) { return h; }, hs);
  for (std::size_t i = 1; i < ratios_h.size(); ++i)
    EXPECT_LT(ratios_h[i], ratios_h[i - 1] / 5);
  const auto ratios_h2 = order_ratios(r, [](double h) { return h * h; }, hs);
  for (double q : ratios_h2) EXPECT_NEAR(q, 0.5, 0.06);  // bounded, -> 1/2
  // Simpson's rule error is O(h^4): with h = 1/n, from chapter 2.
  const auto simpson_err = [](double h) {
    return ch02::Integrate([](double x) { return std::exp(x); }, 0.0, 1.0,
                           static_cast<std::size_t>(std::lround(1.0 / h))) -
           (std::numbers::e - 1.0);
  };
  EXPECT_NEAR(observed_power(simpson_err, 1.0 / 16), 4.0, 0.01);
}

TEST(Ch10BigO, LargeN) {
  // sum k^2 - n^3/3 = O(n^2): power 2, ratio to n^2 -> 1/2.
  const auto r = [](double n) {
    return static_cast<double>(sum_k2(static_cast<std::uint64_t>(n))) -
           n * n * n / 3.0;
  };
  EXPECT_NEAR(observed_power_at_infinity(r, 1000.0), 2.0, 1e-3);
  // ln n! - (n ln n - n) = O(ln n) = o(n).
  const auto s = [](double n) {
    return log_factorial(n) - (n * std::log(n) - n);
  };
  const auto q = order_ratios(
      s, [](double n) { return n; }, std::vector<double>{10, 100, 1000, 10000});
  for (std::size_t i = 1; i < q.size(); ++i) EXPECT_LT(q[i], q[i - 1]);
  EXPECT_LT(q.back(), 1e-3);
}

// ----------------------------------------------------------------- 10.6 ---
TEST(Ch10PowerSeries, MatchesChapter6PartialSums) {
  for (double x : {-0.6, 0.3, 0.9}) {
    EXPECT_NEAR(exp_power_series(12)(x), ch06::ExpSeries(x, 12), 1e-15);
    EXPECT_NEAR(geometric_power_series(30)(x), ch06::GeometricSeries(x, 30),
                1e-14);
    EXPECT_NEAR(log1p_power_series(40)(x), ch06::Log1pSeries(x, 40), 1e-14);
  }
}

TEST(Ch10PowerSeries, TermByTermDifferentiationAndIntegration) {
  // (e^x)' = e^x; int_0^x 1/(1+t) = ln(1+x); arctan = int 1/(1+t^2).
  const auto e = exp_power_series(25);
  const auto de = derivative(e);
  for (std::size_t k = 0; k < de.a.size(); ++k)
    EXPECT_NEAR(de.a[k], e.a[k], 1e-18);
  PowerSeries g = geometric_power_series(200);
  for (std::size_t k = 1; k < g.a.size(); k += 2) g.a[k] = -1.0;  // 1/(1+t)
  EXPECT_NEAR(antiderivative(g)(0.5), std::log1p(0.5), 1e-14);
  PowerSeries h{Vector(201, 0.0), 0.0};  // 1/(1+t^2)
  for (std::size_t k = 0; k <= 200; k += 2) h.a[k] = (k / 2) % 2 ? -1.0 : 1.0;
  EXPECT_NEAR(antiderivative(h)(0.6), std::atan(0.6), 1e-14);
}

TEST(Ch10PowerSeries, CauchyProductAndRadius) {
  // e^x e^x = e^(2x): coefficients 2^k / k!.
  const auto e = exp_power_series(15);
  const auto ee = cauchy_product(e, e);
  for (std::size_t k = 0; k < ee.a.size(); ++k)
    EXPECT_NEAR(ee.a[k], std::pow(2.0, k) * e.a[k], 1e-14);
  // 1/(1-x)^2 = sum (k+1) x^k.
  const auto g2 =
      cauchy_product(geometric_power_series(10), geometric_power_series(10));
  for (std::size_t k = 0; k < g2.a.size(); ++k)
    EXPECT_DOUBLE_EQ(g2.a[k], static_cast<double>(k) + 1.0);
  // Radii: ln(1+x) has R = 1; e^x has R = infinity (estimates grow with n).
  EXPECT_NEAR(radius_ratio_estimate(log1p_power_series(1000).a), 1.0, 1e-2);
  EXPECT_NEAR(radius_root_estimate(log1p_power_series(1000).a), 1.0, 1e-2);
  EXPECT_GT(radius_ratio_estimate(exp_power_series(50).a), 49.0);
  EXPECT_GT(radius_root_estimate(exp_power_series(50).a),
            radius_root_estimate(exp_power_series(10).a));
}

// ----------------------------------------------------------------- 10.7 ---
TEST(Ch10Stirling, RelativeErrorIsOneOver12N) {
  for (double n : {1.0, 5.0, 10.0, 50.0, 150.0}) {
    const double f = std::tgamma(n + 1.0);
    const auto [lo, hi] = stirling_bounds(n);
    EXPECT_LT(lo, f * (1 + 1e-14));
    EXPECT_GT(hi, f * (1 - 1e-14));
    EXPECT_NEAR(f / stirling(n) - 1.0, 1.0 / (12.0 * n), 1.0 / (200.0 * n * n));
  }
  EXPECT_NEAR(stirling(10.0), 3598695.6187, 1e-3);
}

TEST(Ch10Stirling, SeriesImprovesWithTerms) {
  const double n = 8.0, exact = log_factorial(n);
  double prev = std::abs(log_stirling(n) - exact);
  for (unsigned t = 1; t <= 5; ++t) {
    const double err = std::abs(log_stirling(n, t) - exact);
    EXPECT_LT(err, prev) << t;
    prev = err;
  }
  EXPECT_LT(prev, 1e-12);
}

TEST(Ch10Stirling, CentralBinomial) {
  EXPECT_NEAR(central_binomial(10.0), 184756.0, 1e-6);
  for (double n : {10.0, 100.0, 400.0})
    EXPECT_NEAR(central_binomial_stirling(n) / central_binomial(n) - 1.0,
                1.0 / (8.0 * n), 1.0 / (50.0 * n * n));
}

// ----------------------------------------------------------------- 10.8 ---
TEST(Ch10TaylorConvergence, GeometricDerivativeBound) {
  // e^(2x) at 0: f^(k) = 2^k e^(2x), so C = e^(2|x|) and M = 2.
  const double x = 0.8, C = std::exp(2 * x);
  const unsigned n = taylor_terms_needed(C, 2.0, 0.0, x, 1e-10);
  const auto s = exp_power_series(n);
  EXPECT_LT(std::abs(s(2 * x) - std::exp(2 * x)), 1e-10);
  EXPECT_LE(std::abs(exp_power_series(n - 1)(2 * x) - std::exp(2 * x)),
            taylor_bound_geometric(C, 2.0, 0.0, x, n - 1) * (1 + 1e-12));
  EXPECT_GE(taylor_bound_geometric(C, 2.0, 0.0, x, n - 1), 1e-10);
  // Bounds -> 0 even for large |x - a|: factorials beat powers.
  EXPECT_LT(taylor_bound_geometric(1.0, 1.0, 0.0, 10.0, 60), 1e-15);
}

TEST(Ch10TaylorConvergence, FlatFunctionIsNotItsTaylorSeries) {
  // Every derivative at 0 is 0: f(h) = o(h^k) for all k, yet f(h) > 0.
  for (int k : {1, 4, 10})
    EXPECT_LT(flat_function(0.05) / std::pow(0.05, k), 1e-100);
  const std::vector<double> zeros(10, 0.0);
  EXPECT_EQ(taylor_series(zeros, 0.0)(0.5), 0.0);
  EXPECT_GT(flat_function(0.5), 0.018);
}

TEST(Ch10TaylorConvergence, PowerSeriesIsItsOwnTaylorSeries) {
  // sin at a = 1: derivatives sin 1, cos 1, -sin 1, -cos 1, ...
  std::vector<double> d;
  for (int k = 0; k < 25; ++k) {
    const double v[4] = {std::sin(1.0), std::cos(1.0), -std::sin(1.0),
                         -std::cos(1.0)};
    d.push_back(v[k % 4]);
  }
  const auto s = taylor_series(d, 1.0);
  EXPECT_NEAR(s(2.5), std::sin(2.5), 1e-14);
  const auto back = derivatives_at_center(s);
  for (std::size_t k = 0; k < d.size(); ++k) EXPECT_NEAR(back[k], d[k], 1e-12);
}
