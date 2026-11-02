#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/exercises/ex01_04_dice_coins.hpp>
#include <p4mfe/ch03/exercises/ex05_09_distributions.hpp>
#include <p4mfe/ch03/exercises/ex10_11_cauchy_schwarz.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <random>
#include <vector>

using namespace p4mfe::ch03;
using namespace p4mfe::ch03::ex;

TEST(Ch03Ex01, BreakevenWinsByEnumeration) {
  int counts[13] = {};
  for (int a = 1; a <= 6; ++a)
    for (int b = 1; b <= 6; ++b) ++counts[a + b];
  const double expected[13] = {0, 0, 35, 17, 11, 8, 6.2, 5, 6.2, 8, 11, 17, 35};
  for (int k = 2; k <= 12; ++k) {
    EXPECT_EQ(ex01_ways(k), counts[k]) << k;
    EXPECT_NEAR(ex01_breakeven_win(k), expected[k], 1e-12) << k;
    EXPECT_NEAR(ex01_expected_gain(k, ex01_breakeven_win(k)), 0.0, 1e-14) << k;
    EXPECT_LT(ex01_expected_gain(k, ex01_breakeven_win(k) - 0.01), 0.0);
  }
}

TEST(Ch03Ex02, GeometricMomentsBySeries) {
  for (double p : {0.5, 0.2, 0.9}) {
    double m = 0.0, m2 = 0.0;
    for (int n = 1; n < 2000; ++n) {
      const double pn = std::pow(1.0 - p, n - 1) * p;
      m += n * pn;
      m2 += double(n) * n * pn;
    }
    EXPECT_NEAR(ex02_mean(p), m, 1e-10);
    EXPECT_NEAR(ex02_variance(p), m2 - m * m, 1e-9);
  }
  EXPECT_DOUBLE_EQ(ex02_mean(), 2.0);
  EXPECT_DOUBLE_EQ(ex02_variance(), 2.0);
}

TEST(Ch03Ex03_04, HeadsInARow) {
  EXPECT_DOUBLE_EQ(ex03_expected_tosses(), 6.0);
  for (double p : {0.5, 0.3, 0.8}) {
    EXPECT_NEAR(ex03_expected_tosses(p), ex04_expected_tosses(2, p), 1e-12);
    for (int k = 1; k <= 6; ++k) {
      const double e = ex04_expected_tosses(k, p);
      EXPECT_NEAR(e, ex04_expected_tosses_recursive(k, p), 1e-9 * e);
      EXPECT_NEAR(e, ex04_expected_tosses_markov(k, p), 1e-8 * e)
          << k << " " << p;
    }
  }
  for (int k = 1; k <= 10; ++k)
    EXPECT_DOUBLE_EQ(ex04_expected_tosses(k, 0.5), std::pow(2.0, k + 1) - 2.0);
}

TEST(Ch03Ex05, ThreeStepTree) {
  const auto d = ex05_distribution();
  ASSERT_EQ(d.values.size(), 4u);
  EXPECT_NEAR(d.values[0], 35.38944, 1e-10);
  EXPECT_NEAR(d.values[3], 46.305, 1e-10);
  EXPECT_NEAR(d.probs[1], 0.288, 1e-14);
  EXPECT_NEAR(mean(d), ex05_mean(), 1e-12);
  EXPECT_NEAR(variance(d), ex05_variance(), 1e-10);
  EXPECT_NEAR(ex05_mean(), 41.70362976, 1e-10);
  EXPECT_NEAR(ex05_variance(), 9.883487, 1e-6);
}

TEST(Ch03Ex06, UniformByIntegration) {
  for (auto [a, b] :
       {std::pair{0.0, 1.0}, std::pair{-2.0, 5.0}, std::pair{3.0, 3.5}}) {
    const ContinuousDistribution u{
        [a, b](double x) { return (x < a || x > b) ? 0.0 : 1.0 / (b - a); }, a,
        b};
    EXPECT_NEAR(mean(u), ex06_mean(a, b), 1e-8);
    EXPECT_NEAR(variance(u), ex06_variance(a, b), 1e-8);
  }
}

TEST(Ch03Ex07, Exponential) {
  for (double alpha : {0.5, 1.0, 3.0}) {
    const auto e = exponential_distribution(alpha);
    EXPECT_NEAR(expected_value(e, [](double) { return 1.0; }), 1.0, 1e-8);
    EXPECT_NEAR(mean(e), 1.0 / alpha, 1e-7);
    EXPECT_NEAR(variance(e), 1.0 / (alpha * alpha), 1e-6);
    for (double x : {0.1, 1.0, 2.5}) {
      EXPECT_NEAR(cdf(e, x), ex07_cdf(alpha, x), 1e-8);
      EXPECT_NEAR(ex07_tail(alpha, x), 1.0 - ex07_cdf(alpha, x), 1e-15);
      EXPECT_NEAR(ex07_conditional_tail(alpha, 1.7, x), ex07_tail(alpha, x),
                  1e-14);
    }
    EXPECT_EQ(ex07_cdf(alpha, -1.0), 0.0);
  }
}

TEST(Ch03Ex08, NormalAbsoluteMoment) {
  for (auto [mu, sigma] :
       {std::pair{0.0, 1.0}, std::pair{1.5, 0.7}, std::pair{-2.0, 3.0}}) {
    const ContinuousDistribution n{
        [mu, sigma](double x) { return normal_pdf(x, mu, sigma); },
        -p4mfe::ch02::infinity, p4mfe::ch02::infinity};
    EXPECT_NEAR(ex08_mean_abs(mu, sigma),
                expected_value(n, [](double x) { return std::abs(x); }), 1e-7);
    EXPECT_NEAR(ex08_second_moment(mu, sigma),
                expected_value(n, [](double x) { return x * x; }), 1e-7);
  }
  EXPECT_NEAR(ex08_mean_abs(0.0, 2.0), 2.0 * std::sqrt(2.0 / std::numbers::pi),
              1e-15);
}

TEST(Ch03Ex09, PoissonBySummation) {
  for (double lambda : {0.3, 1.0, 4.5, 30.0}) {
    const auto d = ex09_distribution(lambda);
    EXPECT_NEAR(mean(d), lambda, 1e-10 * (1 + lambda));
    EXPECT_NEAR(variance(d), lambda, 1e-9 * (1 + lambda));
  }
}

TEST(Ch03Ex10, CauchySchwarzSums) {
  std::mt19937 gen(3);
  std::normal_distribution<double> z;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> x(1 + trial % 9), y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = z(gen), y[i] = z(gen);
    const auto s = ex10_sides(x, y);
    EXPECT_LE(s.lhs, s.rhs * (1 + 1e-14));
    // The quadratic in alpha is nonnegative, minimal at -B/C.
    double B = 0, C = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      B += x[i] * y[i], C += y[i] * y[i];
    EXPECT_GE(ex10_quadratic(x, y, -B / C), -1e-12);
    EXPECT_NEAR(ex10_quadratic(x, y, -B / C), (s.rhs - s.lhs) / C, 1e-9);
  }
  // Equality iff proportional.
  const std::vector<double> x{1, -2, 3}, y{-2.5, 5, -7.5};
  const auto s = ex10_sides(x, y);
  EXPECT_NEAR(s.lhs, s.rhs, 1e-12);
}

TEST(Ch03Ex11, CauchySchwarzIntegrals) {
  const auto f = [](double x) { return std::exp(x); };
  const auto g = [](double x) { return std::sin(3 * x) + x; };
  const auto s = ex11_sides(f, g, -1.0, 2.0);
  EXPECT_LE(s.lhs, s.rhs);
  EXPECT_LE(std::abs(ex11_sides(
                         f, [](double x) { return -std::cos(x); }, 0.0, 4.0)
                         .lhs),
            ex11_sides(f, [](double x) { return -std::cos(x); }, 0.0, 4.0).rhs);
  // Equality for g = 2 f.
  const auto e = ex11_sides(f, [&](double x) { return 2 * f(x); }, 0.0, 1.0);
  EXPECT_NEAR(e.lhs, e.rhs, 1e-12);
}
