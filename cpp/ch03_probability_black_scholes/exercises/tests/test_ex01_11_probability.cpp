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
    EXPECT_EQ(Ex01Ways(k), counts[k]) << k;
    EXPECT_NEAR(Ex01BreakevenWin(k), expected[k], 1e-12) << k;
    EXPECT_NEAR(Ex01ExpectedGain(k, Ex01BreakevenWin(k)), 0.0, 1e-14) << k;
    EXPECT_LT(Ex01ExpectedGain(k, Ex01BreakevenWin(k) - 0.01), 0.0);
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
    EXPECT_NEAR(Ex02Mean(p), m, 1e-10);
    EXPECT_NEAR(Ex02Variance(p), m2 - m * m, 1e-9);
  }
  EXPECT_DOUBLE_EQ(Ex02Mean(), 2.0);
  EXPECT_DOUBLE_EQ(Ex02Variance(), 2.0);
}

TEST(Ch03Ex03_04, HeadsInARow) {
  EXPECT_DOUBLE_EQ(Ex03ExpectedTosses(), 6.0);
  for (double p : {0.5, 0.3, 0.8}) {
    EXPECT_NEAR(Ex03ExpectedTosses(p), Ex04ExpectedTosses(2, p), 1e-12);
    for (int k = 1; k <= 6; ++k) {
      const double e = Ex04ExpectedTosses(k, p);
      EXPECT_NEAR(e, Ex04ExpectedTossesRecursive(k, p), 1e-9 * e);
      EXPECT_NEAR(e, Ex04ExpectedTossesMarkov(k, p), 1e-8 * e) << k << " " << p;
    }
  }
  for (int k = 1; k <= 10; ++k)
    EXPECT_DOUBLE_EQ(Ex04ExpectedTosses(k, 0.5), std::pow(2.0, k + 1) - 2.0);
}

TEST(Ch03Ex05, ThreeStepTree) {
  const auto d = Ex05Distribution();
  ASSERT_EQ(d.values.size(), 4u);
  EXPECT_NEAR(d.values[0], 35.38944, 1e-10);
  EXPECT_NEAR(d.values[3], 46.305, 1e-10);
  EXPECT_NEAR(d.probs[1], 0.288, 1e-14);
  EXPECT_NEAR(Mean(d), Ex05Mean(), 1e-12);
  EXPECT_NEAR(Variance(d), Ex05Variance(), 1e-10);
  EXPECT_NEAR(Ex05Mean(), 41.70362976, 1e-10);
  EXPECT_NEAR(Ex05Variance(), 9.883487, 1e-6);
}

TEST(Ch03Ex06, UniformByIntegration) {
  for (auto [a, b] :
       {std::pair{0.0, 1.0}, std::pair{-2.0, 5.0}, std::pair{3.0, 3.5}}) {
    const ContinuousDistribution u{
        [a, b](double x) { return (x < a || x > b) ? 0.0 : 1.0 / (b - a); }, a,
        b};
    EXPECT_NEAR(Mean(u), Ex06Mean(a, b), 1e-8);
    EXPECT_NEAR(Variance(u), Ex06Variance(a, b), 1e-8);
  }
}

TEST(Ch03Ex07, Exponential) {
  for (double alpha : {0.5, 1.0, 3.0}) {
    const auto e = ExponentialDistribution(alpha);
    EXPECT_NEAR(ExpectedValue(e, [](double) { return 1.0; }), 1.0, 1e-8);
    EXPECT_NEAR(Mean(e), 1.0 / alpha, 1e-7);
    EXPECT_NEAR(Variance(e), 1.0 / (alpha * alpha), 1e-6);
    for (double x : {0.1, 1.0, 2.5}) {
      EXPECT_NEAR(Cdf(e, x), Ex07Cdf(alpha, x), 1e-8);
      EXPECT_NEAR(Ex07Tail(alpha, x), 1.0 - Ex07Cdf(alpha, x), 1e-15);
      EXPECT_NEAR(Ex07ConditionalTail(alpha, 1.7, x), Ex07Tail(alpha, x),
                  1e-14);
    }
    EXPECT_EQ(Ex07Cdf(alpha, -1.0), 0.0);
  }
}

TEST(Ch03Ex08, NormalAbsoluteMoment) {
  for (auto [mu, sigma] :
       {std::pair{0.0, 1.0}, std::pair{1.5, 0.7}, std::pair{-2.0, 3.0}}) {
    const ContinuousDistribution n{
        [mu, sigma](double x) { return NormalPdf(x, mu, sigma); },
        -p4mfe::ch02::kInfinity, p4mfe::ch02::kInfinity};
    EXPECT_NEAR(Ex08MeanAbs(mu, sigma),
                ExpectedValue(n, [](double x) { return std::abs(x); }), 1e-7);
    EXPECT_NEAR(Ex08SecondMoment(mu, sigma),
                ExpectedValue(n, [](double x) { return x * x; }), 1e-7);
  }
  EXPECT_NEAR(Ex08MeanAbs(0.0, 2.0), 2.0 * std::sqrt(2.0 / std::numbers::pi),
              1e-15);
}

TEST(Ch03Ex09, PoissonBySummation) {
  for (double lambda : {0.3, 1.0, 4.5, 30.0}) {
    const auto d = Ex09Distribution(lambda);
    EXPECT_NEAR(Mean(d), lambda, 1e-10 * (1 + lambda));
    EXPECT_NEAR(Variance(d), lambda, 1e-9 * (1 + lambda));
  }
}

TEST(Ch03Ex10, CauchySchwarzSums) {
  std::mt19937 gen(3);
  std::normal_distribution<double> z;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> x(1 + trial % 9), y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = z(gen), y[i] = z(gen);
    const auto s = Ex10Sides(x, y);
    EXPECT_LE(s.lhs, s.rhs * (1 + 1e-14));
    // The quadratic in alpha is nonnegative, minimal at -B/C.
    double b = 0, c = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      b += x[i] * y[i], c += y[i] * y[i];
    EXPECT_GE(Ex10Quadratic(x, y, -b / c), -1e-12);
    EXPECT_NEAR(Ex10Quadratic(x, y, -b / c), (s.rhs - s.lhs) / c, 1e-9);
  }
  // Equality iff proportional.
  const std::vector<double> x{1, -2, 3}, y{-2.5, 5, -7.5};
  const auto s = Ex10Sides(x, y);
  EXPECT_NEAR(s.lhs, s.rhs, 1e-12);
}

TEST(Ch03Ex11, CauchySchwarzIntegrals) {
  const auto f = [](double x) { return std::exp(x); };
  const auto g = [](double x) { return std::sin(3 * x) + x; };
  const auto s = Ex11Sides(f, g, -1.0, 2.0);
  EXPECT_LE(s.lhs, s.rhs);
  EXPECT_LE(std::abs(Ex11Sides(
                         f, [](double x) { return -std::cos(x); }, 0.0, 4.0)
                         .lhs),
            Ex11Sides(f, [](double x) { return -std::cos(x); }, 0.0, 4.0).rhs);
  // Equality for g = 2 f.
  const auto e = Ex11Sides(f, [&](double x) { return 2 * f(x); }, 0.0, 1.0);
  EXPECT_NEAR(e.lhs, e.rhs, 1e-12);
}
