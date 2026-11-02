#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch10/big_o.hpp>
#include <p4mfe/ch10/exercises/ex09_10_big_o.hpp>
#include <p4mfe/ch10/exercises/ex11_binomial_identities.hpp>
#include <p4mfe/ch10/exercises/ex12_13_harmonic_series.hpp>
#include <p4mfe/ch10/exercises/ex14_power_series_radius.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <vector>

using namespace p4mfe::ch10;
using namespace p4mfe::ch10::ex;

TEST(Ch10Ex09, SumsOfBigOAndLittleO) {
  std::vector<double> xs;
  for (double x = 0.1; x > 1e-6; x *= 0.37) xs.push_back(x);
  const auto x2 = [](double x) { return x * x; };
  const auto big_sum = [](double x) { return Ex09FBig(x) + Ex09GBig(x); };
  for (double r : OrderRatios(Ex09FBig, x2, xs)) EXPECT_LE(r, 1.0 + 1e-12);
  for (double r : OrderRatios(Ex09GBig, x2, xs)) EXPECT_NEAR(r, 3.0, 1e-12);
  for (double r : OrderRatios(big_sum, x2, xs))
    EXPECT_LE(r, 4.0 + 1e-12);  // C_f + C_g
  // Not o(x^2): the ratio of the sum stays >= 2.
  for (double r : OrderRatios(big_sum, x2, xs)) EXPECT_GE(r, 2.0 - 1e-12);
  const auto small_sum = [](double x) { return Ex09FSmall(x) + Ex09GSmall(x); };
  const auto ratios = OrderRatios(small_sum, x2, xs);
  for (std::size_t i = 1; i < ratios.size(); ++i)
    EXPECT_LT(ratios[i], ratios[i - 1]);
  EXPECT_LT(ratios.back(), 2e-3);
  // f + g = o(x^2) at the slower of the two rates, x^(1/2).
  EXPECT_NEAR(ObservedPower(small_sum, 1e-6), 2.5, 0.01);
}

TEST(Ch10Ex10, OrdersOfSumsOfSquaresAndCubes) {
  for (std::uint64_t n = 1; n <= 10000; n = n < 20 ? n + 1 : n * 3 / 2) {
    const double nd = static_cast<double>(n);
    EXPECT_NEAR(Ex10RatioK2(n),
                1.0 / 3.0 + 1.0 / (2.0 * nd) + 1.0 / (6.0 * nd * nd), 1e-13)
        << n;
    EXPECT_NEAR(Ex10RemainderRatioK2(n), 0.5 + 1.0 / (6.0 * nd), 1e-13) << n;
    EXPECT_NEAR(Ex10RatioK3(n), 0.25 + 1.0 / (2.0 * nd) + 1.0 / (4.0 * nd * nd),
                1e-13)
        << n;
    EXPECT_NEAR(Ex10RemainderRatioK3(n), 0.5 + 1.0 / (4.0 * nd), 1e-13) << n;
    EXPECT_LE(Ex10RatioK2(n), 1.0);
    EXPECT_LE(Ex10RemainderRatioK2(n), 2.0 / 3.0);
    EXPECT_LE(Ex10RatioK3(n), 1.0);
    EXPECT_LE(Ex10RemainderRatioK3(n), 0.75);
  }
  // Observed growth exponents 3, 2, 4, 3 (n -> 2n at n = 1000).
  const auto s2 = [](double n) {
    return static_cast<double>(SumK2(static_cast<std::uint64_t>(n)));
  };
  const auto s3 = [](double n) {
    return static_cast<double>(SumK3(static_cast<std::uint64_t>(n)));
  };
  const auto r2 = [&](double n) { return s2(n) - n * n * n / 3.0; };
  const auto r3 = [&](double n) { return s3(n) - n * n * n * n / 4.0; };
  EXPECT_NEAR(ObservedPowerAtInfinity(s2, 1000.0), 3.0, 1e-2);
  EXPECT_NEAR(ObservedPowerAtInfinity(r2, 1000.0), 2.0, 1e-2);
  EXPECT_NEAR(ObservedPowerAtInfinity(s3, 1000.0), 4.0, 1e-2);
  EXPECT_NEAR(ObservedPowerAtInfinity(r3, 1000.0), 3.0, 1e-2);
}

TEST(Ch10Ex11, BinomialIdentities) {
  for (unsigned n = 1; n <= 60; ++n) {
    EXPECT_EQ(Ex11SumBinomials(n), std::uint64_t{1} << n) << n;
    EXPECT_EQ(Ex11SumEvenBinomials(n), std::uint64_t{1} << (n - 1)) << n;
    if (n <= 58) {
      EXPECT_EQ(Ex11SumJBinomials(n), n * (std::uint64_t{1} << (n - 1))) << n;
    }
  }
  EXPECT_EQ(Ex11SumEvenBinomials(0),
            1u);  // n = 0: 1, not 2^(-1) (the identity needs n >= 1)
}

TEST(Ch10Ex12, InverseSquaresConverge) {
  const double pi2_6 = std::numbers::pi * std::numbers::pi / 6.0;
  for (unsigned n : {1u, 10u, 100u, 1000u, 100000u}) {
    const double s = Ex12SumInvSquares(n);
    EXPECT_LT(s, 2.0 - 1.0 / n + 1e-15) << n;
    EXPECT_GT(pi2_6 - s, 1.0 / (n + 1.0)) << n;
    EXPECT_LT(pi2_6 - s, 1.0 / n) << n;
  }
}

TEST(Ch10Ex12, HarmonicAndKLogKDiverge) {
  EXPECT_EQ(Ex12Harmonic(1), 1.0);  // equality at n = 1
  for (unsigned n : {2u, 3u, 10u, 1000u, 1000000u}) {
    const auto [lo, hi] = Ex12HarmonicBounds(n);
    const double h = Ex12Harmonic(n);
    EXPECT_GT(h, lo) << n;
    EXPECT_LT(h, hi) << n;
  }
  for (unsigned n : {3u, 10u, 1000u, 1000000u}) {
    const auto [lo, hi] = Ex12KLogKBounds(n);
    const double s = Ex12SumInvKLogK(n);
    EXPECT_GT(s, lo) << n;
    EXPECT_LT(s, hi) << n;
  }
  // The lower bound grows without limit: past any level for n large enough.
  EXPECT_GT(Ex12Harmonic(1000000), 14.0);
  EXPECT_GT(Ex12SumInvKLogK(1000000), 2.0);
}

TEST(Ch10Ex13, HarmonicMinusLogTendsToGamma) {
  double prev_x = Ex13X(1), prev_y = Ex13Y(1);
  EXPECT_EQ(prev_x, 1.0);
  for (unsigned n = 2; n <= 2000; ++n) {
    const double x = Ex13X(n), y = Ex13Y(n);
    EXPECT_LT(x, prev_x) << n;  // decreasing
    EXPECT_GT(y, prev_y) << n;  // increasing
    EXPECT_GT(x, kEx13EulerGamma);
    EXPECT_LT(y, kEx13EulerGamma);
    prev_x = x;
    prev_y = y;
  }
  EXPECT_NEAR(Ex13Y(1), 1.0 - std::numbers::ln2, 1e-15);
  EXPECT_NEAR(Ex13X(2), 1.5 - std::numbers::ln2, 1e-15);
  for (unsigned n : {100u, 1000u, 10000u}) {
    const double nd = n;
    EXPECT_NEAR(Ex13X(n) - kEx13EulerGamma,
                1.0 / (2.0 * nd) - 1.0 / (12.0 * nd * nd), 1e-3 / (nd * nd))
        << n;
  }
}

TEST(Ch10Ex14, RadiusIsOne) {
  for (unsigned k : {10u, 1000u, 1000000u}) EXPECT_GT(Ex14Ratio(k), 1.0);
  EXPECT_NEAR(Ex14Ratio(1000000), 1.0, 1e-5);
  const Vector a = Ex14Coefficients(100000);
  EXPECT_NEAR(RadiusRatioEstimate(a), 1.0, 1e-4);
  EXPECT_NEAR(RadiusRootEstimate(a), 1.0, 2e-4);  // 1 + (ln k + ln ln k)/k
  // |x| < 1 converges: the tail beyond n is below |x|^(n+1) / ((1-|x|)(n+1)
  // ln(n+1)).
  const double s = Ex14PartialSum(0.9, 400);
  EXPECT_NEAR(Ex14PartialSum(0.9, 200), s,
              std::pow(0.9, 201) / (0.1 * 201 * std::log(201.0)));
  // x = 1: partial sums grow like ln ln n (exercise 12).
  EXPECT_NEAR(Ex14PartialSum(1.0, 100000), Ex12SumInvKLogK(100000), 1e-12);
  // x = -1: alternating, consecutive partial sums bracket the limit.
  const double s_even = Ex14PartialSum(-1.0, 100000),
               s_odd = Ex14PartialSum(-1.0, 100001);
  EXPECT_GT(s_even, s_odd);
  EXPECT_NEAR(s_even - s_odd, 1.0 / (100001.0 * std::log(100001.0)), 1e-12);
  for (unsigned n : {1000u, 10000u}) {
    EXPECT_GE(Ex14PartialSum(-1.0, 2 * n), s_even);
    EXPECT_LE(Ex14PartialSum(-1.0, 2 * n + 1), s_odd);
  }
}
