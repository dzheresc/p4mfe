#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/exercises/ex14_19_duration_convexity.hpp>

using namespace p4mfe::ch02;
using namespace p4mfe::ch02::ex;

namespace {
// -B'/B and B''/B by finite differences of the price.
void ExpectMetricsMatchPrice(const CashFlows& cf, double y) {
  const double h = 1e-4;
  const double price = BondPrice(cf, y), price_up = BondPrice(cf, y + h),
               price_down = BondPrice(cf, y - h);
  const auto m = ComputeBondMetrics(cf, y);
  EXPECT_NEAR(m.price, price, 1e-12);
  EXPECT_NEAR(m.duration, -(price_up - price_down) / (2 * h) / price, 1e-7);
  EXPECT_NEAR(m.convexity,
              (price_up - 2 * price + price_down) / (h * h) / price, 1e-5);
}
}  // namespace

TEST(Ch02Ex14, SemiannualBond) {
  const auto m = Ex14Metrics();
  EXPECT_NEAR(m.price, 92.9839, 1e-4);
  EXPECT_NEAR(m.duration, 2.352418, 1e-6);
  EXPECT_NEAR(m.convexity, 5.736739, 1e-6);
  const auto cf = BondCashFlows(2.5, 0.06, 2);
  ASSERT_EQ(cf.times.size(), 5u);
  EXPECT_DOUBLE_EQ(cf.times.front(), 0.5);
  EXPECT_DOUBLE_EQ(cf.amounts.back(), 103.0);
  ExpectMetricsMatchPrice(cf, 0.09);
}

TEST(Ch02Ex15, QuarterlyBond) {
  const auto m = Ex15Metrics();
  EXPECT_NEAR(m.price, 101.7049, 1e-4);
  EXPECT_NEAR(m.duration, 1.118911, 1e-6);
  EXPECT_NEAR(m.convexity, 1.285705, 1e-6);
  const auto cf = BondCashFlows(14.0 / 12.0, 0.08, 4);
  ASSERT_EQ(cf.times.size(), 5u);
  EXPECT_NEAR(cf.times.front(), 2.0 / 12.0, 1e-14);
  EXPECT_DOUBLE_EQ(cf.amounts.front(), 2.0);
  ExpectMetricsMatchPrice(cf, 0.07);
}

TEST(Ch02Ex16, HigherCouponHigherPriceLowerDuration) {
  for (double y : {0.02, 0.06, 0.12}) {
    double prev_price = 0.0, prev_duration = 1e9;
    for (double c : {0.0, 0.01, 0.03, 0.05, 0.08, 0.15, 0.5}) {
      const auto m = Ex16Metrics(10.0, c, 2, y);
      EXPECT_GT(m.price, prev_price) << c;
      EXPECT_LT(m.duration, prev_duration) << c;
      prev_price = m.price;
      prev_duration = m.duration;
    }
    EXPECT_DOUBLE_EQ(Ex16Metrics(10.0, 0.0, 2, y).duration, 10.0);
  }
}

TEST(Ch02Ex17, TenYearZeroUpTenBp) {
  EXPECT_NEAR(Ex17ExactRelativeChange(), -0.00995017, 1e-8);
  EXPECT_DOUBLE_EQ(kEx17DurationEstimate, -0.01);
  EXPECT_NEAR(Ex17ConvexityEstimate(), -0.00995, 1e-15);
  const auto cf = BondCashFlows(10.0, 0.0, 1);
  for (double y : {0.01, 0.05, 0.1}) {
    const auto m = ComputeBondMetrics(cf, y);
    EXPECT_NEAR(m.duration, 10.0, 1e-14);
    EXPECT_NEAR(m.convexity, 100.0, 1e-12);
    EXPECT_NEAR(BondPrice(cf, y + 0.001) / m.price - 1,
                Ex17ExactRelativeChange(), 1e-14);
  }
}

TEST(Ch02Ex18, DurationApproximation) {
  EXPECT_NEAR(Ex18NewPrice(), 103.785, 1e-12);
  // Check on an actual five year semiannual bond priced at 102:
  // the true price is a bit above the estimate (convexity).
  const auto cf = BondCashFlows(5.0, 0.0905, 2);
  const double y = BondYield(cf, 102.0);
  const auto m = ComputeBondMetrics(cf, y);
  const double true_price = BondPrice(cf, y - 0.005);
  const double estimate = Ex18NewPrice(m.price, m.duration, -0.005);
  EXPECT_GT(true_price, estimate);
  EXPECT_NEAR(
      true_price,
      PriceApproxDurationConvexity(m.price, m.duration, m.convexity, -0.005),
      1e-3);
}

TEST(Ch02Ex19, ConvexityIsDurationSquaredMinusSlope) {
  const auto cfs = {BondCashFlows(2.5, 0.06, 2),
                    BondCashFlows(14.0 / 12.0, 0.08, 4),
                    BondCashFlows(30.0, 0.04, 2), BondCashFlows(7.0, 0.0, 1)};
  for (const auto& cf : cfs)
    for (double y : {0.01, 0.05, 0.09}) {
      const auto m = ComputeBondMetrics(cf, y);
      const double slope = Ex19DurationSlope(cf, y);
      EXPECT_NEAR(m.convexity, m.duration * m.duration - slope,
                  1e-6 * m.convexity);
      EXPECT_LE(slope, 1e-9);
    }
}
