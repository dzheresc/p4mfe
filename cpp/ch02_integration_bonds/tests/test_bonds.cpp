#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/bonds.hpp>

using namespace p4mfe::ch02;

namespace {
double Fd1(auto&& f, double x, double h = 1e-5) {
  return (f(x + h) - f(x - h)) / (2 * h);
}
double Fd2(auto&& f, double x, double h = 1e-4) {
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}
}  // namespace

TEST(Ch02Bonds, CashFlows) {
  // 14 months to maturity, semiannual 5% coupon on 100: coupons of 2.5
  // at 2, 8 and 14 months.
  const auto cf = BondCashFlows(14.0 / 12, 0.05, 2);
  ASSERT_EQ(cf.times.size(), 3u);
  EXPECT_NEAR(cf.times[0], 2.0 / 12, 1e-14);
  EXPECT_NEAR(cf.times[1], 8.0 / 12, 1e-14);
  EXPECT_NEAR(cf.times[2], 14.0 / 12, 1e-14);
  EXPECT_DOUBLE_EQ(cf.amounts[0], 2.5);
  EXPECT_DOUBLE_EQ(cf.amounts[2], 102.5);
  EXPECT_EQ(BondCashFlows(3, 0.06, 2).times.size(), 6u);
  EXPECT_EQ(BondCashFlows(3, 0.0, 2).times.size(), 1u);
}

TEST(Ch02Bonds, PriceFromCurves) {
  const auto cf = BondCashFlows(3, 0.06, 2);
  // A flat zero curve at y prices the bond as yield y.
  EXPECT_NEAR(BondPriceFromZeroRate(cf, [](double) { return 0.05; }),
              BondPrice(cf, 0.05), 1e-12);
  // Zero rate curve vs instantaneous rate curve (2.10).
  const auto r_inst = [](double t) { return 0.0525 + std::log1p(2 * t) / 200; };
  const auto r_zero = [](double t) {
    return 0.0525 + ((1 + 2 * t) * std::log1p(2 * t) - 2 * t) / (400 * t);
  };
  EXPECT_NEAR(BondPriceFromInstRate(cf, r_inst),
              BondPriceFromZeroRate(cf, r_zero), 1e-10);
}

TEST(Ch02Bonds, PriceFromInstRateRejectsUnsortedTimes) {
  const auto r = [](double) { return 0.05; };
  const CashFlows unsorted{{1.0, 0.5}, {3.0, 103.0}};
  EXPECT_THROW((void)BondPriceFromInstRate(unsorted, r), std::invalid_argument);
  const CashFlows negative{{-0.5, 1.0}, {3.0, 103.0}};
  EXPECT_THROW((void)BondPriceFromInstRate(negative, r), std::invalid_argument);
}

TEST(Ch02Bonds, DurationAndConvexityAreYieldDerivatives) {
  const auto cf = BondCashFlows(5, 0.04, 4);
  const double y = 0.045;
  const auto price = [&](double s) { return BondPrice(cf, s); };
  const auto m = ComputeBondMetrics(cf, y);
  EXPECT_DOUBLE_EQ(m.price, price(y));
  EXPECT_NEAR(m.duration, -Fd1(price, y) / m.price, 1e-8);
  EXPECT_NEAR(m.convexity, Fd2(price, y) / m.price, 1e-4);
  EXPECT_DOUBLE_EQ(BondDuration(cf, y), m.duration);
  EXPECT_DOUBLE_EQ(BondConvexity(cf, y), m.convexity);
  // Coupons shorten duration below maturity.
  EXPECT_LT(m.duration, 5.0);
  // DV01 is the linear part of the 1bp move; the convexity term is ~1e-5.
  EXPECT_NEAR(Dv01(cf, y), price(y - 1e-4) - price(y), 2e-5);
  EXPECT_NEAR(Dv01(cf, y), (price(y - 1e-4) - price(y + 1e-4)) / 2, 1e-8);
}

TEST(Ch02Bonds, PriceApproximations) {
  const auto cf = BondCashFlows(10, 0.05, 2);
  const double y = 0.05, dy = 0.01;
  const auto m = ComputeBondMetrics(cf, y);
  const double exact = BondPrice(cf, y + dy);
  const double e1 =
      std::abs(PriceApproxDuration(m.price, m.duration, dy) - exact);
  const double e2 = std::abs(
      PriceApproxDurationConvexity(m.price, m.duration, m.convexity, dy) -
      exact);
  EXPECT_LT(e2, e1 / 10);
  // Convexity: the duration line underestimates the price either way.
  EXPECT_LT(PriceApproxDuration(m.price, m.duration, dy), exact);
  EXPECT_LT(PriceApproxDuration(m.price, m.duration, -dy),
            BondPrice(cf, y - dy));
}

TEST(Ch02Bonds, DiscreteYield) {
  // On a coupon date, a bond whose yield (compounded at the coupon
  // frequency) equals its coupon rate trades at par.
  const auto cf = BondCashFlows(3, 0.06, 2);
  EXPECT_NEAR(BondPriceDiscrete(cf, 0.06, 2), 100.0, 1e-12);
  // Same price as the equivalent continuously compounded yield.
  EXPECT_NEAR(BondPrice(cf, ContinuousFromDiscrete(0.06, 2)), 100.0, 1e-12);
  EXPECT_NEAR(MacaulayDuration(cf, 0.06, 2),
              BondDuration(cf, ContinuousFromDiscrete(0.06, 2)), 1e-12);
  const auto price = [&](double s) { return BondPriceDiscrete(cf, s, 2); };
  EXPECT_NEAR(ModifiedDuration(cf, 0.06, 2), -Fd1(price, 0.06) / 100.0, 1e-8);
}

TEST(Ch02ZeroCoupon, PriceYieldDuration) {
  const double maturity = 4, r = 0.035;
  const double price = ZeroCouponBondPrice(maturity, r);
  EXPECT_NEAR(price, 100 * std::exp(-0.14), 1e-12);
  EXPECT_NEAR(ZeroRateFromZeroCouponPrice(price, maturity), r, 1e-15);
  const auto cf = BondCashFlows(maturity, 0.0, 2);
  EXPECT_NEAR(BondYield(cf, price), r, 1e-12);
  EXPECT_NEAR(BondDuration(cf, r), maturity, 1e-14);
  EXPECT_NEAR(BondConvexity(cf, r), maturity * maturity, 1e-12);
}

TEST(Ch02ZeroCoupon, Bootstrap) {
  // Price four semiannual coupon bonds off a known zero curve, then recover it.
  const auto zero = [](double t) { return 0.03 + 0.01 * std::log1p(t); };
  std::vector<BondQuote> quotes;
  for (double maturity : {2.0, 0.5, 1.5, 1.0}) {  // unsorted on purpose
    const double c = 0.04 + 0.01 * maturity;
    quotes.push_back(
        {maturity, c,
         BondPriceFromZeroRate(BondCashFlows(maturity, c, 2), zero)});
  }
  const auto curve = BootstrapZeroRates(quotes, 2);
  ASSERT_EQ(curve.times.size(), 4u);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(curve.times[i], 0.5 * static_cast<double>(i + 1), 1e-15);
    EXPECT_NEAR(curve.rates[i], zero(curve.times[i]), 1e-13);
  }
  // A gap in maturities leaves a coupon date undetermined.
  EXPECT_THROW(
      (void)BootstrapZeroRates({{0.5, 0.04, 99.0}, {1.5, 0.04, 98.0}}, 2),
      std::invalid_argument);
}

TEST(Ch02Newton, YieldRoundTrip) {
  for (double y : {0.0, 0.02, 0.065, 0.15}) {
    const auto cf = BondCashFlows(7.3, 0.05, 2);
    EXPECT_NEAR(BondYield(cf, BondPrice(cf, y)), y, 1e-12) << y;
  }
  // Premium bond (price > par): yield below the coupon rate.
  EXPECT_LT(BondYield(BondCashFlows(3, 0.06, 2), 101.0), 0.06);
}
