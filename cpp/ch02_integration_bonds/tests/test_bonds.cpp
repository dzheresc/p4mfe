#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/bonds.hpp>

using namespace p4mfe::ch02;

namespace {
double fd1(auto&& f, double x, double h = 1e-5) {
  return (f(x + h) - f(x - h)) / (2 * h);
}
double fd2(auto&& f, double x, double h = 1e-4) {
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}
}  // namespace

TEST(Ch02Bonds, CashFlows) {
  // 14 months to maturity, semiannual 5% coupon on 100: coupons of 2.5
  // at 2, 8 and 14 months.
  const auto cf = bond_cash_flows(14.0 / 12, 0.05, 2);
  ASSERT_EQ(cf.times.size(), 3u);
  EXPECT_NEAR(cf.times[0], 2.0 / 12, 1e-14);
  EXPECT_NEAR(cf.times[1], 8.0 / 12, 1e-14);
  EXPECT_NEAR(cf.times[2], 14.0 / 12, 1e-14);
  EXPECT_DOUBLE_EQ(cf.amounts[0], 2.5);
  EXPECT_DOUBLE_EQ(cf.amounts[2], 102.5);
  EXPECT_EQ(bond_cash_flows(3, 0.06, 2).times.size(), 6u);
  EXPECT_EQ(bond_cash_flows(3, 0.0, 2).times.size(), 1u);
}

TEST(Ch02Bonds, PriceFromCurves) {
  const auto cf = bond_cash_flows(3, 0.06, 2);
  // A flat zero curve at y prices the bond as yield y.
  EXPECT_NEAR(bond_price_from_zero_rate(cf, [](double) { return 0.05; }),
              bond_price(cf, 0.05), 1e-12);
  // Zero rate curve vs instantaneous rate curve (2.10).
  const auto r_inst = [](double t) { return 0.0525 + std::log1p(2 * t) / 200; };
  const auto r_zero = [](double t) {
    return 0.0525 + ((1 + 2 * t) * std::log1p(2 * t) - 2 * t) / (400 * t);
  };
  EXPECT_NEAR(bond_price_from_inst_rate(cf, r_inst),
              bond_price_from_zero_rate(cf, r_zero), 1e-10);
}

TEST(Ch02Bonds, PriceFromInstRateRejectsUnsortedTimes) {
  const auto r = [](double) { return 0.05; };
  const CashFlows unsorted{{1.0, 0.5}, {3.0, 103.0}};
  EXPECT_THROW((void)bond_price_from_inst_rate(unsorted, r),
               std::invalid_argument);
  const CashFlows negative{{-0.5, 1.0}, {3.0, 103.0}};
  EXPECT_THROW((void)bond_price_from_inst_rate(negative, r),
               std::invalid_argument);
}

TEST(Ch02Bonds, DurationAndConvexityAreYieldDerivatives) {
  const auto cf = bond_cash_flows(5, 0.04, 4);
  const double y = 0.045;
  const auto B = [&](double s) { return bond_price(cf, s); };
  const auto m = bond_metrics(cf, y);
  EXPECT_DOUBLE_EQ(m.price, B(y));
  EXPECT_NEAR(m.duration, -fd1(B, y) / m.price, 1e-8);
  EXPECT_NEAR(m.convexity, fd2(B, y) / m.price, 1e-4);
  EXPECT_DOUBLE_EQ(bond_duration(cf, y), m.duration);
  EXPECT_DOUBLE_EQ(bond_convexity(cf, y), m.convexity);
  // Coupons shorten duration below maturity.
  EXPECT_LT(m.duration, 5.0);
  // DV01 is the linear part of the 1bp move; the convexity term is ~1e-5.
  EXPECT_NEAR(dv01(cf, y), B(y - 1e-4) - B(y), 2e-5);
  EXPECT_NEAR(dv01(cf, y), (B(y - 1e-4) - B(y + 1e-4)) / 2, 1e-8);
}

TEST(Ch02Bonds, PriceApproximations) {
  const auto cf = bond_cash_flows(10, 0.05, 2);
  const double y = 0.05, dy = 0.01;
  const auto m = bond_metrics(cf, y);
  const double exact = bond_price(cf, y + dy);
  const double e1 =
      std::abs(price_approx_duration(m.price, m.duration, dy) - exact);
  const double e2 = std::abs(
      price_approx_duration_convexity(m.price, m.duration, m.convexity, dy) -
      exact);
  EXPECT_LT(e2, e1 / 10);
  // Convexity: the duration line underestimates the price either way.
  EXPECT_LT(price_approx_duration(m.price, m.duration, dy), exact);
  EXPECT_LT(price_approx_duration(m.price, m.duration, -dy),
            bond_price(cf, y - dy));
}

TEST(Ch02Bonds, DiscreteYield) {
  // On a coupon date, a bond whose yield (compounded at the coupon
  // frequency) equals its coupon rate trades at par.
  const auto cf = bond_cash_flows(3, 0.06, 2);
  EXPECT_NEAR(bond_price_discrete(cf, 0.06, 2), 100.0, 1e-12);
  // Same price as the equivalent continuously compounded yield.
  EXPECT_NEAR(bond_price(cf, continuous_from_discrete(0.06, 2)), 100.0, 1e-12);
  EXPECT_NEAR(macaulay_duration(cf, 0.06, 2),
              bond_duration(cf, continuous_from_discrete(0.06, 2)), 1e-12);
  const auto B = [&](double s) { return bond_price_discrete(cf, s, 2); };
  EXPECT_NEAR(modified_duration(cf, 0.06, 2), -fd1(B, 0.06) / 100.0, 1e-8);
}

TEST(Ch02ZeroCoupon, PriceYieldDuration) {
  const double T = 4, r = 0.035;
  const double P = zero_coupon_bond_price(T, r);
  EXPECT_NEAR(P, 100 * std::exp(-0.14), 1e-12);
  EXPECT_NEAR(zero_rate_from_zero_coupon_price(P, T), r, 1e-15);
  const auto cf = bond_cash_flows(T, 0.0, 2);
  EXPECT_NEAR(bond_yield(cf, P), r, 1e-12);
  EXPECT_NEAR(bond_duration(cf, r), T, 1e-14);
  EXPECT_NEAR(bond_convexity(cf, r), T * T, 1e-12);
}

TEST(Ch02ZeroCoupon, Bootstrap) {
  // Price four semiannual coupon bonds off a known zero curve, then recover it.
  const auto zero = [](double t) { return 0.03 + 0.01 * std::log1p(t); };
  std::vector<BondQuote> quotes;
  for (double T : {2.0, 0.5, 1.5, 1.0}) {  // unsorted on purpose
    const double c = 0.04 + 0.01 * T;
    quotes.push_back(
        {T, c, bond_price_from_zero_rate(bond_cash_flows(T, c, 2), zero)});
  }
  const auto curve = bootstrap_zero_rates(quotes, 2);
  ASSERT_EQ(curve.times.size(), 4u);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(curve.times[i], 0.5 * static_cast<double>(i + 1), 1e-15);
    EXPECT_NEAR(curve.rates[i], zero(curve.times[i]), 1e-13);
  }
  // A gap in maturities leaves a coupon date undetermined.
  EXPECT_THROW(
      (void)bootstrap_zero_rates({{0.5, 0.04, 99.0}, {1.5, 0.04, 98.0}}, 2),
      std::invalid_argument);
}

TEST(Ch02Newton, YieldRoundTrip) {
  for (double y : {0.0, 0.02, 0.065, 0.15}) {
    const auto cf = bond_cash_flows(7.3, 0.05, 2);
    EXPECT_NEAR(bond_yield(cf, bond_price(cf, y)), y, 1e-12) << y;
  }
  // Premium bond (price > par): yield below the coupon rate.
  EXPECT_LT(bond_yield(bond_cash_flows(3, 0.06, 2), 101.0), 0.06);
}
