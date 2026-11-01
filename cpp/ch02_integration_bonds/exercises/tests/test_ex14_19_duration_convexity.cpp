#include <p4mfe/ch02/exercises/ex14_19_duration_convexity.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace p4mfe::ch02;
using namespace p4mfe::ch02::ex;

namespace {
// -B'/B and B''/B by finite differences of the price.
void expect_metrics_match_price(const CashFlows& cf, double y) {
    const double h = 1e-4;
    const double B = bond_price(cf, y), Bp = bond_price(cf, y + h), Bm = bond_price(cf, y - h);
    const auto m = bond_metrics(cf, y);
    EXPECT_NEAR(m.price, B, 1e-12);
    EXPECT_NEAR(m.duration, -(Bp - Bm) / (2 * h) / B, 1e-7);
    EXPECT_NEAR(m.convexity, (Bp - 2 * B + Bm) / (h * h) / B, 1e-5);
}
}  // namespace

TEST(Ch02Ex14, SemiannualBond) {
    const auto m = ex14_metrics();
    EXPECT_NEAR(m.price, 92.9839, 1e-4);
    EXPECT_NEAR(m.duration, 2.352418, 1e-6);
    EXPECT_NEAR(m.convexity, 5.736739, 1e-6);
    const auto cf = bond_cash_flows(2.5, 0.06, 2);
    ASSERT_EQ(cf.times.size(), 5u);
    EXPECT_DOUBLE_EQ(cf.times.front(), 0.5);
    EXPECT_DOUBLE_EQ(cf.amounts.back(), 103.0);
    expect_metrics_match_price(cf, 0.09);
}

TEST(Ch02Ex15, QuarterlyBond) {
    const auto m = ex15_metrics();
    EXPECT_NEAR(m.price, 101.7049, 1e-4);
    EXPECT_NEAR(m.duration, 1.118911, 1e-6);
    EXPECT_NEAR(m.convexity, 1.285705, 1e-6);
    const auto cf = bond_cash_flows(14.0 / 12.0, 0.08, 4);
    ASSERT_EQ(cf.times.size(), 5u);
    EXPECT_NEAR(cf.times.front(), 2.0 / 12.0, 1e-14);
    EXPECT_DOUBLE_EQ(cf.amounts.front(), 2.0);
    expect_metrics_match_price(cf, 0.07);
}

TEST(Ch02Ex16, HigherCouponHigherPriceLowerDuration) {
    for (double y : {0.02, 0.06, 0.12}) {
        double prev_B = 0.0, prev_D = 1e9;
        for (double c : {0.0, 0.01, 0.03, 0.05, 0.08, 0.15, 0.5}) {
            const auto m = ex16_metrics(10.0, c, 2, y);
            EXPECT_GT(m.price, prev_B) << c;
            EXPECT_LT(m.duration, prev_D) << c;
            prev_B = m.price;
            prev_D = m.duration;
        }
        EXPECT_DOUBLE_EQ(ex16_metrics(10.0, 0.0, 2, y).duration, 10.0);
    }
}

TEST(Ch02Ex17, TenYearZeroUpTenBp) {
    EXPECT_NEAR(ex17_exact_relative_change(), -0.00995017, 1e-8);
    EXPECT_DOUBLE_EQ(ex17_duration_estimate, -0.01);
    EXPECT_NEAR(ex17_convexity_estimate(), -0.00995, 1e-15);
    const auto cf = bond_cash_flows(10.0, 0.0, 1);
    for (double y : {0.01, 0.05, 0.1}) {
        const auto m = bond_metrics(cf, y);
        EXPECT_NEAR(m.duration, 10.0, 1e-14);
        EXPECT_NEAR(m.convexity, 100.0, 1e-12);
        EXPECT_NEAR(bond_price(cf, y + 0.001) / m.price - 1, ex17_exact_relative_change(), 1e-14);
    }
}

TEST(Ch02Ex18, DurationApproximation) {
    EXPECT_NEAR(ex18_new_price(), 103.785, 1e-12);
    // Check on an actual five year semiannual bond priced at 102:
    // the true price is a bit above the estimate (convexity).
    const auto cf = bond_cash_flows(5.0, 0.0905, 2);
    const double y = bond_yield(cf, 102.0);
    const auto m = bond_metrics(cf, y);
    const double true_price = bond_price(cf, y - 0.005);
    const double estimate = ex18_new_price(m.price, m.duration, -0.005);
    EXPECT_GT(true_price, estimate);
    EXPECT_NEAR(true_price, price_approx_duration_convexity(m.price, m.duration, m.convexity, -0.005), 1e-3);
}

TEST(Ch02Ex19, ConvexityIsDurationSquaredMinusSlope) {
    const auto cfs = {bond_cash_flows(2.5, 0.06, 2), bond_cash_flows(14.0 / 12.0, 0.08, 4),
                      bond_cash_flows(30.0, 0.04, 2), bond_cash_flows(7.0, 0.0, 1)};
    for (const auto& cf : cfs)
        for (double y : {0.01, 0.05, 0.09}) {
            const auto m = bond_metrics(cf, y);
            const double slope = ex19_duration_slope(cf, y);
            EXPECT_NEAR(m.convexity, m.duration * m.duration - slope, 1e-6 * m.convexity);
            EXPECT_LE(slope, 1e-9);
        }
}
