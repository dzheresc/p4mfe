#include <p4mfe/ch02/exercises/ex08_09_rate_curves.hpp>
#include <p4mfe/ch02/exercises/ex10_13_bond_pricing.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace p4mfe::ch02;
using namespace p4mfe::ch02::ex;

TEST(Ch02Ex08, ParallelShift) {
    const auto zero1 = [](double t) { return 0.03 + 0.02 * (1 - std::exp(-t)) / (1 + t) + 0.004 * std::sin(t); };
    const double dr = 0.0075;
    const auto zero2 = ex08_shift(zero1, dr);
    for (double t : {0.25, 1.0, 3.0, 10.0}) {
        // zero shift => instantaneous shift
        EXPECT_NEAR(inst_rate_from_zero_rate(zero2, t) - inst_rate_from_zero_rate(zero1, t), dr, 1e-9) << t;
    }
    // instantaneous shift => zero shift
    const auto r1 = [](double t) { return 0.04 + 0.01 * std::log1p(t) - 0.002 * t * std::exp(-t); };
    const auto r2 = [&](double t) { return r1(t) + dr; };
    for (double t : {0.25, 1.0, 3.0, 10.0})
        EXPECT_NEAR(zero_rate_from_inst_rate(r2, t) - zero_rate_from_inst_rate(r1, t), dr, 1e-12) << t;
}

TEST(Ch02Ex09, InstantaneousAndDiscreteCurves) {
    for (double t : {0.5, 1.0, 2.0, 5.0, 30.0}) {
        EXPECT_NEAR(ex09_inst_rate(t), inst_rate_from_zero_rate(ex09_zero_rate, t), 1e-9) << t;
        EXPECT_NEAR(ex09_zero_rate(t), zero_rate_from_inst_rate(ex09_inst_rate, t), 1e-12) << t;
        // Same discount factor under all three compounding conventions.
        const double D = std::exp(-ex09_zero_rate(t) * t);
        EXPECT_NEAR(discount_factor_discrete(ex09_annual_zero_rate(t), t, 1), D, 1e-14) << t;
        EXPECT_NEAR(discount_factor_discrete(ex09_semiannual_zero_rate(t), t, 2), D, 1e-14) << t;
        EXPECT_NEAR(ex09_annual_zero_rate(t), discrete_from_continuous(ex09_zero_rate(t), 1), 1e-15);
        EXPECT_NEAR(ex09_semiannual_zero_rate(t), discrete_from_continuous(ex09_zero_rate(t), 2), 1e-15);
    }
    EXPECT_NEAR(ex09_annual_zero_rate(0), std::exp(0.05) - 1, 1e-15);
    EXPECT_NEAR(ex09_semiannual_zero_rate(0), 2 * (std::exp(0.025) - 1), 1e-15);
}

TEST(Ch02Ex10, PriceFromZeroRates) {
    const auto d = ex10_discount_factors();
    EXPECT_NEAR(d[0], 0.975310, 1e-6);
    EXPECT_NEAR(d[1], 0.948854, 1e-6);
    EXPECT_NEAR(d[2], 0.922886, 1e-6);
    EXPECT_NEAR(d[3], 0.895834, 1e-6);
    EXPECT_NEAR(ex10_price(), 98.9406, 1e-4);
    EXPECT_NEAR(ex10_price(), 2.5 * (d[0] + d[1] + d[2] + d[3]) + 100 * d[3], 1e-12);
}

TEST(Ch02Ex11, ParYield) {
    const double c = ex11_par_yield();
    EXPECT_NEAR(c, 0.055661, 1e-6);
    EXPECT_NEAR(ex10_price(c), 100.0, 1e-10);
}

TEST(Ch02Ex12, ZeroCurveDiscountsAndPrice) {
    for (double t : {0.5, 1.0, 1.5, 2.0, 7.0}) {
        EXPECT_NEAR(ex12_zero_rate(t), zero_rate_from_inst_rate(ex12_inst_rate, t), 1e-12) << t;
        EXPECT_NEAR(ex12_discount_factor(t), discount_factor_from_inst_rate(ex12_inst_rate, t), 1e-12) << t;
    }
    EXPECT_NEAR(ex12_zero_rate(1e-9), 0.05, 1e-11);
    EXPECT_NEAR(ex12_discount_factor(0.5), 0.974782, 1e-6);
    EXPECT_NEAR(ex12_discount_factor(1.0), 0.949394, 1e-6);
    EXPECT_NEAR(ex12_discount_factor(1.5), 0.924083, 1e-6);
    EXPECT_NEAR(ex12_discount_factor(2.0), 0.898994, 1e-6);
    EXPECT_NEAR(ex12_price(), 99.2675, 1e-4);
    EXPECT_NEAR(ex12_price(), bond_price_from_inst_rate(bond_cash_flows(2.0, 0.05, 2), ex12_inst_rate), 1e-10);
}

TEST(Ch02Ex13, SimpsonDiscountFactors) {
    EXPECT_EQ(ex13_integral(1.0, 1e-6).intervals, 8u);
    EXPECT_EQ(ex13_integral(2.0, 1e-6).intervals, 16u);
    EXPECT_EQ(ex13_integral(3.0, 1e-8).intervals, 64u);
    for (double t : {1.0, 2.0, 3.0}) {
        const double exact = ex13_discount_factor(t, 1e-14);
        EXPECT_NEAR(ex13_discount_factor(t, ex13_tol(t)), exact, ex13_tol(t)) << t;
    }
    EXPECT_NEAR(ex13_discount_factor(1.0, 1e-6), 0.956595, 5e-7);
    EXPECT_NEAR(ex13_discount_factor(2.0, 1e-6), 0.910128, 5e-7);
    EXPECT_NEAR(ex13_discount_factor(3.0, 1e-8), 0.86574103, 5e-9);
    EXPECT_NEAR(ex13_price(), 100.2364, 1e-4);
    EXPECT_NEAR(ex13_price(), bond_price_from_inst_rate(bond_cash_flows(3.0, 0.05, 1), ex13_inst_rate), 1e-6);
}
