#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/options.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

using namespace p4mfe::ch01;

TEST(Ch01Payoff, CallAndPut) {
    EXPECT_DOUBLE_EQ(call_payoff(110, 100), 10);
    EXPECT_DOUBLE_EQ(call_payoff(90, 100), 0);
    EXPECT_DOUBLE_EQ(put_payoff(90, 100), 10);
    EXPECT_DOUBLE_EQ(payoff(OptionType::Put, 110, 100), 0);
    // Long call + short put pays S - K for every S.
    for (double S : {50.0, 100.0, 150.0}) EXPECT_DOUBLE_EQ(call_payoff(S, 100) - put_payoff(S, 100), S - 100);
}

TEST(Ch01Parity, RoundTrip) {
    const double S = 42, K = 40, T = 0.5, r = 0.05, q = 0.02, C = 4.0;
    const double P = put_from_call(C, S, K, T, r, q);
    EXPECT_NEAR(call_from_put(P, S, K, T, r, q), C, 1e-12);
    EXPECT_NEAR(parity_violation(C, P, S, K, T, r, q), 0.0, 1e-12);
    EXPECT_FALSE(parity_arbitrage(C, P, S, K, T, r, q).has_value());
}

TEST(Ch01Parity, BoundsHoldForParityConsistentPrices) {
    const double S = 42, K = 40, T = 0.5, r = 0.05, q = 0.02, C = 4.0;
    const auto cb = call_bounds(S, K, T, r, q);
    const auto pb = put_bounds(S, K, T, r, q);
    const double P = put_from_call(C, S, K, T, r, q);
    EXPECT_LE(cb.lower, C);
    EXPECT_LE(C, cb.upper);
    EXPECT_LE(pb.lower, P);
    EXPECT_LE(P, pb.upper);
}

// Check the arbitrage portfolio costs nothing today and pays `profit` at T
// for any terminal price.
static void check_arbitrage(double C, double P, double S, double K, double T, double r, double q) {
    const auto arb = parity_arbitrage(C, P, S, K, T, r, q);
    ASSERT_TRUE(arb.has_value());
    EXPECT_NEAR(arb->calls * C + arb->puts * P + arb->shares * S + arb->cash, 0.0, 1e-12);
    EXPECT_GT(arb->profit, 0.0);
    for (double ST : {0.0, 20.0, K, 60.0, 200.0}) {
        const double value_T = arb->calls * call_payoff(ST, K) + arb->puts * put_payoff(ST, K) +
                               arb->shares * std::exp(q * T) * ST + arb->cash * std::exp(r * T);
        EXPECT_NEAR(value_T, arb->profit, 1e-10) << "S_T=" << ST;
    }
}

TEST(Ch01Parity, ArbitrageWhenCallRich) { check_arbitrage(5.0, 1.0, 42, 40, 0.5, 0.05, 0.02); }
TEST(Ch01Parity, ArbitrageWhenCallCheap) { check_arbitrage(2.0, 1.0, 42, 40, 0.5, 0.05, 0.0); }

TEST(Ch01Forward, PriceAndValue) {
    const double S = 100, T = 1.5, r = 0.04, q = 0.01;
    const double F = forward_price(S, T, r, q);
    EXPECT_NEAR(F, S * std::exp(0.03 * 1.5), 1e-12);
    EXPECT_NEAR(long_forward_value(S, F, T, r, q), 0.0, 1e-12);  // fair forward costs nothing
    const double K = 95;
    EXPECT_NEAR(long_forward_value(S, K, T, r, q), (F - K) * discount_factor(r, T), 1e-12);
}

TEST(Ch01Forward, DiscreteDividends) {
    const std::array<Dividend, 2> divs{{{0.25, 1.0}, {0.75, 1.0}}};
    const double S = 50, T = 1, r = 0.06;
    const double expected = (S - std::exp(-0.06 * 0.25) - std::exp(-0.06 * 0.75)) * std::exp(0.06);
    EXPECT_NEAR(forward_price(S, T, r, divs), expected, 1e-12);
    EXPECT_NEAR(forward_price(S, T, r, std::span<const Dividend>{}), forward_price(S, T, r), 1e-12);
}

TEST(Ch01Forward, SyntheticForwardIsPutCallParity) {
    const double S = 42, K = 40, T = 0.5, r = 0.05, q = 0.02, C = 4.0;
    const double P = put_from_call(C, S, K, T, r, q);
    EXPECT_NEAR(synthetic_forward_value(C, P), long_forward_value(S, K, T, r, q), 1e-12);
    // At K = F the call and put have equal prices.
    const double F = forward_price(S, T, r, q);
    EXPECT_NEAR(put_from_call(C, S, F, T, r, q), C, 1e-12);
}
