#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/options.hpp>

using namespace p4mfe::ch01;

TEST(Ch01Payoff, CallAndPut) {
  EXPECT_DOUBLE_EQ(CallPayoff(110, 100), 10);
  EXPECT_DOUBLE_EQ(CallPayoff(90, 100), 0);
  EXPECT_DOUBLE_EQ(PutPayoff(90, 100), 10);
  EXPECT_DOUBLE_EQ(Payoff(OptionType::kPut, 110, 100), 0);
  // Long call + short put pays S - K for every S.
  for (double spot : {50.0, 100.0, 150.0})
    EXPECT_DOUBLE_EQ(CallPayoff(spot, 100) - PutPayoff(spot, 100), spot - 100);
}

TEST(Ch01Parity, RoundTrip) {
  const double spot = 42, strike = 40, maturity = 0.5, r = 0.05, q = 0.02,
               call = 4.0;
  const double put = PutFromCall(call, spot, strike, maturity, r, q);
  EXPECT_NEAR(CallFromPut(put, spot, strike, maturity, r, q), call, 1e-12);
  EXPECT_NEAR(ParityViolation(call, put, spot, strike, maturity, r, q), 0.0,
              1e-12);
  EXPECT_FALSE(
      FindParityArbitrage(call, put, spot, strike, maturity, r, q).has_value());
}

TEST(Ch01Parity, BoundsHoldForParityConsistentPrices) {
  const double spot = 42, strike = 40, maturity = 0.5, r = 0.05, q = 0.02,
               call = 4.0;
  const auto cb = CallBounds(spot, strike, maturity, r, q);
  const auto pb = PutBounds(spot, strike, maturity, r, q);
  const double put = PutFromCall(call, spot, strike, maturity, r, q);
  EXPECT_LE(cb.lower, call);
  EXPECT_LE(call, cb.upper);
  EXPECT_LE(pb.lower, put);
  EXPECT_LE(put, pb.upper);
}

// Check the arbitrage portfolio costs nothing today and pays `profit` at T
// for any terminal price.
static void CheckArbitrage(double call, double put, double spot, double strike,
                           double maturity, double r, double q) {
  const auto arb = FindParityArbitrage(call, put, spot, strike, maturity, r, q);
  ASSERT_TRUE(arb.has_value());
  EXPECT_NEAR(
      arb->calls * call + arb->puts * put + arb->shares * spot + arb->cash, 0.0,
      1e-12);
  EXPECT_GT(arb->profit, 0.0);
  for (double spot_t : {0.0, 20.0, strike, 60.0, 200.0}) {
    const double value_at_maturity =
        arb->calls * CallPayoff(spot_t, strike) +
        arb->puts * PutPayoff(spot_t, strike) +
        arb->shares * std::exp(q * maturity) * spot_t +
        arb->cash * std::exp(r * maturity);
    EXPECT_NEAR(value_at_maturity, arb->profit, 1e-10) << "S_T=" << spot_t;
  }
}

TEST(Ch01Parity, ArbitrageWhenCallRich) {
  CheckArbitrage(5.0, 1.0, 42, 40, 0.5, 0.05, 0.02);
}
TEST(Ch01Parity, ArbitrageWhenCallCheap) {
  CheckArbitrage(2.0, 1.0, 42, 40, 0.5, 0.05, 0.0);
}

TEST(Ch01Forward, PriceAndValue) {
  const double spot = 100, maturity = 1.5, r = 0.04, q = 0.01;
  const double forward = ForwardPrice(spot, maturity, r, q);
  EXPECT_NEAR(forward, spot * std::exp(0.03 * 1.5), 1e-12);
  EXPECT_NEAR(LongForwardValue(spot, forward, maturity, r, q), 0.0,
              1e-12);  // fair forward costs nothing
  const double strike = 95;
  EXPECT_NEAR(LongForwardValue(spot, strike, maturity, r, q),
              (forward - strike) * DiscountFactor(r, maturity), 1e-12);
}

TEST(Ch01Forward, DiscreteDividends) {
  const std::array<Dividend, 2> divs{{{0.25, 1.0}, {0.75, 1.0}}};
  const double spot = 50, maturity = 1, r = 0.06;
  const double expected =
      (spot - std::exp(-0.06 * 0.25) - std::exp(-0.06 * 0.75)) * std::exp(0.06);
  EXPECT_NEAR(ForwardPrice(spot, maturity, r, divs), expected, 1e-12);
  EXPECT_NEAR(ForwardPrice(spot, maturity, r, std::span<const Dividend>{}),
              ForwardPrice(spot, maturity, r), 1e-12);
}

TEST(Ch01Forward, SyntheticForwardIsPutCallParity) {
  const double spot = 42, strike = 40, maturity = 0.5, r = 0.05, q = 0.02,
               call = 4.0;
  const double put = PutFromCall(call, spot, strike, maturity, r, q);
  EXPECT_NEAR(SyntheticForwardValue(call, put),
              LongForwardValue(spot, strike, maturity, r, q), 1e-12);
  // At K = F the call and put have equal prices.
  const double forward = ForwardPrice(spot, maturity, r, q);
  EXPECT_NEAR(PutFromCall(call, spot, forward, maturity, r, q), call, 1e-12);
}
