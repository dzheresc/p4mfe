#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/cdf_approx.hpp>
#include <stdexcept>

using namespace p4mfe::ch03;
using p4mfe::ch01::Derivative;
using p4mfe::ch01::SecondDerivative;

TEST(Ch03BlackScholes, ReferenceValues) {
  // Hull's example: S = 42, K = 40, T = 0.5, sigma = 20%, r = 10%.
  EXPECT_NEAR(BsCall(42, 40, 0.5, 0.2, 0.1), 4.7594, 5e-5);
  EXPECT_NEAR(BsPut(42, 40, 0.5, 0.2, 0.1), 0.8086, 5e-5);
  EXPECT_NEAR(BsCall(100, 100, 1, 0.2, 0.05), 10.450583572185565, 1e-12);
  EXPECT_NEAR(BsPut(100, 100, 1, 0.2, 0.05), 5.573526022256971, 1e-12);
}

TEST(Ch03BlackScholes, PutCallParityWithDividends) {
  const double spot = 40, strike = 45, maturity = 0.75, sigma = 0.3, r = 0.04,
               q = 0.02;
  const double call = BsCall(spot, strike, maturity, sigma, r, q);
  const double put = BsPut(spot, strike, maturity, sigma, r, q);
  EXPECT_NEAR(
      p4mfe::ch01::ParityViolation(call, put, spot, strike, maturity, r, q),
      0.0, 1e-12);
}

TEST(Ch03BlackScholes, LimitsAndBadInput) {
  // Very low volatility: price tends to the discounted forward intrinsic value.
  EXPECT_NEAR(BsCall(50, 40, 1, 1e-6, 0.05, 0.01),
              50 * std::exp(-0.01) - 40 * std::exp(-0.05), 1e-10);
  EXPECT_THROW((void)BsCall(50, 40, 0, 0.2, 0.05), std::invalid_argument);
  EXPECT_THROW((void)BsCall(50, 40, 1, 0.0, 0.05), std::invalid_argument);
}

class Ch03Greeks : public ::testing::TestWithParam<OptionType> {};

// Closed-form Greeks agree with finite differences of the price (3.6.1).
TEST_P(Ch03Greeks, MatchFiniteDifferences) {
  const OptionType type = GetParam();
  const double spot = 40, strike = 42, maturity = 0.5, sigma = 0.27, r = 0.05,
               q = 0.015;
  const auto g = BsGreeks(type, spot, strike, maturity, sigma, r, q);
  EXPECT_NEAR(g.delta,
              Derivative(
                  [&](double s) {
                    return BsPrice(type, s, strike, maturity, sigma, r, q);
                  },
                  spot),
              1e-8);
  EXPECT_NEAR(g.gamma,
              SecondDerivative(
                  [&](double s) {
                    return BsPrice(type, s, strike, maturity, sigma, r, q);
                  },
                  spot),
              1e-6);
  EXPECT_NEAR(g.vega,
              Derivative(
                  [&](double v) {
                    return BsPrice(type, spot, strike, maturity, v, r, q);
                  },
                  sigma),
              1e-7);
  EXPECT_NEAR(g.rho,
              Derivative(
                  [&](double x) {
                    return BsPrice(type, spot, strike, maturity, sigma, x, q);
                  },
                  r),
              1e-7);
  // Theta is dV/dt = -dV/dT.
  EXPECT_NEAR(
      g.theta,
      -Derivative(
          [&](double t) { return BsPrice(type, spot, strike, t, sigma, r, q); },
          maturity),
      1e-7);
}

TEST_P(Ch03Greeks, SatisfyBlackScholesPde) {
  for (double spot : {20.0, 40.0, 60.0})
    EXPECT_NEAR(BsPdeResidual(GetParam(), spot, 42, 0.5, 0.27, 0.05, 0.015), 0,
                1e-12);
}

INSTANTIATE_TEST_SUITE_P(CallAndPut, Ch03Greeks,
                         ::testing::Values(OptionType::kCall,
                                           OptionType::kPut));

TEST(Ch03Greeks, CallPutRelations) {
  const double spot = 40, strike = 42, maturity = 0.5, sigma = 0.27, r = 0.05,
               q = 0.015;
  const auto c =
      BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r, q);
  const auto p =
      BsGreeks(OptionType::kPut, spot, strike, maturity, sigma, r, q);
  EXPECT_NEAR(c.delta - p.delta, std::exp(-q * maturity),
              1e-15);  // from Put-Call parity
  EXPECT_DOUBLE_EQ(c.gamma, p.gamma);
  EXPECT_DOUBLE_EQ(c.vega, p.vega);
  EXPECT_NEAR(c.rho - p.rho, strike * maturity * std::exp(-r * maturity),
              1e-13);
}

TEST(Ch03Greeks, PhiIdentity) {
  for (double spot : {10.0, 40.0, 100.0})
    EXPECT_NEAR(BsPhiIdentityGap(spot, 42, 0.5, 0.27, 0.05, 0.015), 0, 1e-14);
}
