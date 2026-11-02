#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/exercises/ex12_15_black_scholes.hpp>
#include <p4mfe/ch03/exercises/ex16_21_greeks_strike.hpp>

using namespace p4mfe::ch03;
using namespace p4mfe::ch03::ex;

namespace {
struct Case {
  double spot, strike, maturity, sigma, r, q;
};
const Case kCases[] = {{50, 45, 0.5, 0.2, 0.06, 0.02},
                       {100, 110, 1.5, 0.35, 0.03, 0.0},
                       {40, 38, 0.25, 0.15, 0.01, 0.04},
                       {75, 75, 3.0, 0.25, 0.05, 0.05}};
}  // namespace

TEST(Ch03Ex12, CallAndPut) {
  const auto [d1, d2] = BsD1D2(50, 45, 0.5, 0.2, 0.06, 0.02);
  EXPECT_NEAR(d1, 0.957143, 1e-6);
  EXPECT_NEAR(d2, 0.815722, 1e-6);
  EXPECT_NEAR(Ex12Call(), 6.508365, 1e-6);
  EXPECT_NEAR(Ex12Put(), 0.675922, 1e-6);
  EXPECT_NEAR(Ex12Call() - Ex12Put(),
              50 * std::exp(-0.01) - 45 * std::exp(-0.03), 1e-12);
}

TEST(Ch03Ex13, ZeroStrike) {
  EXPECT_EQ(Ex13PutZeroStrike(), 0.0);
  // Black-Scholes as K -> 0 agrees with the model-free values.
  EXPECT_NEAR(BsCall(80, 1e-8, 2.0, 0.3, 0.05, 0.03),
              Ex13CallZeroStrike(80, 2.0, 0.03), 1e-7);
  EXPECT_NEAR(BsPut(80, 1e-8, 2.0, 0.3, 0.05, 0.03), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(Ex13CallZeroStrike(80, 2.0), 80.0);
}

TEST(Ch03Ex14, PutRhoFromParity) {
  for (const auto& c : kCases) {
    const double rho = BsGreeks(OptionType::kPut, c.spot, c.strike, c.maturity,
                                c.sigma, c.r, c.q)
                           .rho;
    EXPECT_NEAR(
        Ex14RhoPutFromParity(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q),
        rho, 1e-10);
    const double h = 1e-5;
    const double fd =
        (BsPut(c.spot, c.strike, c.maturity, c.sigma, c.r + h, c.q) -
         BsPut(c.spot, c.strike, c.maturity, c.sigma, c.r - h, c.q)) /
        (2 * h);
    EXPECT_NEAR(rho, fd, 1e-5);
  }
}

TEST(Ch03Ex15, AssetOrNothingDelta) {
  for (const auto& c : kCases) {
    const double sigma = 0.3, h = 1e-4;
    const double fd =
        (Ex15AonCall(c.spot + h, c.strike, c.maturity, sigma, c.r, c.q) -
         Ex15AonCall(c.spot - h, c.strike, c.maturity, sigma, c.r, c.q)) /
        (2 * h);
    EXPECT_NEAR(Ex15AonDelta(c.spot, c.strike, c.maturity, sigma, c.r, c.q), fd,
                1e-7);
    // Larger than the plain call's Delta.
    EXPECT_GT(Ex15AonDelta(c.spot, c.strike, c.maturity, sigma, c.r, c.q),
              BsGreeks(OptionType::kCall, c.spot, c.strike, c.maturity, sigma,
                       c.r, c.q)
                  .delta);
  }
}

TEST(Ch03Ex16, VolgaVanna) {
  for (const auto& c : kCases) {
    const double h = 1e-4;
    const auto vega = [&](OptionType t, double spot, double sg) {
      return BsGreeks(t, spot, c.strike, c.maturity, sg, c.r, c.q).vega;
    };
    const auto delta = [&](OptionType t, double sg) {
      return BsGreeks(t, c.spot, c.strike, c.maturity, sg, c.r, c.q).delta;
    };
    for (auto t : {OptionType::kCall, OptionType::kPut}) {
      const double volga_fd =
          (vega(t, c.spot, c.sigma + h) - vega(t, c.spot, c.sigma - h)) /
          (2 * h);
      const double vanna_fd =
          (vega(t, c.spot + h, c.sigma) - vega(t, c.spot - h, c.sigma)) /
          (2 * h);
      const double vanna_fd2 =
          (delta(t, c.sigma + h) - delta(t, c.sigma - h)) / (2 * h);
      EXPECT_NEAR(Ex16Volga(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q),
                  volga_fd, 1e-5);
      EXPECT_NEAR(Ex16Vanna(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q),
                  vanna_fd, 1e-6);
      EXPECT_NEAR(Ex16Vanna(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q),
                  vanna_fd2, 1e-6);
    }
  }
}

TEST(Ch03Ex17, AtmCallVersusPut) {
  for (double r : {0.01, 0.04, 0.08})
    for (double q : {0.0, 0.04, 0.1}) {
      const double diff = Ex17AtmCallMinusPut(60, 1.5, 0.25, r, q);
      EXPECT_NEAR(diff, 60 * (std::exp(-q * 1.5) - std::exp(-r * 1.5)), 1e-11);
      if (q < r) {
        EXPECT_GT(diff, 0.0);
      }
      if (q > r) {
        EXPECT_LT(diff, 0.0);
      }
      if (q == r) {
        EXPECT_NEAR(diff, 0.0, 1e-11);
      }
    }
}

TEST(Ch03Ex18, ThetaSign) {
  for (double spot : {20.0, 50.0, 80.0})
    for (double maturity : {0.01, 0.5, 5.0, 50.0})
      for (double r : {0.0, 0.03, 0.1})
        EXPECT_LT(Ex18ThetaCall(spot, 50, maturity, 0.3, r), 0.0);
  // q = r ATM: negative for short, positive for long maturities.
  const double maturity_star = Ex18ThetaSignChange(0.2, 0.05);
  EXPECT_NEAR(maturity_star, 9.6814, 1e-4);
  EXPECT_LT(Ex18ThetaCall(100, 100, 1.0, 0.2, 0.05, 0.05), 0.0);
  EXPECT_GT(Ex18ThetaCall(100, 100, 10.0, 0.2, 0.05, 0.05), 0.0);
  EXPECT_NEAR(Ex18ThetaCall(100, 100, 10.0, 0.2, 0.05, 0.05), 0.024754, 1e-6);
  for (double maturity : {0.5, 5.0, 20.0})
    EXPECT_NEAR(Ex18ThetaCall(100, 100, maturity, 0.2, 0.05, 0.05),
                100 * std::exp(-0.05 * maturity) *
                    Ex18AtmQEqRBracket(maturity, 0.2, 0.05),
                1e-11);
}

TEST(Ch03Ex19_20, StrikeDerivatives) {
  for (const auto& c : kCases) {
    const double h = 1e-3;
    const auto call = [&](double strike) {
      return BsCall(c.spot, strike, c.maturity, c.sigma, c.r, c.q);
    };
    const auto put = [&](double strike) {
      return BsPut(c.spot, strike, c.maturity, c.sigma, c.r, c.q);
    };
    const auto d =
        Ex19StrikeDerivatives(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q);
    EXPECT_NEAR(d.d_call_d_strike,
                (call(c.strike + h) - call(c.strike - h)) / (2 * h), 1e-8);
    EXPECT_NEAR(d.d_put_d_strike,
                (put(c.strike + h) - put(c.strike - h)) / (2 * h), 1e-8);
    EXPECT_NEAR(d.d2_call_d_strike2,
                (call(c.strike + h) - 2 * call(c.strike) + call(c.strike - h)) /
                    (h * h),
                1e-5);
    EXPECT_NEAR(
        d.d2_put_d_strike2,
        (put(c.strike + h) - 2 * put(c.strike) + put(c.strike - h)) / (h * h),
        1e-5);
    EXPECT_GT(d.d2_call_d_strike2, 0.0);  // exercise 20
    EXPECT_NEAR(d.d_put_d_strike - d.d_call_d_strike,
                std::exp(-c.r * c.maturity), 1e-14);
  }
}

TEST(Ch03Ex21, MinimalStraddle) {
  for (const auto& c : kCases) {
    const double min_strike =
        Ex21MinStraddleStrike(c.spot, c.maturity, c.sigma, c.r, c.q);
    EXPECT_NEAR(BsD1D2(c.spot, min_strike, c.maturity, c.sigma, c.r, c.q).d2,
                0.0, 1e-12);
    const double f =
        Ex21Straddle(c.spot, min_strike, c.maturity, c.sigma, c.r, c.q);
    for (double d_strike : {-1.0, -0.1, 0.1, 1.0})
      EXPECT_GT(Ex21Straddle(c.spot, min_strike + d_strike, c.maturity, c.sigma,
                             c.r, c.q),
                f);
    // Golden-section search lands on the same strike.
    double a = 0.5 * c.spot, b = 1.5 * c.spot;
    const double g = (std::sqrt(5.0) - 1) / 2;
    for (int i = 0; i < 100; ++i) {
      const double x1 = b - g * (b - a), x2 = a + g * (b - a);
      if (Ex21Straddle(c.spot, x1, c.maturity, c.sigma, c.r, c.q) <
          Ex21Straddle(c.spot, x2, c.maturity, c.sigma, c.r, c.q))
        b = x2;
      else
        a = x1;
    }
    EXPECT_NEAR(0.5 * (a + b), min_strike, 1e-4 * c.spot);
  }
}
