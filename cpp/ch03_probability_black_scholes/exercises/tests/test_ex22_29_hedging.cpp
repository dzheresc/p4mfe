#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <p4mfe/ch03/exercises/ex22_24_maturity_bounds.hpp>
#include <p4mfe/ch03/exercises/ex25_29_hedging.hpp>

using namespace p4mfe;
using namespace p4mfe::ch03;
using namespace p4mfe::ch03::ex;

TEST(Ch03Ex22, AtmGammaFallsWithMaturity) {
  EXPECT_NEAR(Ex22AtmGamma(kEx22Maturities[0]), 0.130022, 1e-6);
  EXPECT_NEAR(Ex22AtmGamma(kEx22Maturities[1]), 0.052530, 1e-6);
  EXPECT_NEAR(Ex22AtmGamma(kEx22Maturities[2]), 0.025296, 1e-6);
  // ~ 1/sqrt(T) as T -> 0.
  const double maturity = 1e-6;
  EXPECT_NEAR(Ex22AtmGamma(maturity) * 50 * 0.3 * std::sqrt(maturity),
              norm_pdf(0.0), 1e-5);
}

TEST(Ch03Ex23, AtmVegaRisesWithMaturity) {
  EXPECT_NEAR(Ex23AtmVega(kEx22Maturities[0]), 4.069779, 1e-6);
  EXPECT_NEAR(Ex23AtmVega(kEx22Maturities[1]), 9.945546, 1e-6);
  EXPECT_NEAR(Ex23AtmVega(kEx22Maturities[2]), 19.723967, 1e-6);
  for (double tau : {0.1, 1.0, 10.0, 44.0, 60.0}) {
    const double h = 1e-5;
    const double fd = (Ex23AtmVega(tau + h) - Ex23AtmVega(tau - h)) / (2 * h);
    EXPECT_NEAR(Ex23VegaMaturitySlope(tau), fd, 1e-6);
  }
  EXPECT_GT(Ex23VegaMaturitySlope(44.0), 0.0);
  EXPECT_LT(Ex23VegaMaturitySlope(45.0), 0.0);  // 4 / 0.3^2 = 44.4
  // vega(P) = vega(C) > 0.
  EXPECT_DOUBLE_EQ(
      BsGreeks(OptionType::kPut, 50, 55, 0.7, 0.3, 0.02, 0.01).vega,
      BsGreeks(OptionType::kCall, 50, 55, 0.7, 0.3, 0.02, 0.01).vega);
}

TEST(Ch03Ex24, BoundsHoldForBlackScholes) {
  for (double spot : {60.0, 100.0, 140.0})
    for (double sigma : {0.05, 0.3, 1.5})
      for (double q : {0.0, 0.03}) {
        const double strike = 100, maturity = 1.2, r = 0.05;
        const auto cb = Ex24CallBounds(spot, strike, maturity, r, q);
        const auto pb = Ex24PutBounds(spot, strike, maturity, r, q);
        const double call = BsCall(spot, strike, maturity, sigma, r, q),
                     put = BsPut(spot, strike, maturity, sigma, r, q);
        EXPECT_LE(cb.lower, call + 1e-12);
        EXPECT_LE(call, cb.upper + 1e-12);
        EXPECT_LE(pb.lower,
                  put + 1e-12);  // equal to rounding when deep ITM, low vol
        EXPECT_LE(put, pb.upper + 1e-12);
        // Through Put-Call parity, C >= 0 gives P's lower bound, and C's
        // lower bound gives P >= 0.
        EXPECT_NEAR(ch01::PutFromCall(0.0, spot, strike, maturity, r, q),
                    pb.lower, 1e-12);
        EXPECT_NEAR(ch01::PutFromCall(cb.lower, spot, strike, maturity, r, q),
                    0.0, 1e-12);
        EXPECT_NEAR(ch01::PutFromCall(cb.upper, spot, strike, maturity, r, q),
                    pb.upper, 1e-12);
        EXPECT_NEAR(std::max(pb.lower, 0.0),
                    ch01::PutBounds(spot, strike, maturity, r, q).lower, 1e-12);
      }
}

TEST(Ch03Ex25, DeltaGammaNeutral) {
  const auto h = Ex25OptionsOnly(5000, -200, 0.4, 0.05, -0.5, 0.07);
  EXPECT_NEAR(h.calls, -4716.98, 0.01);
  EXPECT_NEAR(h.puts, 6226.42, 0.01);
  EXPECT_NEAR(5000 + 0.4 * h.calls - 0.5 * h.puts, 0.0, 1e-9);
  EXPECT_NEAR(-200 + 0.05 * h.calls + 0.07 * h.puts, 0.0, 1e-9);
  // With the asset: the library's Delta-Gamma hedge.
  const auto c = ComputeDeltaGammaHedge(5000, -200, 0.4, 0.05);
  EXPECT_NEAR(c.hedge_options, 4000, 1e-9);
  EXPECT_NEAR(c.shares, -6600, 1e-9);
  const auto p = ComputeDeltaGammaHedge(5000, -200, -0.5, 0.07);
  EXPECT_NEAR(p.hedge_options, 2857.142857, 1e-6);
  EXPECT_NEAR(p.shares, -3571.428571, 1e-6);
}

TEST(Ch03Ex26, ShortSharesAgainstCalls) {
  EXPECT_NEAR(Ex26Shares(), -653.5007, 1e-4);
  // Delta does not depend on the drift: only r enters d1.
  EXPECT_NEAR(BsD1D2(92, 90, 0.25, 0.2, 0.05).d1,
              (std::log(92.0 / 90) + 0.07 * 0.25) / 0.1, 1e-14);
}

TEST(Ch03Ex27, DeltaHedgedCalls) {
  const auto e = Ex27();
  EXPECT_NEAR(e.premium, 9634.8766, 1e-4);
  EXPECT_NEAR(e.shares, -588.5891, 1e-4);
  EXPECT_NEAR(e.cash, 49224.0347, 1e-4);
  EXPECT_NEAR(e.position_value, -49228.6098, 1e-4);
  EXPECT_NEAR(e.portfolio_value, 5.1926, 1e-4);
  EXPECT_NEAR(e.unhedged_loss, 1181.7533, 1e-4);
  EXPECT_LT(std::abs(e.portfolio_value), 0.01 * e.unhedged_loss);
}

TEST(Ch03Ex28, PutPortfolio) {
  const auto e = Ex28();
  EXPECT_NEAR(e.value0, 22927.3505, 1e-4);
  EXPECT_NEAR(e.port_delta0, -402.7713, 1e-4);
  EXPECT_NEAR(e.shares1, 802.7713, 1e-4);
  EXPECT_NEAR(e.cash1, 1944.5731, 1e-4);
  EXPECT_NEAR(e.value1, 23399.3380, 1e-4);
  EXPECT_NEAR(e.shares2, 511.1342, 1e-4);
  EXPECT_NEAR(e.shares1 - e.shares2, 291.6371, 1e-4);
}

TEST(Ch03Ex29, DeltaGammaVegaNeutral) {
  const auto h = Ex29ComputeHedge();
  EXPECT_NEAR(h.calls, -670, 1e-9);
  EXPECT_NEAR(h.puts, -110, 1e-9);
  EXPECT_NEAR(h.shares, -254, 1e-9);
  EXPECT_NEAR(300 + 0.2 * h.calls - 0.8 * h.puts + h.shares, 0.0, 1e-9);
  EXPECT_NEAR(100 + 0.1 * h.calls + 0.3 * h.puts, 0.0, 1e-9);
  EXPECT_NEAR(89 + 0.1 * h.calls + 0.2 * h.puts, 0.0, 1e-9);
}
