#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/hedging.hpp>
#include <p4mfe/ch03/implied_vol.hpp>
#include <stdexcept>

using namespace p4mfe::ch03;

TEST(Ch03ImpliedVol, NewtonRecoversSigma) {
  const double spot = 30, maturity = 0.5, r = 0.03, q = 0.01;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    for (double strike :
         {20.0, 28.0, 30.0 * std::exp((r - q) * maturity), 35.0, 45.0}) {
      for (double sigma : {0.1, 0.3, 0.8}) {
        const double price = BsPrice(type, spot, strike, maturity, sigma, r, q);
        const auto res =
            ImpliedVolNewton(type, price, spot, strike, maturity, r, q);
        // sigma is only pinned down to ~(price rounding)/vega; vega is ~1e-7
        // for the deep in/out-of-the-money, low-vol cases.
        const double vega =
            BsGreeks(type, spot, strike, maturity, sigma, r, q).vega;
        EXPECT_NEAR(res.sigma, sigma, 1e-8 + 1e-12 / vega)
            << strike << ' ' << sigma;
        EXPECT_NEAR(BsPrice(type, spot, strike, maturity, res.sigma, r, q),
                    price, 1e-12);
        EXPECT_LT(res.iterations,
                  30u);  // monotone, but slow for deep OTM, low-vol cases
      }
    }
  }
}

TEST(Ch03ImpliedVol, BisectionAgreesWithNewton) {
  const double spot = 30, strike = 33, maturity = 0.5, r = 0.03, q = 0.01,
               price = 2.5;
  const auto n =
      ImpliedVolNewton(OptionType::kCall, price, spot, strike, maturity, r, q);
  const auto b = ImpliedVolBisection(OptionType::kCall, price, spot, strike,
                                     maturity, r, q);
  EXPECT_NEAR(n.sigma, b.sigma, 1e-9);
  EXPECT_GT(b.iterations, n.iterations);
}

TEST(Ch03ImpliedVol, RejectsArbitrageablePrices) {
  // A call cannot be worth more than S e^(-qT) or less than its forward
  // intrinsic value.
  EXPECT_THROW(
      (void)ImpliedVolNewton(OptionType::kCall, 31.0, 30, 30, 0.5, 0.03),
      std::domain_error);
  EXPECT_THROW(
      (void)ImpliedVolNewton(OptionType::kCall, 0.5, 40, 30, 0.5, 0.03),
      std::domain_error);
}

TEST(Ch03Hedging, DeltaHedgeIsSelfFinancingAndNeutral) {
  // Short 1000 calls, Delta-hedged with shares and cash.
  const double spot = 50, strike = 50, maturity = 0.25, sigma = 0.3, r = 0.05;
  const auto g = BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r);
  const double value = BsCall(spot, strike, maturity, sigma, r);
  const auto pos = DeltaHedge(-1000, value, g.delta, spot);
  EXPECT_NEAR(pos.shares, 1000 * g.delta, 1e-12);
  EXPECT_NEAR(pos.Value(value, spot, r, 0.0), 0.0, 1e-9);

  // After a small move, the hedged P&L is second order: ~ -1000 (1/2 Gamma dS^2
  // + Theta dt) + r cash dt.
  const double d_spot = 0.5, dt = 1.0 / 252;
  const double value1 = BsCall(spot + d_spot, strike, maturity - dt, sigma, r);
  const double pnl = pos.Value(value1, spot + d_spot, r, dt);
  const double unhedged = -1000 * (value1 - value);
  const double approx =
      -1000 * PortfolioChangeApprox(0.0, g.gamma, d_spot, g.theta, dt) +
      pos.cash * r * dt;
  EXPECT_LT(std::abs(pnl), 0.1 * std::abs(unhedged));
  EXPECT_NEAR(pnl, approx, 0.05 * std::abs(approx) + 0.5);
}

TEST(Ch03Hedging, DeltaGammaNeutral) {
  const double spot = 50, sigma = 0.3, r = 0.05;
  // Portfolio: short 1000 calls K = 50, T = 0.25. Hedge with calls K = 55, T =
  // 0.5.
  const auto gp = BsGreeks(OptionType::kCall, spot, 50, 0.25, sigma, r);
  const auto gh = BsGreeks(OptionType::kCall, spot, 55, 0.5, sigma, r);
  const double pd = -1000 * gp.delta, pg = -1000 * gp.gamma;
  const auto h = ComputeDeltaGammaHedge(pd, pg, gh.delta, gh.gamma);
  EXPECT_NEAR(pg + h.hedge_options * gh.gamma, 0.0, 1e-10);
  EXPECT_NEAR(pd + h.hedge_options * gh.delta + h.shares, 0.0, 1e-10);

  // A 2-dollar move: the Delta-Gamma hedged portfolio barely moves.
  const double d_spot = 2.0;
  const auto value = [&](double s) {
    return -1000 * BsCall(s, 50, 0.25, sigma, r) +
           h.hedge_options * BsCall(s, 55, 0.5, sigma, r) + h.shares * s;
  };
  const double dg_change = value(spot + d_spot) - value(spot);
  const double delta_only_change =
      -1000 * (BsCall(spot + d_spot, 50, 0.25, sigma, r) -
               BsCall(spot, 50, 0.25, sigma, r)) -
      pd * d_spot;
  EXPECT_LT(std::abs(dg_change), 0.1 * std::abs(delta_only_change));
  EXPECT_THROW((void)ComputeDeltaGammaHedge(1, 1, 0.5, 0.0),
               std::invalid_argument);
}
