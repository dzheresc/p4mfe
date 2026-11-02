#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/hedging.hpp>
#include <p4mfe/ch03/implied_vol.hpp>
#include <stdexcept>

using namespace p4mfe::ch03;

TEST(Ch03ImpliedVol, NewtonRecoversSigma) {
  const double S = 30, T = 0.5, r = 0.03, q = 0.01;
  for (OptionType type : {OptionType::Call, OptionType::Put}) {
    for (double K : {20.0, 28.0, 30.0 * std::exp((r - q) * T), 35.0, 45.0}) {
      for (double sigma : {0.1, 0.3, 0.8}) {
        const double price = bs_price(type, S, K, T, sigma, r, q);
        const auto res = implied_vol_newton(type, price, S, K, T, r, q);
        // sigma is only pinned down to ~(price rounding)/vega; vega is ~1e-7
        // for the deep in/out-of-the-money, low-vol cases.
        const double vega = bs_greeks(type, S, K, T, sigma, r, q).vega;
        EXPECT_NEAR(res.sigma, sigma, 1e-8 + 1e-12 / vega) << K << ' ' << sigma;
        EXPECT_NEAR(bs_price(type, S, K, T, res.sigma, r, q), price, 1e-12);
        EXPECT_LT(res.iterations,
                  30u);  // monotone, but slow for deep OTM, low-vol cases
      }
    }
  }
}

TEST(Ch03ImpliedVol, BisectionAgreesWithNewton) {
  const double S = 30, K = 33, T = 0.5, r = 0.03, q = 0.01, price = 2.5;
  const auto n = implied_vol_newton(OptionType::Call, price, S, K, T, r, q);
  const auto b = implied_vol_bisection(OptionType::Call, price, S, K, T, r, q);
  EXPECT_NEAR(n.sigma, b.sigma, 1e-9);
  EXPECT_GT(b.iterations, n.iterations);
}

TEST(Ch03ImpliedVol, RejectsArbitrageablePrices) {
  // A call cannot be worth more than S e^(-qT) or less than its forward
  // intrinsic value.
  EXPECT_THROW(
      (void)implied_vol_newton(OptionType::Call, 31.0, 30, 30, 0.5, 0.03),
      std::domain_error);
  EXPECT_THROW(
      (void)implied_vol_newton(OptionType::Call, 0.5, 40, 30, 0.5, 0.03),
      std::domain_error);
}

TEST(Ch03Hedging, DeltaHedgeIsSelfFinancingAndNeutral) {
  // Short 1000 calls, Delta-hedged with shares and cash.
  const double S = 50, K = 50, T = 0.25, sigma = 0.3, r = 0.05;
  const auto g = bs_greeks(OptionType::Call, S, K, T, sigma, r);
  const double V = bs_call(S, K, T, sigma, r);
  const auto pos = delta_hedge(-1000, V, g.delta, S);
  EXPECT_NEAR(pos.shares, 1000 * g.delta, 1e-12);
  EXPECT_NEAR(pos.value(V, S, r, 0.0), 0.0, 1e-9);

  // After a small move, the hedged P&L is second order: ~ -1000 (1/2 Gamma dS^2
  // + Theta dt) + r cash dt.
  const double dS = 0.5, dt = 1.0 / 252;
  const double V1 = bs_call(S + dS, K, T - dt, sigma, r);
  const double pnl = pos.value(V1, S + dS, r, dt);
  const double unhedged = -1000 * (V1 - V);
  const double approx =
      -1000 * portfolio_change_approx(0.0, g.gamma, dS, g.theta, dt) +
      pos.cash * r * dt;
  EXPECT_LT(std::abs(pnl), 0.1 * std::abs(unhedged));
  EXPECT_NEAR(pnl, approx, 0.05 * std::abs(approx) + 0.5);
}

TEST(Ch03Hedging, DeltaGammaNeutral) {
  const double S = 50, sigma = 0.3, r = 0.05;
  // Portfolio: short 1000 calls K = 50, T = 0.25. Hedge with calls K = 55, T =
  // 0.5.
  const auto gp = bs_greeks(OptionType::Call, S, 50, 0.25, sigma, r);
  const auto gh = bs_greeks(OptionType::Call, S, 55, 0.5, sigma, r);
  const double pd = -1000 * gp.delta, pg = -1000 * gp.gamma;
  const auto h = delta_gamma_hedge(pd, pg, gh.delta, gh.gamma);
  EXPECT_NEAR(pg + h.hedge_options * gh.gamma, 0.0, 1e-10);
  EXPECT_NEAR(pd + h.hedge_options * gh.delta + h.shares, 0.0, 1e-10);

  // A 2-dollar move: the Delta-Gamma hedged portfolio barely moves.
  const double dS = 2.0;
  const auto value = [&](double s) {
    return -1000 * bs_call(s, 50, 0.25, sigma, r) +
           h.hedge_options * bs_call(s, 55, 0.5, sigma, r) + h.shares * s;
  };
  const double dg_change = value(S + dS) - value(S);
  const double delta_only_change =
      -1000 * (bs_call(S + dS, 50, 0.25, sigma, r) -
               bs_call(S, 50, 0.25, sigma, r)) -
      pd * dS;
  EXPECT_LT(std::abs(dg_change), 0.1 * std::abs(delta_only_change));
  EXPECT_THROW((void)delta_gamma_hedge(1, 1, 0.5, 0.0), std::invalid_argument);
}
