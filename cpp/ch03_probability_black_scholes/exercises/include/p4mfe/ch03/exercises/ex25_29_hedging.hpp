#pragma once
// Exercises 25-29 (section 3.11): Delta, Delta-Gamma and Delta-Gamma-vega
// neutral hedges, and the value of a Delta-hedged position as the spot moves.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/hedging.hpp>
#include <stdexcept>

namespace p4mfe::ch03::ex {

// ----------------------------------------------------------- Exercise 25 ---
// A portfolio of derivatives on one asset has Delta 5000 and Gamma -200. A
// call with Delta 0.4, Gamma 0.05 and a put with Delta -0.5, Gamma 0.07 are
// traded. How do you make the portfolio Delta-neutral and Gamma-neutral?
//
// With the two options only: buy x calls and y puts with
//   0.4 x - 0.5 y  = -5000      (Delta)
//   0.05 x + 0.07 y =  200      (Gamma)
// Solving (determinant 0.4*0.07 + 0.5*0.05 = 0.053):
//   x = (-5000*0.07 + 0.5*200)/0.053 = -4716.98,  y = (0.4*200 +
//   0.05*5000)/0.053 = 6226.42,
// i.e. sell 4717 calls and buy 6226 puts.
// If the asset can be traded too (Delta 1, Gamma 0), one option is enough:
//   buy 200/0.05 = 4000 calls (Delta +1600), then sell 6600 shares; or
//   buy 200/0.07 = 2857.14 puts (Delta -1428.57), then sell 3571.43 shares.
struct TwoOptionHedge {
  double calls;
  double puts;
};
// Units of two options that zero both the Delta and the Gamma.
[[nodiscard]] inline TwoOptionHedge ex25_options_only(double port_delta,
                                                      double port_gamma,
                                                      double dc, double gc,
                                                      double dp, double gp) {
  const double det = dc * gp - dp * gc;
  if (det == 0.0)
    throw std::invalid_argument(
        "ex25: the two options have proportional (Delta, Gamma)");
  return {(-port_delta * gp + dp * port_gamma) / det,
          (-dc * port_gamma + gc * port_delta) / det};
}

// ----------------------------------------------------------- Exercise 26 ---
// You are long 1000 calls with strike 90 and three months to maturity. The
// asset is lognormal with drift mu = 0.08 and volatility 0.2, spot 92, and
// r = 0.05. What Delta-hedging position do you need to take? (Delta does not
// depend on mu.)
//
// d1 = (ln(92/90) + (0.05 + 0.02) * 0.25) / (0.2 * 0.5) = 0.394789,
// Delta(C) = N(d1) = 0.653501, so short 1000 * 0.653501 = 653.5 shares
// (about 654).
[[nodiscard]] inline double ex26_shares() {
  return delta_hedge_shares(
      1000.0 * bs_greeks(OptionType::Call, 92.0, 90.0, 0.25, 0.2, 0.05).delta);
}

// ----------------------------------------------------------- Exercise 27 ---
// You buy 1000 six-month ATM calls on a non-dividend-paying asset with spot
// 100 and volatility 30%; r = 5%.
// (i)   How much do you pay for the options?
// (ii)  What Delta-hedging position do you have to take?
// (iii) On the next trading day the asset opens at 98. What is the value of
//       your position (the option and shares)?
// (iv)  Had you not Delta-hedged, how much would you have lost?
//
// (i)   C = 9.634877, so 1000 calls cost 9634.88.
// (ii)  Delta = N(d1) = 0.588589: short 588.59 shares, receiving 58858.91.
//       The net 49224.03 is put in the money market at 5%.
// (iii) One trading day later (dt = 1/252), with T = 0.5 - 1/252:
//       C = 8.453123, options worth 8453.12, short shares worth -57681.73,
//       so the option-and-shares position is -49228.61 (it was -49224.03,
//       a change of -4.58). With the cash, 49224.03 e^(0.05/252) = 49233.80,
//       the hedged portfolio is worth +5.19: Delta hedging has removed almost
//       all of the loss (what is left is the Gamma gain, 0.5 * Gamma * 2^2 per
//       option, less one day of Theta).
// (iv)  Unhedged, the options lose 1000 * (9.634877 - 8.453123) = 1181.75.
struct Ex27Result {
  double premium;          // 1000 C(0)
  double shares;           // Delta hedge (negative: short)
  double cash;             // money market balance after the trades
  double position_value;   // options + shares, next day
  double portfolio_value;  // options + shares + cash with interest, next day
  double unhedged_loss;    // 1000 (C(0) - C(next day))
};
[[nodiscard]] inline Ex27Result ex27(double dt = 1.0 / 252.0,
                                     double S1 = 98.0) {
  constexpr double n = 1000.0, S = 100.0, K = 100.0, T = 0.5, sigma = 0.3,
                   r = 0.05;
  const double C0 = bs_call(S, K, T, sigma, r);
  const auto pos = delta_hedge(
      n, C0, bs_greeks(OptionType::Call, S, K, T, sigma, r).delta, S);
  const double C1 = bs_call(S1, K, T - dt, sigma, r);
  return {n * C0,
          pos.shares,
          pos.cash,
          n * C1 + pos.shares * S1,
          pos.value(C1, S1, r, dt),
          n * (C0 - C1)};
}

// ----------------------------------------------------------- Exercise 28 ---
// Long 1000 puts (strike 25, six months) on a non-dividend-paying stock with
// volatility 30% and spot 20, long 400 shares, and 10000 in cash; r = 4%.
// (i)   How much is the portfolio worth?
// (ii)  How do you adjust the stock position to make it Delta-neutral?
// (iii) A month later the spot is 24. What is the new value of the portfolio,
//       and how do you adjust the stock position to make it Delta-neutral?
//
// (i)   P = 4.927351; value = 4927.35 + 400 * 20 + 10000 = 22927.35.
// (ii)  Delta(P) = N(d1) - 1 = -0.802771, so the portfolio Delta is
//       1000 * (-0.802771) + 400 = -402.77. Buy 402.77 shares (to 802.77),
//       paying 8055.43 from the cash, which falls to 1944.57.
// (iii) T = 5/12, S = 24: P = 2.181760 and Delta(P) = -0.511134. Value:
//       2181.76 + 802.77 * 24 + 1944.57 e^(0.04/12) = 23399.34
//       (23392.85 if the cash earns no interest). Delta-neutral now needs
//       511.13 shares: sell 291.64 shares.
struct Ex28Result {
  double value0;       // (i)
  double port_delta0;  // Delta before hedging
  double shares1;      // shares held after (ii)
  double cash1;        // cash after (ii)
  double value1;       // (iii), one month later before rebalancing
  double shares2;      // shares held after rebalancing in (iii)
};
[[nodiscard]] inline Ex28Result ex28() {
  constexpr double n = 1000.0, K = 25.0, T = 0.5, sigma = 0.3, r = 0.04,
                   S0 = 20.0, S1 = 24.0, dt = 1.0 / 12.0;
  constexpr double shares0 = 400.0, cash0 = 10000.0;
  const double P0 = bs_put(S0, K, T, sigma, r);
  const double D0 = bs_greeks(OptionType::Put, S0, K, T, sigma, r).delta;
  const double value0 = n * P0 + shares0 * S0 + cash0;
  const double port_delta0 = n * D0 + shares0;
  const double shares1 = shares0 + delta_hedge_shares(port_delta0);
  const double cash1 = cash0 - (shares1 - shares0) * S0;
  const double P1 = bs_put(S1, K, T - dt, sigma, r);
  const double D1 = bs_greeks(OptionType::Put, S1, K, T - dt, sigma, r).delta;
  const double value1 = n * P1 + shares1 * S1 + cash1 * std::exp(r * dt);
  return {value0, port_delta0, shares1, cash1, value1, -n * D1};
}

// ----------------------------------------------------------- Exercise 29 ---
// A portfolio has Delta 300, Gamma 100 and vega 89. You can trade the asset, a
// call with (Delta, Gamma, vega) = (0.2, 0.1, 0.1) and a put with
// (-0.8, 0.3, 0.2). What trades give a Delta-, Gamma- and vega-neutral
// portfolio?
//
// The asset has no Gamma or vega, so first neutralize those with x calls and
// y puts:
//   100 + 0.1 x + 0.3 y = 0     (Gamma)
//    89 + 0.1 x + 0.2 y = 0     (vega)
// Subtracting, 11 + 0.1 y = 0, so y = -110 and x = -670. Then Delta is
// 300 + 0.2 (-670) - 0.8 (-110) = 254, removed by z = -254 shares.
// Sell 670 calls, sell 110 puts and sell 254 shares.
struct Ex29Hedge {
  double calls;
  double puts;
  double shares;
};
[[nodiscard]] inline Ex29Hedge ex29_hedge(double pd = 300.0, double pg = 100.0,
                                          double pv = 89.0) {
  constexpr double cd = 0.2, cg = 0.1, cv = 0.1, qd = -0.8, qg = 0.3, qv = 0.2;
  const double det = cg * qv - qg * cv;
  const double x = (-pg * qv + qg * pv) / det;
  const double y = (-cg * pv + cv * pg) / det;
  return {x, y, -(pd + x * cd + y * qd)};
}

}  // namespace p4mfe::ch03::ex
