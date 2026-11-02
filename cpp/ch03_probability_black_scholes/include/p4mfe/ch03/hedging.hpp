#pragma once
// Chapter 3, section 3.8: Delta hedging and Delta-Gamma hedging.
//
// For a small move dS in the spot over a short time dt, a portfolio with
// Greeks (Delta, Gamma, Theta) changes by approximately
//   dPi ~ Delta dS + 1/2 Gamma dS^2 + Theta dt.
// A Delta-neutral portfolio removes the first-order term; a Delta-Gamma
// neutral one also removes the second-order term, using a second option on
// the same asset because the asset itself has Delta = 1 and Gamma = 0.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <stdexcept>

namespace p4mfe::ch03 {

// Second-order approximation of the change in portfolio value.
[[nodiscard]] constexpr double PortfolioChangeApprox(double delta, double gamma,
                                                     double d_spot,
                                                     double theta = 0.0,
                                                     double dt = 0.0) noexcept {
  return delta * d_spot + 0.5 * gamma * d_spot * d_spot + theta * dt;
}

// Shares to trade so that a portfolio with Delta `portfolio_delta` becomes
// Delta-neutral: -portfolio_delta.
[[nodiscard]] constexpr double DeltaHedgeShares(
    double portfolio_delta) noexcept {
  return -portfolio_delta;
}

// A self-financing Delta-hedged position: `options` units of an option worth
// V (Delta `delta`), shares bought to make it Delta-neutral, and the cash
// balance that makes the initial value zero (borrowed if negative), which
// accrues at rate r.
struct HedgedPosition {
  double options;
  double shares;
  double cash;

  // Value after dt years, when the option is worth V_new and the spot is S_new.
  [[nodiscard]] double Value(double value_new, double spot_new, double r,
                             double dt) const {
    return options * value_new + shares * spot_new + cash * std::exp(r * dt);
  }
};

[[nodiscard]] inline HedgedPosition DeltaHedge(double options, double value,
                                               double delta, double spot) {
  const double shares = DeltaHedgeShares(options * delta);
  return {options, shares, -(options * value + shares * spot)};
}

// Delta-Gamma neutral hedge with a second option H (Delta delta_h, Gamma
// gamma_h) and the underlying asset:
//   units of H:  -Gamma_P / gamma_h
//   shares:      -(Delta_P + units_H * delta_h)
struct DeltaGammaHedge {
  double hedge_options;
  double shares;
};

[[nodiscard]] inline DeltaGammaHedge ComputeDeltaGammaHedge(
    double portfolio_delta, double portfolio_gamma, double delta_h,
    double gamma_h) {
  if (gamma_h == 0.0)
    throw std::invalid_argument(
        "delta_gamma_hedge: hedging option needs nonzero Gamma");
  const double units = -portfolio_gamma / gamma_h;
  return {units, -(portfolio_delta + units * delta_h)};
}

}  // namespace p4mfe::ch03
