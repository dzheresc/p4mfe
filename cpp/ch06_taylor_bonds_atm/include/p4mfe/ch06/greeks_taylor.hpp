#pragma once
// Chapter 6, section 6.9: the Greeks and Taylor's formula.
//
// An option value V(S, t, sigma, r) is a function of several variables, so
// Taylor's formula (6.2) expresses its change through its partial
// derivatives, the Greeks:
//   dV ~ Delta dS + Theta dt + Vega dsigma + Rho dr + 1/2 Gamma dS^2.
// Only the second-order term in dS is kept: dS is of order sigma S sqrt(dt),
// so dS^2 is of the same order as dt, while the other second-order terms are
// smaller. Chapter 3's portfolio_change_approx is the (dS, dt) part of this.
//
// For a Delta-neutral portfolio of options and shares (no dividends), the
// Black-Scholes PDE Theta + 1/2 sigma^2 S^2 Gamma + r S Delta - r V = 0 lets
// Theta be eliminated:
//   dPi ~ r Pi dt + 1/2 Gamma (dS^2 - sigma^2 S^2 dt).
// The hedged position earns the risk-free rate plus a Gamma term that is
// positive when the realized squared move dS^2 exceeds the variance
// sigma^2 S^2 dt priced into the option, negative otherwise.

#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/hedging.hpp>

namespace p4mfe::ch06 {

using ch03::Greeks;

// First order in dS only: dV ~ Delta dS.
[[nodiscard]] constexpr double OptionChangeDelta(const Greeks& g,
                                                 double d_spot) noexcept {
  return g.delta * d_spot;
}

// Delta-Gamma: dV ~ Delta dS + 1/2 Gamma dS^2.
[[nodiscard]] constexpr double OptionChangeDeltaGamma(const Greeks& g,
                                                      double d_spot) noexcept {
  return ch03::PortfolioChangeApprox(g.delta, g.gamma, d_spot);
}

// Full approximation: Delta, Gamma, Theta, Vega and Rho terms.
[[nodiscard]] constexpr double OptionChangeTaylor(const Greeks& g,
                                                  double d_spot, double dt,
                                                  double dsigma = 0.0,
                                                  double dr = 0.0) noexcept {
  return ch03::PortfolioChangeApprox(g.delta, g.gamma, d_spot, g.theta, dt) +
         g.vega * dsigma + g.rho * dr;
}

// Theta implied by the Black-Scholes PDE:
//   Theta = r V - (r - q) S Delta - 1/2 sigma^2 S^2 Gamma.
[[nodiscard]] constexpr double ThetaFromPde(double value, double spot,
                                            double delta, double gamma,
                                            double sigma, double r,
                                            double q = 0.0) noexcept {
  return r * value - (r - q) * spot * delta -
         0.5 * sigma * sigma * spot * spot * gamma;
}

// Change of a Delta-neutral portfolio worth Pi with Gamma `gamma` (q = 0):
//   dPi ~ r Pi dt + 1/2 Gamma (dS^2 - sigma^2 S^2 dt).
[[nodiscard]] constexpr double DeltaNeutralChange(double portfolio_value,
                                                  double spot, double gamma,
                                                  double sigma, double r,
                                                  double d_spot,
                                                  double dt) noexcept {
  return r * portfolio_value * dt +
         0.5 * gamma * (d_spot * d_spot - sigma * sigma * spot * spot * dt);
}

}  // namespace p4mfe::ch06
