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
[[nodiscard]] constexpr double option_change_delta(const Greeks& g,
                                                   double dS) noexcept {
  return g.delta * dS;
}

// Delta-Gamma: dV ~ Delta dS + 1/2 Gamma dS^2.
[[nodiscard]] constexpr double option_change_delta_gamma(const Greeks& g,
                                                         double dS) noexcept {
  return ch03::portfolio_change_approx(g.delta, g.gamma, dS);
}

// Full approximation: Delta, Gamma, Theta, Vega and Rho terms.
[[nodiscard]] constexpr double option_change_taylor(const Greeks& g, double dS,
                                                    double dt,
                                                    double dsigma = 0.0,
                                                    double dr = 0.0) noexcept {
  return ch03::portfolio_change_approx(g.delta, g.gamma, dS, g.theta, dt) +
         g.vega * dsigma + g.rho * dr;
}

// Theta implied by the Black-Scholes PDE:
//   Theta = r V - (r - q) S Delta - 1/2 sigma^2 S^2 Gamma.
[[nodiscard]] constexpr double theta_from_pde(double V, double S, double delta,
                                              double gamma, double sigma,
                                              double r,
                                              double q = 0.0) noexcept {
  return r * V - (r - q) * S * delta - 0.5 * sigma * sigma * S * S * gamma;
}

// Change of a Delta-neutral portfolio worth Pi with Gamma `gamma` (q = 0):
//   dPi ~ r Pi dt + 1/2 Gamma (dS^2 - sigma^2 S^2 dt).
[[nodiscard]] constexpr double delta_neutral_change(double Pi, double S,
                                                    double gamma, double sigma,
                                                    double r, double dS,
                                                    double dt) noexcept {
  return r * Pi * dt + 0.5 * gamma * (dS * dS - sigma * sigma * S * S * dt);
}

}  // namespace p4mfe::ch06
