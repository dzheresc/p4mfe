#pragma once
// Exercises 13-14 (section 7.6): the call price and the forward contract solve
// the Black-Scholes PDE.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/bs_pde.hpp>

namespace p4mfe::ch07::ex {

// Market for the checks; tau = T - t, with T the maturity and t the calendar
// time.
struct PdeMarket {
  double strike = 40.0, maturity = 1.0, sigma = 0.3, r = 0.05, q = 0.02;
};

// ----------------------------------------------------------- Exercise 13 ---
// Show that C(S, t) = S e^(-q(T-t)) N(d1) - K e^(-r(T-t)) N(d2) satisfies
//   dC/dt + 1/2 sigma^2 S^2 d2C/dS2 + (r - q) S dC/dS - r C = 0.
//
// Use the PDE in Greek letters, Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta
// - r C = 0, with tau = T - t and the Greeks (3.72), (3.74), (3.78):
//   Delta = e^(-q tau) N(d1),
//   Gamma = e^(-q tau) phi(d1) / (S sigma sqrt(tau)),
//   Theta = -S sigma e^(-q tau) phi(d1) / (2 sqrt(tau)) + q S e^(-q tau) N(d1)
//   - r K e^(-r tau) N(d2).
// (Their derivation rests on S e^(-q tau) phi(d1) = K e^(-r tau) phi(d2), which
// makes the terms from d(d1)/dS, d(d2)/dS and d(d1)/dt, d(d2)/dt cancel.)
// Term by term:
//   Theta                       = -S sigma e^(-q tau) phi(d1)/(2 sqrt(tau)) + q
//   S e^(-q tau) N(d1) - r K e^(-r tau) N(d2) 1/2 sigma^2 S^2 Gamma       = +S
//   sigma e^(-q tau) phi(d1)/(2 sqrt(tau)) (r - q) S Delta             = (r -
//   q) S e^(-q tau) N(d1) -r C                        = -r S e^(-q tau) N(d1) +
//   r K e^(-r tau) N(d2)
// The phi(d1) terms cancel; the S e^(-q tau) N(d1) terms add to
// (q + r - q - r) = 0; the K e^(-r tau) N(d2) terms add to (-r + r) = 0. The
// sum is 0, so C solves the Black-Scholes PDE. (By put-call parity and exercise
// 14, so does the put.)
[[nodiscard]] inline double Ex13Call(double spot, double t,
                                     const PdeMarket& m = {}) {
  return ch03::BsCall(spot, m.strike, m.maturity - t, m.sigma, m.r, m.q);
}
// The PDE residual from the closed-form Greeks (zero up to rounding) ...
[[nodiscard]] inline double Ex13ResidualGreeks(double spot, double t,
                                               const PdeMarket& m = {}) {
  return ch03::BsPdeResidual(ch03::OptionType::kCall, spot, m.strike,
                             m.maturity - t, m.sigma, m.r, m.q);
}
// ... and from finite differences of the price (zero up to O(h^2)).
[[nodiscard]] inline double Ex13ResidualFd(double spot, double t,
                                           const PdeMarket& m = {}) {
  return BsPdeResidualFd([&](double s, double u) { return Ex13Call(s, u, m); },
                         spot, t, m.sigma, m.r, m.q);
}

// ----------------------------------------------------------- Exercise 14 ---
// The value of a forward contract struck at K and maturing at T, on an asset
// paying dividends continuously at rate q, is f(S, t) = S e^(-q(T-t)) - K
// e^(-r(T-t)). Show that f satisfies the Black-Scholes PDE.
//
//   df/dt   = q S e^(-q(T-t)) - r K e^(-r(T-t)),
//   df/dS   = e^(-q(T-t)),
//   d2f/dS2 = 0.
// Then
//   df/dt + 1/2 sigma^2 S^2 * 0 + (r - q) S e^(-q(T-t)) - r (S e^(-q(T-t)) - K
//   e^(-r(T-t))) = S e^(-q(T-t)) (q + r - q - r) + K e^(-r(T-t)) (-r + r) = 0.
// The PDE holds for every sigma: f is linear in S, so it has no Gamma.
// Note that f = C - P (put-call parity), so with exercise 13 the put also
// solves the PDE (the PDE is linear).
[[nodiscard]] inline double Ex14Forward(double spot, double t,
                                        const PdeMarket& m = {}) {
  return ForwardContractValue(spot, t, m.strike, m.maturity, m.r, m.q);
}
// Residual with the exact partial derivatives above (zero up to rounding).
[[nodiscard]] inline double Ex14ResidualExact(double spot, double t,
                                              const PdeMarket& m = {}) {
  const double tau = m.maturity - t;
  const double eq = std::exp(-m.q * tau), er = std::exp(-m.r * tau);
  const double f_t = m.q * spot * eq - m.r * m.strike * er;
  const double f_spot = eq;
  return f_t + (m.r - m.q) * spot * f_spot - m.r * Ex14Forward(spot, t, m);
}
[[nodiscard]] inline double Ex14ResidualFd(double spot, double t,
                                           const PdeMarket& m = {}) {
  return BsPdeResidualFd(
      [&](double s, double u) { return Ex14Forward(s, u, m); }, spot, t,
      m.sigma, m.r, m.q);
}

}  // namespace p4mfe::ch07::ex
