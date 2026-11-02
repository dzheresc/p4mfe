#pragma once
// Chapter 3, sections 3.5-3.6.1: the Black-Scholes formula for European calls
// and puts on an asset with continuous dividend yield q, and the Greeks.
//
// Arguments follow the book's order: spot S, strike K, time to maturity T (in
// years), volatility sigma, risk-free rate r, dividend yield q; r and q are
// continuously compounded. Theta is dV/dt in calendar time t (so it is minus
// dV/dT) and is per year; vega and rho are per unit (not per 1%) change.

#include <cmath>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch03 {

using ch01::OptionType;

// ---------------------------------------------------------------- 3.5 ------
//   d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T)),  d2 = d1 - sigma
//   sqrt(T)
struct D1D2 {
  double d1;
  double d2;
};

[[nodiscard]] inline D1D2 BsD1D2(double spot, double strike, double maturity,
                                 double sigma, double r, double q = 0.0) {
  if (!(spot > 0.0 && strike > 0.0 && maturity > 0.0 && sigma > 0.0))
    throw std::invalid_argument("Black-Scholes: need S, K, T, sigma > 0");
  const double sd = sigma * std::sqrt(maturity);
  const double d1 =
      (std::log(spot / strike) + (r - q + 0.5 * sigma * sigma) * maturity) / sd;
  return {d1, d1 - sd};
}

//   C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
//   P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
[[nodiscard]] inline double BsPrice(OptionType type, double spot, double strike,
                                    double maturity, double sigma, double r,
                                    double q = 0.0) {
  const auto [d1, d2] = BsD1D2(spot, strike, maturity, sigma, r, q);
  const double fwd_spot = spot * std::exp(-q * maturity);
  const double disc_strike = strike * std::exp(-r * maturity);
  return type == OptionType::kCall
             ? fwd_spot * NormCdf(d1) - disc_strike * NormCdf(d2)
             : disc_strike * NormCdf(-d2) - fwd_spot * NormCdf(-d1);
}
[[nodiscard]] inline double BsCall(double spot, double strike, double maturity,
                                   double sigma, double r, double q = 0.0) {
  return BsPrice(OptionType::kCall, spot, strike, maturity, sigma, r, q);
}
[[nodiscard]] inline double BsPut(double spot, double strike, double maturity,
                                  double sigma, double r, double q = 0.0) {
  return BsPrice(OptionType::kPut, spot, strike, maturity, sigma, r, q);
}

// ---------------------------------------------------------------- 3.6 ------
struct Greeks {
  double delta;  // dV/dS
  double gamma;  // d2V/dS2
  double vega;   // dV/dsigma
  double theta;  // dV/dt
  double rho;    // dV/dr
};

//   Delta_C = e^(-qT) N(d1),            Delta_P = -e^(-qT) N(-d1)
//   Gamma   = e^(-qT) phi(d1) / (S sigma sqrt(T))                (same for C,
//   P) Vega    = S e^(-qT) phi(d1) sqrt(T)                          (same for
//   C, P) Theta_C = -S sigma e^(-qT) phi(d1) / (2 sqrt(T)) + q S e^(-qT) N(d1)
//   - r K e^(-rT) N(d2) Theta_P = -S sigma e^(-qT) phi(d1) / (2 sqrt(T)) - q S
//   e^(-qT) N(-d1) + r K e^(-rT) N(-d2) Rho_C   = K T e^(-rT) N(d2), Rho_P = -K
//   T e^(-rT) N(-d2)
[[nodiscard]] inline Greeks BsGreeks(OptionType type, double spot,
                                     double strike, double maturity,
                                     double sigma, double r, double q = 0.0) {
  const auto [d1, d2] = BsD1D2(spot, strike, maturity, sigma, r, q);
  const double eq = std::exp(-q * maturity);
  const double er = std::exp(-r * maturity);
  const double sqrt_maturity = std::sqrt(maturity);
  const double pdf1 = NormPdf(d1);

  Greeks g{};
  g.gamma = eq * pdf1 / (spot * sigma * sqrt_maturity);
  g.vega = spot * eq * pdf1 * sqrt_maturity;
  const double decay = -spot * sigma * eq * pdf1 / (2.0 * sqrt_maturity);
  if (type == OptionType::kCall) {
    g.delta = eq * NormCdf(d1);
    g.theta =
        decay + q * spot * eq * NormCdf(d1) - r * strike * er * NormCdf(d2);
    g.rho = strike * maturity * er * NormCdf(d2);
  } else {
    g.delta = -eq * NormCdf(-d1);
    g.theta =
        decay - q * spot * eq * NormCdf(-d1) + r * strike * er * NormCdf(-d2);
    g.rho = -strike * maturity * er * NormCdf(-d2);
  }
  return g;
}

// -------------------------------------------------------------- 3.6.1 ------
// The identity behind every Greek computation: differentiating N(d1) and
// N(d2) produces phi(d1) and phi(d2), and these are tied by
//   S e^(-qT) phi(d1) = K e^(-rT) phi(d2).
// This is why, e.g., dC/dS = e^(-qT) N(d1): the terms from d(d1)/dS and
// d(d2)/dS cancel. Returns the difference of the two sides (zero up to
// rounding).
[[nodiscard]] inline double BsPhiIdentityGap(double spot, double strike,
                                             double maturity, double sigma,
                                             double r, double q = 0.0) {
  const auto [d1, d2] = BsD1D2(spot, strike, maturity, sigma, r, q);
  return spot * std::exp(-q * maturity) * NormPdf(d1) -
         strike * std::exp(-r * maturity) * NormPdf(d2);
}

// The Greeks satisfy the Black-Scholes PDE
//   Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0.
// Returns the left-hand side (zero up to rounding).
[[nodiscard]] inline double BsPdeResidual(OptionType type, double spot,
                                          double strike, double maturity,
                                          double sigma, double r,
                                          double q = 0.0) {
  const auto g = BsGreeks(type, spot, strike, maturity, sigma, r, q);
  const double value = BsPrice(type, spot, strike, maturity, sigma, r, q);
  return g.theta + 0.5 * sigma * sigma * spot * spot * g.gamma +
         (r - q) * spot * g.delta - r * value;
}

}  // namespace p4mfe::ch03
