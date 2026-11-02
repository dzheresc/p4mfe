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

[[nodiscard]] inline D1D2 bs_d1_d2(double S, double K, double T, double sigma,
                                   double r, double q = 0.0) {
  if (!(S > 0.0 && K > 0.0 && T > 0.0 && sigma > 0.0))
    throw std::invalid_argument("Black-Scholes: need S, K, T, sigma > 0");
  const double sd = sigma * std::sqrt(T);
  const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd;
  return {d1, d1 - sd};
}

//   C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
//   P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
[[nodiscard]] inline double bs_price(OptionType type, double S, double K,
                                     double T, double sigma, double r,
                                     double q = 0.0) {
  const auto [d1, d2] = bs_d1_d2(S, K, T, sigma, r, q);
  const double fwd_S = S * std::exp(-q * T);
  const double disc_K = K * std::exp(-r * T);
  return type == OptionType::kCall
             ? fwd_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
             : disc_K * norm_cdf(-d2) - fwd_S * norm_cdf(-d1);
}
[[nodiscard]] inline double bs_call(double S, double K, double T, double sigma,
                                    double r, double q = 0.0) {
  return bs_price(OptionType::kCall, S, K, T, sigma, r, q);
}
[[nodiscard]] inline double bs_put(double S, double K, double T, double sigma,
                                   double r, double q = 0.0) {
  return bs_price(OptionType::kPut, S, K, T, sigma, r, q);
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
[[nodiscard]] inline Greeks bs_greeks(OptionType type, double S, double K,
                                      double T, double sigma, double r,
                                      double q = 0.0) {
  const auto [d1, d2] = bs_d1_d2(S, K, T, sigma, r, q);
  const double eq = std::exp(-q * T);
  const double er = std::exp(-r * T);
  const double sqrtT = std::sqrt(T);
  const double pdf1 = norm_pdf(d1);

  Greeks g{};
  g.gamma = eq * pdf1 / (S * sigma * sqrtT);
  g.vega = S * eq * pdf1 * sqrtT;
  const double decay = -S * sigma * eq * pdf1 / (2.0 * sqrtT);
  if (type == OptionType::kCall) {
    g.delta = eq * norm_cdf(d1);
    g.theta = decay + q * S * eq * norm_cdf(d1) - r * K * er * norm_cdf(d2);
    g.rho = K * T * er * norm_cdf(d2);
  } else {
    g.delta = -eq * norm_cdf(-d1);
    g.theta = decay - q * S * eq * norm_cdf(-d1) + r * K * er * norm_cdf(-d2);
    g.rho = -K * T * er * norm_cdf(-d2);
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
[[nodiscard]] inline double bs_phi_identity_gap(double S, double K, double T,
                                                double sigma, double r,
                                                double q = 0.0) {
  const auto [d1, d2] = bs_d1_d2(S, K, T, sigma, r, q);
  return S * std::exp(-q * T) * norm_pdf(d1) -
         K * std::exp(-r * T) * norm_pdf(d2);
}

// The Greeks satisfy the Black-Scholes PDE
//   Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0.
// Returns the left-hand side (zero up to rounding).
[[nodiscard]] inline double bs_pde_residual(OptionType type, double S, double K,
                                            double T, double sigma, double r,
                                            double q = 0.0) {
  const auto g = bs_greeks(type, S, K, T, sigma, r, q);
  const double V = bs_price(type, S, K, T, sigma, r, q);
  return g.theta + 0.5 * sigma * sigma * S * S * g.gamma +
         (r - q) * S * g.delta - r * V;
}

}  // namespace p4mfe::ch03
