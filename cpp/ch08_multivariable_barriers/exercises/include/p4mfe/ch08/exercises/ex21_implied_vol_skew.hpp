#pragma once
// Exercise 21 (section 8.12): the strike derivative of the implied volatility.

#include <cmath>
#include <concepts>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 21 ---
// S, T, q, r are given and C(K) is the (known) price of the call with strike K,
// for every K. sigma_imp(K) is the solution of C(K) = C_BS(K, sigma_imp(K)).
// Find d sigma_imp/dK as a function of sigma_imp(K).
//
// Differentiate the identity C(K) = C_BS(K, sigma_imp(K)) with respect to K
// (chain rule in the two arguments of C_BS):
//   C'(K) = dC_BS/dK (K, sigma_imp(K)) + dC_BS/dsigma (K, sigma_imp(K)) *
//   sigma_imp'(K).
// With the Black-Scholes strike derivative and vega
//   dC_BS/dK = -e^(-rT) N(d2),   dC_BS/dsigma = S e^(-qT) sqrt(T) phi(d1),
// this gives the implicit ODE
//   sigma_imp'(K) = ( C'(K) + e^(-rT) N(d2) ) / ( S e^(-qT) sqrt(T) phi(d1) ),
// with d1, d2 evaluated at sigma = sigma_imp(K):
//   d1 = (ln(S/K) + (r - q + sigma_imp(K)^2/2) T)/(sigma_imp(K) sqrt(T)),  d2 =
//   d1 - sigma_imp(K) sqrt(T).
// The ODE is implicit in the sense that its right side depends on
// sigma_imp(K) (through d1, d2) and on the market data C'(K).
// A flat smile (sigma_imp' = 0) means C'(K) = -e^(-rT) N(d2), the Black-Scholes
// strike derivative; observed C'(K) above it gives an upward-sloping smile.
//
// Check: generate prices from a known smile sigma(K), compute C'(K) by central
// differences in K, and compare the right side with sigma'(K).
struct SmileMarket {
  double spot = 100.0, maturity = 0.75, r = 0.04, q = 0.01;
};

[[nodiscard]] inline double Ex21DsigmaDK(double strike, double sigma_imp,
                                         double d_call_d_strike,
                                         const SmileMarket& m = {}) {
  const auto [d1, d2] =
      ch03::BsD1D2(m.spot, strike, m.maturity, sigma_imp, m.r, m.q);
  return (d_call_d_strike + std::exp(-m.r * m.maturity) * norm_cdf(d2)) /
         (m.spot * std::exp(-m.q * m.maturity) * std::sqrt(m.maturity) *
          norm_pdf(d1));
}

// A smooth skewed smile for the checks: sigma(K) = 0.2 - 0.1 k + 0.15 k^2, k =
// ln(K/S).
[[nodiscard]] inline double Ex21Smile(double strike,
                                      const SmileMarket& m = {}) {
  const double k = std::log(strike / m.spot);
  return 0.2 - 0.1 * k + 0.15 * k * k;
}
[[nodiscard]] inline double Ex21SmileSlope(double strike,
                                           const SmileMarket& m = {}) {
  const double k = std::log(strike / m.spot);
  return (-0.1 + 0.3 * k) / strike;
}
[[nodiscard]] inline double Ex21Price(double strike,
                                      const SmileMarket& m = {}) {
  return ch03::BsCall(m.spot, strike, m.maturity, Ex21Smile(strike, m), m.r,
                      m.q);
}

// sigma_imp'(K) from prices only: implied vol by Newton (chapter 3) and C'(K)
// by central differences with step h K.
template <std::invocable<double> Price>
[[nodiscard]] double Ex21SlopeFromPrices(Price&& call, double strike,
                                         const SmileMarket& m = {},
                                         double h = 1e-4) {
  const double sigma =
      ch03::ImpliedVolNewton(ch03::OptionType::kCall, call(strike), m.spot,
                             strike, m.maturity, m.r, m.q, 1e-12)
          .sigma;
  const double d_strike = h * strike;
  const double d_call =
      (call(strike + d_strike) - call(strike - d_strike)) / (2.0 * d_strike);
  return Ex21DsigmaDK(strike, sigma, d_call, m);
}

}  // namespace p4mfe::ch08::ex
