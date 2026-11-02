#pragma once
// Exercises 16-21 (section 3.11): volga and vanna, ATM call versus ATM put,
// the sign of Theta, derivatives with respect to the strike, convexity in the
// strike, and the strike that minimizes a straddle.
//
// Notation: tau = T - t is the time to maturity (written T below, as in
// black_scholes.hpp), phi(x) = e^(-x^2/2)/sqrt(2 pi), and
//   d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T)),  d2 = d1 - sigma
//   sqrt(T), d(d1)/dsigma = -d2/sigma,  d(d2)/dsigma = -d1/sigma, d(d1)/dS =
//   d(d2)/dS = 1/(S sigma sqrt(T)),  d(d1)/dK = d(d2)/dK = -1/(K sigma
//   sqrt(T)), phi'(x) = -x phi(x),   S e^(-qT) phi(d1) = K e^(-rT) phi(d2).

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch03::ex {

// ----------------------------------------------------------- Exercise 16 ---
// volga(V) = d(vega)/dsigma = d2V/dsigma2 and vanna(V) = d(vega)/dS =
// d2V/dS dsigma = d(Delta)/dsigma.
// (i)  Compute the volga and vanna of a European call on an asset paying
//      dividends continuously at rate q.
// (ii) Use the Put-Call parity to compute them for a European put.
//
// (i) vega(C) = S e^(-qT) sqrt(T) phi(d1).
//   volga = S e^(-qT) sqrt(T) phi'(d1) d(d1)/dsigma = S e^(-qT) sqrt(T) (-d1
//   phi(d1)) (-d2/sigma)
//         = S e^(-qT) sqrt(T) phi(d1) d1 d2 / sigma = vega * d1 d2 / sigma.
//   vanna = d/dS [S phi(d1)] e^(-qT) sqrt(T)
//         = e^(-qT) sqrt(T) (phi(d1) - S d1 phi(d1) / (S sigma sqrt(T)))
//         = e^(-qT) phi(d1) (sigma sqrt(T) - d1) / sigma = -e^(-qT) phi(d1) d2
//         / sigma.
//   (Or: vanna = d/dsigma [e^(-qT) N(d1)] = e^(-qT) phi(d1) (-d2/sigma).)
// (ii) C - P = S e^(-qT) - K e^(-rT) does not depend on sigma, so
//   vega(P) = vega(C), volga(P) = volga(C), vanna(P) = vanna(C).
[[nodiscard]] inline double Ex16Volga(double spot, double strike,
                                      double maturity, double sigma, double r,
                                      double q = 0.0) {
  const auto [d1, d2] = BsD1D2(spot, strike, maturity, sigma, r, q);
  return spot * std::exp(-q * maturity) * std::sqrt(maturity) * norm_pdf(d1) *
         d1 * d2 / sigma;
}
[[nodiscard]] inline double Ex16Vanna(double spot, double strike,
                                      double maturity, double sigma, double r,
                                      double q = 0.0) {
  const auto [d1, d2] = BsD1D2(spot, strike, maturity, sigma, r, q);
  return -std::exp(-q * maturity) * norm_pdf(d1) * d2 / sigma;
}

// ----------------------------------------------------------- Exercise 17 ---
// Show that an ATM call on an asset paying dividends at rate q is worth more
// than an ATM put with the same maturity iff q <= r. Use the Put-Call parity,
// then the Black-Scholes formula.
//
// Parity with K = S:  C - P = S e^(-qT) - S e^(-rT) = S (e^(-qT) - e^(-rT)),
// which is >= 0 iff e^(-qT) >= e^(-rT) iff q <= r (strictly: C > P iff q < r,
// and C = P when q = r).
// Black-Scholes with K = S: using N(x) + N(-x) = 1,
//   C - P = S e^(-qT) (N(d1) + N(-d1)) - S e^(-rT) (N(d2) + N(-d2))
//         = S (e^(-qT) - e^(-rT)),
// the same, so the conclusion holds within the model too.
[[nodiscard]] inline double Ex17AtmCallMinusPut(double spot, double maturity,
                                                double sigma, double r,
                                                double q) {
  return BsCall(spot, spot, maturity, sigma, r, q) -
         BsPut(spot, spot, maturity, sigma, r, q);
}

// ----------------------------------------------------------- Exercise 18 ---
// (i)  Show that the Theta of a European call on a non-dividend-paying asset
//      is always negative.
// (ii) Show that the Theta of long dated ATM calls on an asset paying
//      dividends at rate q = r may be positive.
//
// (i) With q = 0,
//   Theta(C) = -S sigma phi(d1) / (2 sqrt(T)) - r K e^(-rT) N(d2).
// Both terms are negative (r >= 0), so Theta(C) < 0.
// (ii) With q = r and K = S, d1 = sigma sqrt(T)/2 = -d2, and
//   Theta(C) = -S sigma e^(-rT) phi(d1) / (2 sqrt(T)) + r S e^(-rT) N(d1) - r S
//   e^(-rT) N(d2)
//            = S e^(-rT) [ r (2 N(sigma sqrt(T)/2) - 1) - sigma phi(sigma
//            sqrt(T)/2) / (2 sqrt(T)) ].
// As T -> inf the first term in brackets tends to r > 0 while the second
// tends to 0, so Theta(C) > 0 for T large enough. For sigma = 0.2,
// r = q = 0.05 the sign changes at T* = 9.68 years (e.g. Theta = 0.0248 per
// year at T = 10 with S = 100).
[[nodiscard]] inline double Ex18ThetaCall(double spot, double strike,
                                          double maturity, double sigma,
                                          double r, double q = 0.0) {
  return BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r, q).theta;
}
// The bracket in (ii): Theta(C) / (S e^(-rT)) for an ATM call with q = r.
[[nodiscard]] inline double Ex18AtmQEqRBracket(double maturity, double sigma,
                                               double r) {
  const double x = 0.5 * sigma * std::sqrt(maturity);
  return r * (2.0 * norm_cdf(x) - 1.0) -
         sigma * norm_pdf(x) / (2.0 * std::sqrt(maturity));
}
// The maturity where that Theta changes sign, by bisection on the bracket
// (which is negative for small T and positive for large T).
[[nodiscard]] inline double Ex18ThetaSignChange(double sigma, double r,
                                                double lo = 1e-6,
                                                double hi = 1e3) {
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    (Ex18AtmQEqRBracket(mid, sigma, r) < 0.0 ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

// ---------------------------------------------------------- Exercises 19-20 -
// 19. Compute dC/dK and d2C/dK2; then use Put-Call parity for dP/dK, d2P/dK2.
// 20. Show that the call price is convex in the strike, d2C/dK2 >= 0.
//     Hint: by Lemma 3.15, dC/dK = -e^(-r(T-t)) N(d2).
//
// C = S e^(-qT) N(d1) - K e^(-rT) N(d2). Differentiating in K,
//   dC/dK = S e^(-qT) phi(d1) d(d1)/dK - e^(-rT) N(d2) - K e^(-rT) phi(d2)
//   d(d2)/dK
//         = -e^(-rT) N(d2) - (S e^(-qT) phi(d1) - K e^(-rT) phi(d2)) / (K sigma
//         sqrt(T)) = -e^(-rT) N(d2),
// the bracket vanishing by the identity S e^(-qT) phi(d1) = K e^(-rT) phi(d2).
// Then
//   d2C/dK2 = -e^(-rT) phi(d2) d(d2)/dK = e^(-rT) phi(d2) / (K sigma sqrt(T)) >
//   0,
// which is exercise 20 (it equals S e^(-qT) phi(d1) / (K^2 sigma sqrt(T))).
// Parity P = C - S e^(-qT) + K e^(-rT):
//   dP/dK = -e^(-rT) N(d2) + e^(-rT) = e^(-rT) N(-d2),
//   d2P/dK2 = d2C/dK2 = e^(-rT) phi(d2) / (K sigma sqrt(T)).
struct StrikeDerivatives {
  double d_call_d_strike;
  double d2_call_d_strike2;
  double d_put_d_strike;
  double d2_put_d_strike2;
};
[[nodiscard]] inline StrikeDerivatives Ex19StrikeDerivatives(
    double spot, double strike, double maturity, double sigma, double r,
    double q = 0.0) {
  const double d2 = BsD1D2(spot, strike, maturity, sigma, r, q).d2;
  const double er = std::exp(-r * maturity);
  const double second =
      er * norm_pdf(d2) / (strike * sigma * std::sqrt(maturity));
  return {-er * norm_cdf(d2), second, er * norm_cdf(-d2), second};
}

// ----------------------------------------------------------- Exercise 21 ---
// In the Black-Scholes world, for what strike is the value of a straddle
// P(K) + C(K) minimal, given S, T, q, r?
//
// f(K) = C(K) + P(K). From exercise 19,
//   f'(K)  = -e^(-rT) N(d2) + e^(-rT) N(-d2) = e^(-rT) (1 - 2 N(d2)),
//   f''(K) = 2 e^(-rT) phi(d2) / (K sigma sqrt(T)) > 0,
// so f is strictly convex and its minimum is where N(d2) = 1/2, i.e. d2 = 0:
//   ln(S/K) + (r - q - sigma^2/2) T = 0   =>   K* = S e^((r - q - sigma^2/2)
//   T),
// the forward price S e^((r-q)T) times e^(-sigma^2 T/2). (It is the median of
// S_T under the risk-neutral measure: N(d2) = P(S_T > K).)
[[nodiscard]] inline double Ex21MinStraddleStrike(double spot, double maturity,
                                                  double sigma, double r,
                                                  double q = 0.0) {
  return spot * std::exp((r - q - 0.5 * sigma * sigma) * maturity);
}
[[nodiscard]] inline double Ex21Straddle(double spot, double strike,
                                         double maturity, double sigma,
                                         double r, double q = 0.0) {
  return BsCall(spot, strike, maturity, sigma, r, q) +
         BsPut(spot, strike, maturity, sigma, r, q);
}

}  // namespace p4mfe::ch03::ex
