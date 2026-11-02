#pragma once
// Exercises 12-13 (section 4.9): closed forms for the squared call payoff
// ((S_T - K)^+)^2 and the power call (S_T^alpha - K)^+ by risk-neutral
// pricing.

#include <cmath>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch04::ex {

// Both exercises use one fact. If X ~ N(m, s^2), then for any k and a
//   E[e^(kX) 1_{X > a}] = e^(k m + k^2 s^2 / 2) N((m - a)/s + k s)
// (complete the square: e^(kx) phi((x - m)/s) = e^(k m + k^2 s^2/2) phi((x - m
// - k s^2)/s)). k = 0 gives P(X > a) = N((m - a)/s). Under the risk-neutral
// measure ln S_T ~ N(ln S_0 + (r - q - sigma^2/2) T, sigma^2 T).

// ----------------------------------------------------------- Exercise 12 ---
// Price the option that pays (max(S(T) - K, 0))^2 at maturity:
//   V(0) = e^(-rT) E_RN[(max(S(T) - K, 0))^2],
// S lognormal with drift r and volatility sigma, no dividends (q = 0).
//
// Expand the square on {S_T > K}:
//   E[((S_T - K)^+)^2] = E[S_T^2 1] - 2K E[S_T 1] + K^2 P(S_T > K).
// With m = ln S_0 + (r - q - sigma^2/2) T, s = sigma sqrt(T) and
// d2 = (m - ln K)/s, d1 = d2 + s (the Black-Scholes d1, d2):
//   P(S_T > K)       = N(d2),
//   E[S_T 1_{S_T>K}]   = S_0 e^((r - q) T) N(d1),
//   E[S_T^2 1_{S_T>K}] = S_0^2 e^((2(r - q) + sigma^2) T) N(d1 + sigma sqrt(T))
// (k = 2: e^(2m + 2s^2) = S_0^2 e^((2(r - q) + sigma^2) T) and d2 + 2s = d1 +
// s). Hence, for q = 0,
//   V(0) = S_0^2 e^((r + sigma^2) T) N(d1 + sigma sqrt(T)) - 2 K S_0 N(d1) +
//   K^2 e^(-rT) N(d2),
// and in general
//   V(0) = S_0^2 e^((r - 2q + sigma^2) T) N(d1 + sigma sqrt(T)) - 2 K S_0
//   e^(-qT) N(d1) + K^2 e^(-rT) N(d2).
[[nodiscard]] inline double Ex12SquaredCall(double spot0, double strike,
                                            double maturity, double sigma,
                                            double r, double q = 0.0) {
  const double s = sigma * std::sqrt(maturity);
  const double d2 =
      (std::log(spot0 / strike) + (r - q - 0.5 * sigma * sigma) * maturity) / s;
  const double d1 = d2 + s;
  return spot0 * spot0 * std::exp((r - 2.0 * q + sigma * sigma) * maturity) *
             norm_cdf(d1 + s) -
         2.0 * strike * spot0 * std::exp(-q * maturity) * norm_cdf(d1) +
         strike * strike * std::exp(-r * maturity) * norm_cdf(d2);
}

// ----------------------------------------------------------- Exercise 13 ---
// Value of an option on a non-dividend-paying lognormal asset with payoff
// max((S(T))^alpha - K, 0), alpha > 0.
//
// Y = S_T^alpha is lognormal: ln Y ~ N(alpha m, alpha^2 sigma^2 T). Then
// E[(Y - K)^+] = E[Y 1_{Y>K}] - K P(Y > K), and the fact above with k = 1, 0
// applied to ln Y gives:
//   d2(alpha) = (alpha ln S_0 + alpha (r - q - sigma^2/2) T - ln K) / (alpha
//   sigma sqrt(T)), d1(alpha) = d2(alpha) + alpha sigma sqrt(T), V(0) = e^(-rT)
//   [S_0^alpha e^(alpha (r - q - sigma^2/2) T + alpha^2 sigma^2 T / 2)
//   N(d1(alpha)) - K N(d2(alpha))]
//        = S_0^alpha e^((alpha - 1) r T - alpha q T + alpha (alpha - 1) sigma^2
//        T / 2) N(d1(alpha)) - K e^(-rT) N(d2(alpha)).
// alpha = 1 gives the Black-Scholes call; the statement has q = 0. The
// d's need alpha > 0 (for alpha < 0, S^alpha decreases in S and the event
// flips).
[[nodiscard]] inline double Ex13PowerCall(double spot0, double strike,
                                          double maturity, double sigma,
                                          double r, double alpha,
                                          double q = 0.0) {
  const double s = alpha * sigma * std::sqrt(maturity);
  const double m =
      alpha * (std::log(spot0) + (r - q - 0.5 * sigma * sigma) * maturity);
  const double d2 = (m - std::log(strike)) / s;
  const double d1 = d2 + s;
  return std::exp(-r * maturity) *
         (std::exp(m + 0.5 * s * s) * norm_cdf(d1) - strike * norm_cdf(d2));
}

}  // namespace p4mfe::ch04::ex
