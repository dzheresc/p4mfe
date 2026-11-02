#pragma once
// Exercise 1 (section 8.12): the Gamma of a call rises to a maximum and falls.

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/black_scholes.hpp>

namespace p4mfe::ch08::ex {

// ------------------------------------------------------------ Exercise 1 ---
// For t = 0 and q = 0, the Gamma (3.74) of a plain vanilla European call is
//   Gamma = 1/(S sigma sqrt(2 pi T)) exp(-d1^2/2),
//   d1 = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T)).
// Show that, as a function of S > 0, Gamma increases until it reaches a
// maximum point and then decreases, and that Gamma -> 0 as S -> 0 and as
// S -> infinity.
//
// Derivative. Gamma > 0, so look at
//   ln Gamma(S) = -ln S - d1^2/2 - ln(sigma sqrt(2 pi T)).
// Since d(d1)/dS = 1/(S sigma sqrt(T)),
//   d/dS ln Gamma = -1/S - d1/(S sigma sqrt(T)) = -(sigma sqrt(T) + d1) / (S
//   sigma sqrt(T)),
// and Gamma'(S) = Gamma(S) * d/dS ln Gamma has the sign of -(d1 + sigma
// sqrt(T)). d1 is strictly increasing in S, from -infinity to +infinity, so
// there is a unique S* with d1(S*) = -sigma sqrt(T):
//   ln(S*/K) + (r + sigma^2/2) T = -sigma^2 T   <=>   S* = K exp(-(r + 3
//   sigma^2/2) T).
// For S < S*, d1 < -sigma sqrt(T) and Gamma' > 0; for S > S*, Gamma' < 0. So
// Gamma increases on (0, S*], has its maximum at S*, and decreases on [S*,
// inf). The maximum is below the strike (S* < K for r > -3 sigma^2/2) and
//   Gamma(S*) = exp(-sigma^2 T / 2) / (S* sigma sqrt(2 pi T)).
//
// Limits. Write L = ln S, so d1 = (L - ln K + (r + sigma^2/2) T)/(sigma
// sqrt(T)) is affine in L with positive slope 1/(sigma sqrt(T)), and
//   ln Gamma = -L - d1^2/2 + const,
// a quadratic in L with negative leading coefficient -1/(2 sigma^2 T). Hence
// ln Gamma -> -infinity both as L -> -infinity (S -> 0) and as L -> +infinity
// (S -> infinity), i.e. Gamma -> 0 at both ends. (As S -> 0 the factor 1/S
// blows up, but exp(-d1^2/2) = exp(-(ln S)^2/(2 sigma^2 T) + ...) goes to 0
// faster than any power of S.) In the variable L, Gamma is a Gaussian bump
// times e^(-L), which is again a Gaussian bump: centered at ln S*.
struct Ex01Market {
  double strike = 50.0, maturity = 0.5, sigma = 0.3, r = 0.05;
};

[[nodiscard]] inline double Ex01Gamma(double spot, const Ex01Market& m = {}) {
  return ch03::BsGreeks(ch03::OptionType::kCall, spot, m.strike, m.maturity,
                        m.sigma, m.r, 0.0)
      .gamma;
}

// S* = K exp(-(r + 3 sigma^2/2) T), where d1 = -sigma sqrt(T).
[[nodiscard]] inline double Ex01Argmax(const Ex01Market& m = {}) {
  return m.strike * std::exp(-(m.r + 1.5 * m.sigma * m.sigma) * m.maturity);
}

[[nodiscard]] inline double Ex01MaxGamma(const Ex01Market& m = {}) {
  const double s = Ex01Argmax(m);
  return std::exp(-0.5 * m.sigma * m.sigma * m.maturity) /
         (s * m.sigma * std::sqrt(2.0 * std::numbers::pi * m.maturity));
}

// d/dS ln Gamma = -(sigma sqrt(T) + d1) / (S sigma sqrt(T)).
[[nodiscard]] inline double Ex01LogDerivative(double spot,
                                              const Ex01Market& m = {}) {
  const double st = m.sigma * std::sqrt(m.maturity);
  const double d1 =
      ch03::BsD1D2(spot, m.strike, m.maturity, m.sigma, m.r, 0.0).d1;
  return -(st + d1) / (spot * st);
}

}  // namespace p4mfe::ch08::ex
