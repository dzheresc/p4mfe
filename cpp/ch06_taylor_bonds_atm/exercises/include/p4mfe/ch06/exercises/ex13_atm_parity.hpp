#pragma once
// Exercise 13 (section 6.11): ATM approximations that satisfy Put-Call parity.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/exercises/ex12_atm_put_error.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 13 ---
// The approximations (6.102)-(6.103) violate Put-Call parity:
//   P + S e^(-qT) - C = S (e^(-qT) - (r - q)T) != S e^(-rT) = K e^(-rT).
// Replacing rT and qT by 1 - e^(-rT) and 1 - e^(-qT) (from e^(-x) ~ 1 - x)
// gives
//   C ~ sigma S sqrt(T/(2 pi)) (e^(-qT) + e^(-rT))/2 + S (e^(-qT) - e^(-rT))/2,
//   (6.138) P ~ sigma S sqrt(T/(2 pi)) (e^(-qT) + e^(-rT))/2 - S (e^(-qT) -
//   e^(-rT))/2.   (6.139)
// (Indeed 1 - (r + q)T/2 -> (e^(-rT) + e^(-qT))/2 and (r - q)T -> e^(-qT) -
// e^(-rT).)
//
// (i) Subtracting, the first terms cancel:
//   C - P = S (e^(-qT) - e^(-rT)) = S e^(-qT) - K e^(-rT)   (K = S),
// which is Put-Call parity.
// (ii) S = 60, q = 0.03, sigma = 0.25, T = 1, r = 0.06:
//   P(6.139) = 5.984134 * 0.956105 - 60 * 0.028681 / 2 = 4.861031,
// P_BS = 4.886987, relative error 0.005311 (0.53%), against 0.014761 (1.48%)
// for (6.103) in exercise 12: the parity-consistent version is almost three
// times more accurate here.
[[nodiscard]] inline double ex13_call(double S, double T, double sigma,
                                      double r, double q) {
  const double eq = std::exp(-q * T), er = std::exp(-r * T);
  return atm_price_approx(S, T, sigma) * 0.5 * (eq + er) + 0.5 * S * (eq - er);
}
[[nodiscard]] inline double ex13_put(double S, double T, double sigma, double r,
                                     double q) {
  const double eq = std::exp(-q * T), er = std::exp(-r * T);
  return atm_price_approx(S, T, sigma) * 0.5 * (eq + er) - 0.5 * S * (eq - er);
}
// C - P - (S e^(-qT) - K e^(-rT)) with K = S: zero up to rounding for
// (6.138)-(6.139).
[[nodiscard]] inline double ex13_parity_gap(double S, double T, double sigma,
                                            double r, double q) {
  return ex13_call(S, T, sigma, r, q) - ex13_put(S, T, sigma, r, q) -
         S * (std::exp(-q * T) - std::exp(-r * T));
}
// The same gap for (6.102)-(6.103): S (e^(-rT) - 1 + (r - q)T + 1 - e^(-qT)) !=
// 0.
[[nodiscard]] inline double ex13_parity_gap_6103(double S, double T,
                                                 double sigma, double r,
                                                 double q) {
  return atm_call_approx(S, T, sigma, r, q) -
         atm_put_approx(S, T, sigma, r, q) -
         S * (std::exp(-q * T) - std::exp(-r * T));
}

[[nodiscard]] inline Ex12Result ex13_solve(const Ex12Market& m = {}) {
  const double a = ex13_put(m.S, m.T, m.sigma, m.r, m.q);
  const double bs = ch03::BsPut(m.S, m.S, m.T, m.sigma, m.r, m.q);
  return {a, bs, ex12_relative_error(bs, a)};
}

}  // namespace p4mfe::ch06::ex
