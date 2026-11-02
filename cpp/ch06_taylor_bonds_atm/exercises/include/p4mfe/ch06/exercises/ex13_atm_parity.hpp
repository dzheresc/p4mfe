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
[[nodiscard]] inline double Ex13Call(double spot, double maturity, double sigma,
                                     double r, double q) {
  const double eq = std::exp(-q * maturity), er = std::exp(-r * maturity);
  return AtmPriceApprox(spot, maturity, sigma) * 0.5 * (eq + er) +
         0.5 * spot * (eq - er);
}
[[nodiscard]] inline double Ex13Put(double spot, double maturity, double sigma,
                                    double r, double q) {
  const double eq = std::exp(-q * maturity), er = std::exp(-r * maturity);
  return AtmPriceApprox(spot, maturity, sigma) * 0.5 * (eq + er) -
         0.5 * spot * (eq - er);
}
// C - P - (S e^(-qT) - K e^(-rT)) with K = S: zero up to rounding for
// (6.138)-(6.139).
[[nodiscard]] inline double Ex13ParityGap(double spot, double maturity,
                                          double sigma, double r, double q) {
  return Ex13Call(spot, maturity, sigma, r, q) -
         Ex13Put(spot, maturity, sigma, r, q) -
         spot * (std::exp(-q * maturity) - std::exp(-r * maturity));
}
// The same gap for (6.102)-(6.103): S (e^(-rT) - 1 + (r - q)T + 1 - e^(-qT)) !=
// 0.
[[nodiscard]] inline double Ex13ParityGap6103(double spot, double maturity,
                                              double sigma, double r,
                                              double q) {
  return AtmCallApprox(spot, maturity, sigma, r, q) -
         AtmPutApprox(spot, maturity, sigma, r, q) -
         spot * (std::exp(-q * maturity) - std::exp(-r * maturity));
}

[[nodiscard]] inline Ex12Result Ex13Solve(const Ex12Market& m = {}) {
  const double a = Ex13Put(m.spot, m.maturity, m.sigma, m.r, m.q);
  const double bs = ch03::BsPut(m.spot, m.spot, m.maturity, m.sigma, m.r, m.q);
  return {a, bs, Ex12RelativeError(bs, a)};
}

}  // namespace p4mfe::ch06::ex
