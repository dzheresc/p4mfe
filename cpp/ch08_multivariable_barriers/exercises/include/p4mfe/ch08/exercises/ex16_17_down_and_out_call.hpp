#pragma once
// Exercises 16-17 (section 8.12): pricing a down-and-out call by (8.46), and
// its limit as the barrier goes to 0.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch08/barrier.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch08::ex {

// Formula (8.46), for B < K (and S > B):
//   V_DOC(S) = C(S) - (B/S)^(2(r - q)/sigma^2 - 1) C(B^2/S),
// where C(.) is the Black-Scholes call with strike K. The chapter library's
// down_and_out_call implements it (W - W*, section 8.9).

// ----------------------------------------------------------- Exercise 16 ---
// Six months down-and-out call, non-dividend-paying asset, sigma = 30%,
// S = 40, B = 35, K = 40, r = 5%.
//
//   p = 2r/sigma^2 - 1 = 0.111111,   B^2/S = 30.625,
//   C(40) = 3.853951,   (B/S)^p C(B^2/S) = 0.875^0.111111 * 0.461869 = 0.985273
//   * 0.461869 = 0.455067, V_DOC = 3.853951 - 0.455067 = 3.398883.
// The barrier knocks 0.455 (about 12%) off the vanilla value; the down-and-in
// call is worth 0.455067.
struct Ex16Market {
  double spot = 40.0, strike = 40.0, barrier = 35.0, maturity = 0.5,
         sigma = 0.3, r = 0.05, q = 0.0;
};

[[nodiscard]] inline double Ex16Formula846(const Ex16Market& m = {}) {
  const double p = 2.0 * (m.r - m.q) / (m.sigma * m.sigma) - 1.0;
  return ch03::BsCall(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q) -
         std::pow(m.barrier / m.spot, p) *
             ch03::BsCall(m.barrier * m.barrier / m.spot, m.strike, m.maturity,
                          m.sigma, m.r, m.q);
}

[[nodiscard]] inline double Ex16Value(const Ex16Market& m = {}) {
  return DownAndOutCall(m.spot, m.strike, m.barrier, m.maturity, m.sigma, m.r,
                        m.q);
}

// ----------------------------------------------------------- Exercise 17 ---
// Show that a down-and-out call with B < K converges to the plain vanilla
// call with strike K as B -> 0 (q = 0, r = 0).
//
// With r = q = 0 the exponent in (8.46) is -1, so
//   V(S) = C(S) - (S/B) C(B^2/S).
// By (7.41) with spot B^2/S, C(B^2/S) = (B^2/S) N(d1) - K N(d2), with
//   d1 = (ln(B^2/(S K)) + sigma^2 T/2)/(sigma sqrt(T)),  d2 = d1 - sigma
//   sqrt(T),
// hence
//   V(S) = C(S) - ( B N(d1) - (S K / B) N(d2) ).
// (The hint writes S K e^(-rT)/B; e^(-rT) = 1 here.)
// As B -> 0: B N(d1) -> 0 since 0 <= N <= 1. For the second term, d2 ->
// -infinity so N(d2) -> 0 and N(d2)/B is of the form 0/0. By l'Hopital, with
// d(d2)/dB = 2/(B sigma sqrt(T)),
//   lim N(d2)/B = lim N'(d2) * 2/(B sigma sqrt(T)) / 1 = (2/(sigma sqrt(T)))
//   lim phi(d2)/B.
// Now ln(phi(d2)/B) = -d2^2/2 - ln B - ln sqrt(2 pi), and d2 = 2 ln(B)/(sigma
// sqrt(T)) + O(1), so -d2^2/2 ~ -2 (ln B)^2/(sigma^2 T) dominates -ln B:
// ln(phi(d2)/B) -> -infinity and phi(d2)/B -> 0. Hence (1/B) N(d2) -> 0, the
// bracket -> 0, and V(S) -> C(S).
//
// The same holds for any r, q: (B/S)^p C(B^2/S) <= (B/S)^p (B^2/S) e^(-qT)
// N(d1) and N(d1) decays like phi(d1), faster than any power of B.
[[nodiscard]] inline double Ex17Bracket(double spot, double strike,
                                        double barrier, double maturity,
                                        double sigma) {
  const double st = sigma * std::sqrt(maturity);
  const double d1 = (std::log(barrier * barrier / (spot * strike)) +
                     0.5 * sigma * sigma * maturity) /
                    st;
  return barrier * norm_cdf(d1) - spot * strike / barrier * norm_cdf(d1 - st);
}

// (1/B) N(d2), the term the hint asks about.
[[nodiscard]] inline double Ex17ND2OverB(double spot, double strike,
                                         double barrier, double maturity,
                                         double sigma) {
  const double st = sigma * std::sqrt(maturity);
  const double d2 = (std::log(barrier * barrier / (spot * strike)) -
                     0.5 * sigma * sigma * maturity) /
                    st;
  return norm_cdf(d2) / barrier;
}

}  // namespace p4mfe::ch08::ex
