#pragma once
// Exercise 12 (section 6.11): ATM put approximations (r = q = 0 and with
// rates) against Black-Scholes.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 12 ---
// ATM put on a non-dividend-paying asset, S = 60, sigma = 0.25, T = 1.
// (i) What is P_approx for r = 0?
// (ii) Compute the Black-Scholes value P_BS and the relative error
//   |P_BS - P_approx| / P_BS.
// (iii) For r = 0.06 and q = 0.03, use (6.103) to compute P_approx and the
// relative error (6.137).
//
// (i) (6.102) with r = q = 0: P_approx = sigma S sqrt(T/(2 pi)) = 0.25 * 60 *
// 0.398942
//   = 5.984134.
// (ii) With r = q = 0 and K = S, d1 = -d2 = sigma sqrt(T)/2 = 0.125 and
//   P_BS = S (N(d1) - N(d2)) = 60 (2 N(0.125) - 1) = 5.968587,
// relative error 0.002605 (0.26%), just under the sigma^2 T/24 = 0.0026 of
// section 6.8.3 (which divides by P_approx instead).
// (iii) (6.103): P_approx = sigma S sqrt(T/(2 pi)) (1 - (r + q)T/2) - (r - q)T
// S/2
//   = 5.984134 * 0.955 - 0.9 = 4.814848,
// P_BS = 4.886987, relative error (6.137) = 0.014761 (1.48%): the rate terms
// make the approximation about six times worse here.
struct Ex12Market {
  double S = 60.0, sigma = 0.25, T = 1.0, r = 0.06, q = 0.03;
};
struct Ex12Result {
  double approx, black_scholes, relative_error;
};

[[nodiscard]] inline double ex12_relative_error(double exact, double approx) {
  return std::abs(exact - approx) / exact;
}
[[nodiscard]] inline Ex12Result ex12_zero_rates(const Ex12Market& m = {}) {
  const double a = atm_price_approx(m.S, m.T, m.sigma);
  const double bs = ch03::BsPut(m.S, m.S, m.T, m.sigma, 0.0, 0.0);
  return {a, bs, ex12_relative_error(bs, a)};
}
[[nodiscard]] inline Ex12Result ex12_with_rates(const Ex12Market& m = {}) {
  const double a = atm_put_approx(m.S, m.T, m.sigma, m.r, m.q);
  const double bs = ch03::BsPut(m.S, m.S, m.T, m.sigma, m.r, m.q);
  return {a, bs, ex12_relative_error(bs, a)};
}

}  // namespace p4mfe::ch06::ex
