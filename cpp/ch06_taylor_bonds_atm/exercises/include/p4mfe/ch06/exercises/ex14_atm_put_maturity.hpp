#pragma once
// Exercise 14 (section 6.11): accuracy of P ~ sigma S sqrt(T/(2 pi)) for long
// maturities (r = q = 0).

#include <array>
#include <cstddef>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/exercises/ex12_atm_put_error.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 14 ---
// ATM put, strike 40, non-dividend-paying asset, volatility 30%, zero rates.
// Relative error of P ~ sigma S sqrt(T/(2 pi)) for T = 3, 5, 10, 20 years.
//
// ATM means S = K = 40. With r = q = 0, P_BS = S (2 N(sigma sqrt(T)/2) - 1)
// and the relative error |P_BS - P_approx| / P_BS (as in exercise 12) is
//   T     P_approx    P_BS        rel. error
//   3     8.291860    8.199513    1.126%
//   5    10.704745   10.507373    1.878%
//   10   15.138795   14.589748    3.763%
//   20   21.409489   19.906602    7.550%
// The approximation always overestimates (section 6.8.3), and the error grows
// roughly like sigma^2 T / 24 = 0.375% per year: it is linear in T because
// it comes from the neglected (sigma sqrt(T))^2 term. The approximation is
// not reliable for long-dated options.
inline constexpr std::array<double, 4> ex14_maturities{3.0, 5.0, 10.0, 20.0};

[[nodiscard]] inline Ex12Result ex14_row(double T, double S = 40.0,
                                         double sigma = 0.3) {
  const double a = atm_price_approx(S, T, sigma);
  const double bs = ch03::bs_put(S, S, T, sigma, 0.0, 0.0);
  return {a, bs, ex12_relative_error(bs, a)};
}

[[nodiscard]] inline std::array<Ex12Result, 4> ex14_table() {
  std::array<Ex12Result, 4> out{};
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = ex14_row(ex14_maturities[i]);
  return out;
}

}  // namespace p4mfe::ch06::ex
