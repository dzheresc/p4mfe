#pragma once
// Exercise 15 (section 6.11): accuracy of the ATM put approximation (6.103)
// with rates and dividends, for maturities up to 20 years.

#include <array>
#include <cstddef>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/exercises/ex12_atm_put_error.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 15 ---
// ATM put, strike 40, volatility 30%, dividend yield 2%, r = 4.5%. Relative
// error of
//   P_approx = sigma S sqrt(T/(2 pi)) (1 - (r + q)T/2) - (r - q)T S/2
// for T = 1, 3, 5, 10, 20 years.
//
//   T     P_approx    P_BS        rel. error
//   1     4.131720    4.149140     0.420%
//   3     5.983403    6.157683     2.830%
//   5     6.465224    6.971412     7.261%
//   10    5.218687    7.339780    28.900%
//   20   -2.506679    6.059522   141.368%
// The linearization of e^(-rT), e^(-qT) breaks down as (r + q)T grows: at
// T = 20 the factor 1 - (r + q)T/2 is 0.35 and the "price" is negative. Only
// the 1-year value is usable.
inline constexpr std::array<double, 5> ex15_maturities{1.0, 3.0, 5.0, 10.0,
                                                       20.0};

[[nodiscard]] inline Ex12Result ex15_row(double T, double S = 40.0,
                                         double sigma = 0.3, double r = 0.045,
                                         double q = 0.02) {
  const double a = atm_put_approx(S, T, sigma, r, q);
  const double bs = ch03::bs_put(S, S, T, sigma, r, q);
  return {a, bs, ex12_relative_error(bs, a)};
}

[[nodiscard]] inline std::array<Ex12Result, 5> ex15_table() {
  std::array<Ex12Result, 5> out{};
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = ex15_row(ex15_maturities[i]);
  return out;
}

}  // namespace p4mfe::ch06::ex
