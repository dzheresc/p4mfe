#pragma once
// Exercise 5 (section 6.11): Taylor series of ln(1 - x^2) and 1/(1 - x^2).

#include <p4mfe/ch06/taylor_series.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 5 ---
// Find the Taylor series expansions of ln(1 - x^2) and 1/(1 - x^2) around 0,
// using the expansions (6.48) and (6.49) of ln(1 - x) and 1/(1 - x).
//
//   ln(1 - y)  = -sum_{k>=1} y^k / k,   1/(1 - y) = sum_{k>=0} y^k,   |y| < 1.
// Substitute y = x^2 (|x| < 1 <=> |y| < 1):
//   ln(1 - x^2)  = -sum_{k>=1} x^(2k) / k = -x^2 - x^4/2 - x^6/3 - ...
//   1/(1 - x^2)  =  sum_{k>=0} x^(2k)     =  1 + x^2 + x^4 + ...
// both for |x| < 1. A power series is unique, so these are the Taylor series.
// (Check: d/dx ln(1 - x^2) = -2x/(1 - x^2) = -2 sum x^(2k+1), the derivative
// of the first series term by term.)
//
// Partial sums with terms k = 1..n and k = 0..n, reusing chapter 6's
// log1p_series (ln(1 + z) at z = -x^2) and geometric_series (at x^2).
[[nodiscard]] inline double ex05_log_series(double x, unsigned n) { return log1p_series(-x * x, n); }
[[nodiscard]] inline double ex05_inverse_series(double x, unsigned n) { return geometric_series(x * x, n); }

// Coefficient of x^m in each series (zero for odd m).
[[nodiscard]] constexpr double ex05_log_coefficient(unsigned m) noexcept {
    return (m == 0 || m % 2 != 0) ? 0.0 : -2.0 / static_cast<double>(m);
}
[[nodiscard]] constexpr double ex05_inverse_coefficient(unsigned m) noexcept { return m % 2 == 0 ? 1.0 : 0.0; }

}  // namespace p4mfe::ch06::ex
