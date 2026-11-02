#pragma once
// Exercise 8 (section 6.11): the series 1 + x^4/2! + x^8/4! + ... and
// cosh(x^2).

#include <cmath>
#include <limits>
#include <p4mfe/ch06/taylor_series.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 8 ---
// (i) Find the radius of convergence of the series
//   1 + x^4/2! + x^8/4! + x^12/6! + ...                      (6.127)
// (ii) Show that it is the Taylor series expansion of (e^(x^2) + e^(-x^2))/2.
//
// (i) The terms are a_k = x^(4k)/(2k)!. For x != 0,
//   |a_{k+1}/a_k| = x^4 / ((2k+1)(2k+2)) -> 0 as k -> infinity,
// so by the ratio test the series converges for every x: R = infinity.
// (ii) With y = x^2, e^y = sum y^m/m! and e^(-y) = sum (-1)^m y^m/m!, both for
// all y. In the average (e^y + e^(-y))/2 the odd m cancel and the even m = 2k
// survive with coefficient 1:
//   (e^(x^2) + e^(-x^2))/2 = sum_{k>=0} y^(2k)/(2k)! = sum_{k>=0} x^(4k)/(2k)!,
// which is (6.127). It is cosh(x^2), and since the power series converges to
// the function everywhere, it is its Taylor series around 0.
//
// Partial sum with terms k = 0..n.
[[nodiscard]] inline double ex08_series(double x, unsigned n) noexcept {
  const double x4 = x * x * x * x;
  double term = 1.0, sum = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    term *= x4 / ((2.0 * k - 1.0) * (2.0 * k));
    sum += term;
  }
  return sum;
}
// The same, as the average of chapter 6's e^y partial sums of degree 2n.
[[nodiscard]] inline double ex08_series_from_exp(double x,
                                                 unsigned n) noexcept {
  return 0.5 * (exp_series(x * x, 2 * n) + exp_series(-x * x, 2 * n));
}
[[nodiscard]] inline double ex08_function(double x) { return std::cosh(x * x); }
inline constexpr double ex08_radius = std::numeric_limits<double>::infinity();

}  // namespace p4mfe::ch06::ex
