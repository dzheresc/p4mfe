#pragma once
// Exercises 7-10 (section 1.12): inequalities and limits around e, Taylor
// remainders by l'Hopital, and limits at infinity.
//
// Naive floating-point evaluation of these expressions suffers catastrophic
// cancellation (sqrt(1+x) - 1 - x/2 at x = 1e-8 is pure noise). Where it
// matters, the functions below use the algebraically rearranged, stable form
// that the analytic solution produces anyway.

#include <cmath>
#include <limits>

namespace p4mfe::ch01::ex {

// ------------------------------------------------------------ Exercise 7 ---
// Show that (1 + 1/x)^x < e < (1 + 1/x)^(x+1) for all x >= 1.
//
// Take logs. The claim is  x ln(1 + 1/x) < 1 < (x + 1) ln(1 + 1/x), i.e.
//   1/(x + 1) < ln(1 + 1/x) < 1/x.
// But ln(1 + 1/x) = ln(x + 1) - ln(x) = integral_x^(x+1) dt/t, and on
// (x, x+1) the integrand 1/t lies strictly between 1/(x+1) and 1/x; the
// interval has length 1. (The argument works for every x > 0.)
//
// Returns the three quantities, lower < e < upper.
struct Ex07Bounds {
  double lower;  // (1 + 1/x)^x
  double upper;  // (1 + 1/x)^(x+1)
};
[[nodiscard]] inline Ex07Bounds ex07_e_bounds(double x) {
  const double l = std::log1p(1.0 / x);  // ln(1 + 1/x), accurate for large x
  return {std::exp(x * l), std::exp((x + 1.0) * l)};
}

// ------------------------------------------------------------ Exercise 8 ---
// Use l'Hopital's rule to show that, for x close to 0,
//   sqrt(1 + x) ~ 1 + x/2      and      e^x ~ 1 + x + x^2/2,
// i.e. that the following limits exist and are constant:
//   lim_{x->0} (sqrt(1+x) - (1 + x/2)) / x^2,
//   lim_{x->0} (e^x - (1 + x + x^2/2)) / x^3.
//
// First limit: 0/0; differentiate twice:
//   (1/2 (1+x)^(-1/2) - 1/2) / (2x)  ->  0/0 again,
//   (-1/4 (1+x)^(-3/2)) / 2  ->  -1/8.
// Second limit: three applications give (e^x - 1 - x)/(3x^2), (e^x - 1)/(6x),
// e^x / 6  ->  1/6.
inline constexpr double ex08_sqrt_limit = -1.0 / 8.0;
inline constexpr double ex08_exp_limit = 1.0 / 6.0;

// The quotients themselves, evaluated stably for small x:
//   sqrt(1+x) - 1 - x/2 = -x^2 / (2 (sqrt(1+x) + 1)^2),
// which follows from sqrt(1+x) - 1 = x / (sqrt(1+x) + 1) twice. So the first
// quotient is exactly -1 / (2 (sqrt(1+x) + 1)^2), which -> -1/8.
[[nodiscard]] inline double ex08_sqrt_quotient(double x) {
  const double s = std::sqrt(1.0 + x) + 1.0;
  return -1.0 / (2.0 * s * s);
}
// e^x - 1 - x - x^2/2 = expm1(x) - x - x^2/2; for |x| < 1e-2 use the series
// sum_{k>=3} x^k / k! directly to avoid cancellation.
[[nodiscard]] inline double ex08_exp_quotient(double x) {
  if (std::abs(x) < 1e-2) {
    double term = 1.0 / 6.0, sum = 0.0;  // x^(k-3) * 3!/k! / 6
    for (int k = 3; k < 12; ++k) {
      sum += term;
      term *= x / static_cast<double>(k + 1);
    }
    return sum;
  }
  return (std::expm1(x) - x - 0.5 * x * x) / (x * x * x);
}

// ------------------------------------------------------------ Exercise 9 ---
// Compute
//   (i)  lim_{x->inf} 1 / (sqrt(x^2 - 4x + 1) - x);
//   (ii) lim_{x->inf} 1 / (sqrt(x^2 - 4x + 1) - x + 2).
//
// (i) Multiply by the conjugate:
//   sqrt(x^2 - 4x + 1) - x = (1 - 4x) / (sqrt(x^2 - 4x + 1) + x) -> -4/2 = -2,
//   so the limit is -1/2.
// (ii) Same trick with x - 2:
//   sqrt(x^2 - 4x + 1) - (x - 2) = (x^2 - 4x + 1 - (x-2)^2) / (sqrt(...) + x -
//   2)
//                                = -3 / (sqrt(x^2 - 4x + 1) + x - 2)  ->  0
//                                from below,
//   so 1/(...) = -(sqrt(x^2 - 4x + 1) + x - 2) / 3  ->  -infinity
//   (it behaves like -2x/3).
inline constexpr double ex09_limit_i = -0.5;
inline constexpr double ex09_limit_ii =
    -std::numeric_limits<double>::infinity();

// Stable forms of the two expressions (valid for x > 2 + sqrt(3)).
[[nodiscard]] inline double ex09_expr_i(double x) {
  return (std::sqrt(x * x - 4.0 * x + 1.0) + x) / (1.0 - 4.0 * x);
}
[[nodiscard]] inline double ex09_expr_ii(double x) {
  return -(std::sqrt(x * x - 4.0 * x + 1.0) + x - 2.0) / 3.0;
}

// ----------------------------------------------------------- Exercise 10 ---
// Use the definition e = lim_{x->inf} (1 + 1/x)^x to show that
//   1/e = lim_{x->inf} (1 - 1/x)^x.
// Hint: 1/(1 + 1/x) = x/(x+1) = 1 - 1/(x+1).
//
// With y = x - 1:  1 - 1/x = 1 - 1/(y+1) = 1/(1 + 1/y), so
//   (1 - 1/x)^x = 1 / (1 + 1/y)^(y+1) = 1 / ((1 + 1/y)^y (1 + 1/y))
//              -> 1 / (e * 1) = 1/e   as y -> infinity.
[[nodiscard]] inline double ex10_expr(double x) {
  return std::exp(x * std::log1p(-1.0 / x));
}

}  // namespace p4mfe::ch01::ex
