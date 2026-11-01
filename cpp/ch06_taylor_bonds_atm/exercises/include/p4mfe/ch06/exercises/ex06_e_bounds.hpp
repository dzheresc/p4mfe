#pragma once
// Exercise 6 (section 6.11): sharp bounds for e from (1 + 1/x)^(...).

#include <cmath>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 6 ---
// Prove that
//   (1 + 1/x)^(x + 1/2 - 1/(12x))  <  e  <  (1 + 1/x)^(x + 1/2),   for all x >= 1.
// (Hint in the book: (1 + 1/x)^x < e < (1 + 1/x)^(x+1) for x >= 1.)
//
// Both sides are powers of 1 + 1/x > 1, so the claim is about exponents:
// e = (1 + 1/x)^(1 / ln(1 + 1/x)), and the hint says x < 1/ln(1 + 1/x) < x + 1.
// We sharpen it to
//   x + 1/2 - 1/(12x)  <  1 / ln(1 + 1/x)  <  x + 1/2.
// Put t = 1/x in (0, 1]. The two inequalities become
//   (a)  ln(1 + t) > 2t / (2 + t),
//   (b)  ln(1 + t) < 12t / (12 + 6t - t^2)       (12 + 6t - t^2 > 0 on (0, 1]).
//
// (a) h(t) = ln(1 + t) - 2t/(2 + t) has h(0) = 0 and
//       h'(t) = 1/(1 + t) - 4/(2 + t)^2 = t^2 / ((1 + t)(2 + t)^2) > 0  for t > 0,
//     so h(t) > 0 for t > 0. This gives the right inequality (for every x > 0).
// (b) k(t) = 12t/(12 + 6t - t^2) - ln(1 + t) has k(0) = 0 and
//       k'(t) = 12(12 + t^2)/(12 + 6t - t^2)^2 - 1/(1 + t)
//             = [12(12 + t^2)(1 + t) - (12 + 6t - t^2)^2] / ((1 + t)(12 + 6t - t^2)^2)
//             = t^3 (24 - t) / ((1 + t)(12 + 6t - t^2)^2) > 0   for 0 < t < 24,
//     because 12(12 + t^2)(1 + t) = 144 + 144t + 12t^2 + 12t^3 and
//     (12 + 6t - t^2)^2 = 144 + 144t + 12t^2 - 12t^3 + t^4. So k(t) > 0 on (0, 1].
//     This gives the left inequality.
// Since 1/2 - 1/(12x) > 0 for x >= 1, the new exponents lie strictly inside
// (x, x + 1): the bounds refine the hint. They are tight: expanding,
//   (1/t + 1/2) ln(1 + t) = 1 + t^2/12 + O(t^3),
//   (1/t + 1/2 - t/12) ln(1 + t) = 1 - t^3/24 + O(t^4),
// so both bounds approach e, the lower one at rate 1/x^3, the upper at 1/x^2.
[[nodiscard]] constexpr double ex06_lower_exponent(double x) noexcept { return x + 0.5 - 1.0 / (12.0 * x); }
[[nodiscard]] constexpr double ex06_upper_exponent(double x) noexcept { return x + 0.5; }

[[nodiscard]] inline double ex06_lower(double x) { return std::exp(ex06_lower_exponent(x) * std::log1p(1.0 / x)); }
[[nodiscard]] inline double ex06_upper(double x) { return std::exp(ex06_upper_exponent(x) * std::log1p(1.0 / x)); }

// The logs of the bounds minus 1: negative / positive iff the inequalities hold.
// Computed in log space, which keeps the tiny gaps for large x accurate.
[[nodiscard]] inline double ex06_log_gap_lower(double x) { return ex06_lower_exponent(x) * std::log1p(1.0 / x) - 1.0; }
[[nodiscard]] inline double ex06_log_gap_upper(double x) { return ex06_upper_exponent(x) * std::log1p(1.0 / x) - 1.0; }

// The auxiliary functions of the proof and their closed-form derivatives.
[[nodiscard]] inline double ex06_h(double t) { return std::log1p(t) - 2.0 * t / (2.0 + t); }
[[nodiscard]] constexpr double ex06_h_prime(double t) noexcept { return t * t / ((1.0 + t) * (2.0 + t) * (2.0 + t)); }
[[nodiscard]] inline double ex06_k(double t) { return 12.0 * t / (12.0 + 6.0 * t - t * t) - std::log1p(t); }
[[nodiscard]] constexpr double ex06_k_prime(double t) noexcept {
    const double D = 12.0 + 6.0 * t - t * t;
    return t * t * t * (24.0 - t) / ((1.0 + t) * D * D);
}

}  // namespace p4mfe::ch06::ex
