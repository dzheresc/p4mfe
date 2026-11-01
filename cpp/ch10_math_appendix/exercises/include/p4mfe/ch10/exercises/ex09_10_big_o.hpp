#pragma once
// Exercises 9-10 (section 10.10): sums of O and o terms; the orders of
// sum k^2 and sum k^3.

#include <p4mfe/ch10/big_o.hpp>
#include <p4mfe/ch10/sums.hpp>

#include <cmath>
#include <cstdint>

namespace p4mfe::ch10::ex {

// ------------------------------------------------------------ Exercise 9 ---
// Let n > 0. Show that, as x -> 0,
//   O(x^n) + O(x^n) = O(x^n)  (10.97),   o(x^n) + o(x^n) = o(x^n)  (10.98).
//
// (10.97). Let f(x) = O(x^n) and g(x) = O(x^n): limsup_{x->0} |f(x)/x^n| = A < oo
// and limsup |g(x)/x^n| = B < oo. By the triangle inequality
//   |(f(x) + g(x)) / x^n| <= |f(x)/x^n| + |g(x)/x^n|,
// and the limsup of a sum is at most the sum of the limsups, so
//   limsup_{x->0} |(f + g)/x^n| <= A + B < oo,  i.e. f + g = O(x^n).
// Equivalently: |f| <= C_f |x|^n for |x| < d_f and |g| <= C_g |x|^n for
// |x| < d_g give |f + g| <= (C_f + C_g) |x|^n for |x| < min(d_f, d_g).
//
// (10.98). If f/x^n -> 0 and g/x^n -> 0 then (f + g)/x^n = f/x^n + g/x^n -> 0
// (the limit of a sum is the sum of the limits), so f + g = o(x^n).
//
// (The equalities are one-way statements about classes of functions. In the
// same sense O(x^n) - O(x^n) = O(x^n), not 0, and c O(x^n) = O(x^n).
// For non-integer n read x^n as |x|^n.)
//
// Numerical illustration (x > 0, n = 2): f = x^2 sin(1/x) and g = 3x^2 are
// O(x^2) but not o(x^2), and |(f + g)/x^2| = |sin(1/x) + 3| <= 4; f = x^3 and
// g = x^2 sqrt(x) are o(x^2), and (f + g)/x^2 = x + sqrt(x) -> 0.
[[nodiscard]] inline double ex09_f_big(double x) { return x * x * std::sin(1.0 / x); }
[[nodiscard]] inline double ex09_g_big(double x) { return 3.0 * x * x; }
[[nodiscard]] inline double ex09_f_small(double x) { return x * x * x; }
[[nodiscard]] inline double ex09_g_small(double x) { return x * x * std::sqrt(x); }

// ----------------------------------------------------------- Exercise 10 ---
// Prove that, as n -> oo,
//   sum k^2 = O(n^3),  sum k^2 = n^3/3 + O(n^2),
//   sum k^3 = O(n^4),  sum k^3 = n^4/4 + O(n^3).
//
// From (10.15), sum_{k=1}^{n} k^2 = n(n+1)(2n+1)/6 = n^3/3 + n^2/2 + n/6, so
// for n >= 1
//   (sum k^2) / n^3 = 1/3 + 1/(2n) + 1/(6n^2) <= 1/3 + 1/2 + 1/6 = 1,
//   (sum k^2 - n^3/3) / n^2 = 1/2 + 1/(6n) <= 2/3.
// Both ratios are bounded (they tend to 1/3 and 1/2), so the limsups are finite.
// From (10.16), sum k^3 = n^2(n+1)^2/4 = n^4/4 + n^3/2 + n^2/4, so
//   (sum k^3) / n^4 = 1/4 + 1/(2n) + 1/(4n^2) <= 1,
//   (sum k^3 - n^4/4) / n^3 = 1/2 + 1/(4n) <= 3/4.
// (The first bound of each pair also follows without the closed form:
// k^j <= n^j for k <= n gives sum_{k=1}^{n} k^j <= n^(j+1).)

// (sum k^2)/n^3, (sum k^2 - n^3/3)/n^2, (sum k^3)/n^4, (sum k^3 - n^4/4)/n^3,
// from the exact integer sums of chapter 10 (n up to about 10^4 in 64 bits).
[[nodiscard]] inline double ex10_ratio_k2(std::uint64_t n) {
    const double N = static_cast<double>(n);
    return static_cast<double>(sum_k2(n)) / (N * N * N);
}
[[nodiscard]] inline double ex10_remainder_ratio_k2(std::uint64_t n) {
    // 3 S - n^3 is an exact integer; divide by 3 n^2.
    const double N = static_cast<double>(n);
    return static_cast<double>(3 * sum_k2(n) - n * n * n) / (3.0 * N * N);
}
[[nodiscard]] inline double ex10_ratio_k3(std::uint64_t n) {
    const double N = static_cast<double>(n);
    return static_cast<double>(sum_k3(n)) / (N * N * N * N);
}
[[nodiscard]] inline double ex10_remainder_ratio_k3(std::uint64_t n) {
    const double N = static_cast<double>(n);
    return static_cast<double>(4 * sum_k3(n) - n * n * n * n) / (4.0 * N * N * N);
}

}  // namespace p4mfe::ch10::ex
