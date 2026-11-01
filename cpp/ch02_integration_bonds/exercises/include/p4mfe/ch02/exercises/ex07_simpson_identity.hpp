#pragma once
// Exercise 7 (section 2.12): Simpson's rule as a weighted average of the
// midpoint and trapezoidal rules.

#include <p4mfe/ch02/integration.hpp>

#include <concepts>
#include <cstddef>

namespace p4mfe::ch02::ex {

// ------------------------------------------------------------ Exercise 7 ---
// Let I_n^M, I_n^T and I_n^S be the Midpoint, Trapezoidal and Simpson's rule
// approximations of I = integral_a^b f(x) dx on a uniform partition of [a, b]
// into n intervals. Show that
//   I_n^S = (2/3) I_n^M + (1/3) I_n^T.
//
// Solution. Let h = (b - a)/n, a_i = a + i h, m_i = a + (i - 1/2) h. On
// [a_{i-1}, a_i] the three rules are
//   M: h f(m_i),   T: h (f(a_{i-1}) + f(a_i)) / 2,
//   S: h (f(a_{i-1}) + 4 f(m_i) + f(a_i)) / 6
//    = (2/3) h f(m_i) + (1/3) h (f(a_{i-1}) + f(a_i)) / 2.
// Summing over i = 1:n gives the identity.
//
// Why 2/3 and 1/3: for smooth f the errors are I - I_n^M ~ +(h^2/24) J and
// I - I_n^T ~ -(h^2/12) J with J = integral f''; the combination
// 2(I - M) + (I - T) cancels the h^2 term, which is why Simpson is O(h^4).

// (2/3) M_n + (1/3) T_n, to compare with simpson_rule(f, a, b, n).
template <std::invocable<double> F>
[[nodiscard]] double ex07_simpson_from_mt(F&& f, double a, double b, std::size_t n) {
    return (2.0 * midpoint_rule(f, a, b, n) + trapezoidal_rule(f, a, b, n)) / 3.0;
}

}  // namespace p4mfe::ch02::ex
