#pragma once
// Exercise 9 (section 7.6): forward Euler for y' = y is first order convergent.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/ch07/ode.hpp>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 9 ---
// y'(x) = y(x) on [0, 1], y(0) = 1; nodes x_i = i h, h = 1/n.
//
// (i) The forward difference (y_{i+1} - y_i)/h for y'(x_i) gives
//   (y_{i+1} - y_i) / h = y_i   <=>   y_{i+1} = (1 + h) y_i,   i = 0 : n - 1,
// with y_0 = 1. By induction y_i = (1 + h)^i: true for i = 0, and
// y_{i+1} = (1 + h) (1 + h)^i = (1 + h)^(i+1).
//
// (ii) The exact solution is y(x) = e^x, so the error at x_i is
//   e_i = (1 + h)^i - e^(ih) = e^(i ln(1+h)) - e^(ih) = e^(ih) (e^(i (ln(1+h) -
//   h)) - 1).
// With ln(1 + h) = h - h^2/2 + O(h^3),
//   i (ln(1 + h) - h) = -i h^2/2 + O(i h^3) = -i h^2/2 + O(h^2)   (i h <= 1),
// a quantity of size at most h/2 + O(h^2) -> 0, and e^z - 1 = z + O(z^2) gives
//   e^(i (ln(1+h) - h)) - 1 = -i h^2/2 + O(h^2).
// [The book writes O(h^3) for the remainder; uniformly in i = 0 : n it is
// O(i h^3) + O(i^2 h^4) = O(h^2), still o(h), which is what is needed.]
// Since e^(ih) <= e and i h^2 = (ih) h <= h,
//   max_i |e_i| <= e * h/2 + O(h^2) = (e/2) (1/n) + O(1/n^2),
// so max_i |e_i| -> 0 as n -> infinity: the scheme converges, and it is first
// order convergent, max_i |e_i| = O(1/n).
// In fact ln(1 + h) - h < 0, so every e_i <= 0, |e_i| increases with i, and
// the maximum is |e_n| = e - (1 + 1/n)^n = e/(2n) - 11e/(24 n^2) + O(1/n^3):
// the bound e/(2n) is sharp.
[[nodiscard]] inline OdeSolution ex09_solve(std::size_t n) {
  return euler_forward([](double, double y) { return y; }, 0.0, 1.0, 1.0, n);
}
// y_i = (1 + h)^i, the closed form of (i).
[[nodiscard]] inline double ex09_closed_form(std::size_t i, std::size_t n) {
  return std::pow(1.0 + 1.0 / static_cast<double>(n), static_cast<double>(i));
}
[[nodiscard]] inline double ex09_max_error(std::size_t n) {
  return max_error(ex09_solve(n), [](double x) { return std::exp(x); });
}
// The bound e/(2n) of (ii) without the o(1/n) term.
[[nodiscard]] inline double ex09_bound(std::size_t n) {
  return std::numbers::e / (2.0 * static_cast<double>(n));
}

}  // namespace p4mfe::ch07::ex
