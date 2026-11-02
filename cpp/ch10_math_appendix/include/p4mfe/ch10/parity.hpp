#pragma once
// Chapter 10, section 10.1: even and odd functions.
//
// f is even if f(-x) = f(x) and odd if f(-x) = -f(x) for all x. Every f is
// the sum of an even and an odd part,
//   f_e(x) = (f(x) + f(-x)) / 2,   f_o(x) = (f(x) - f(-x)) / 2,
// and on a symmetric interval the odd part integrates to zero:
//   int_{-a}^{a} f(x) dx = 2 int_0^a f_e(x) dx
// (= 2 int_0^a f for even f, = 0 for odd f). The derivative of an even
// function is odd and vice versa; products follow the sign rule
// (even * odd = odd, odd * odd = even). Finance example: the standard normal
// density is even, so N(-x) = 1 - N(x).

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <p4mfe/ch02/integration.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch10 {

enum class Parity { Even, Odd, Neither };

namespace detail {
template <typename F>
bool symmetric_with_sign(F& f, std::span<const double> xs, double sign,
                         double tol) {
  return std::all_of(xs.begin(), xs.end(), [&](double x) {
    const double fx = f(x), fmx = f(-x);
    return std::abs(fmx - sign * fx) <= tol * std::max(1.0, std::abs(fx));
  });
}
}  // namespace detail

// Checks f(-x) = f(x) (resp. -f(x)) on the sample points, to a relative
// tolerance.
template <std::invocable<double> F>
[[nodiscard]] bool is_even(F&& f, std::span<const double> xs,
                           double tol = 1e-12) {
  return detail::symmetric_with_sign(f, xs, 1.0, tol);
}
template <std::invocable<double> F>
[[nodiscard]] bool is_odd(F&& f, std::span<const double> xs,
                          double tol = 1e-12) {
  return detail::symmetric_with_sign(f, xs, -1.0, tol);
}

// Even, odd, or neither on the sample points. The zero function is reported as
// even.
template <std::invocable<double> F>
[[nodiscard]] Parity parity(F&& f, std::span<const double> xs,
                            double tol = 1e-12) {
  if (is_even(f, xs, tol)) return Parity::Even;
  if (is_odd(f, xs, tol)) return Parity::Odd;
  return Parity::Neither;
}

template <std::invocable<double> F>
[[nodiscard]] double even_part(F&& f, double x) {
  return 0.5 * (f(x) + f(-x));
}
template <std::invocable<double> F>
[[nodiscard]] double odd_part(F&& f, double x) {
  return 0.5 * (f(x) - f(-x));
}

// int_{-a}^{a} f = 2 int_0^a f_e, with chapter 2's quadrature on [0, a]
// (n intervals). The odd part is dropped exactly rather than integrated.
template <std::invocable<double> F>
[[nodiscard]] double symmetric_integral(F&& f, double a, std::size_t n = 200,
                                        ch02::Rule rule = ch02::Rule::Simpson) {
  if (a < 0.0) throw std::invalid_argument("symmetric_integral: need a >= 0");
  return 2.0 * ch02::integrate([&](double x) { return even_part(f, x); }, 0.0,
                               a, n, rule);
}

}  // namespace p4mfe::ch10
