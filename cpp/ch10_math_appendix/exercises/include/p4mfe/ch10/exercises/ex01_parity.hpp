#pragma once
// Exercise 1 (section 10.10): parity of x^i f(x^j) for an odd function f.

#include <p4mfe/ch10/parity.hpp>

#include <cmath>
#include <concepts>

namespace p4mfe::ch10::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Let f : R -> R be an odd function, f(-x) = -f(x).
//
// (i) Show that x f(x) is even and x^2 f(x) is odd.
//   (-x) f(-x) = (-x)(-f(x)) = x f(x), so x f(x) is even.
//   (-x)^2 f(-x) = x^2 (-f(x)) = -x^2 f(x), so x^2 f(x) is odd.
//
// (ii) Show that g1(x) = f(x^2) is even and g2(x) = f(x^3) is odd.
//   g1(-x) = f((-x)^2) = f(x^2) = g1(x): even (for any f, odd or not).
//   g2(-x) = f((-x)^3) = f(-x^3) = -f(x^3) = -g2(x): odd.
//
// (iii) Let h(x) = x^i f(x^j), i and j positive integers. Show that h is odd
// if i + j is odd and even if i + j is even.
//   (-x)^j = (-1)^j x^j, and f(-y) = -f(y), i.e. f((-1)^j y) = (-1)^j f(y)
// (for j even trivially, for j odd by oddness). Hence
//   h(-x) = (-1)^i x^i f((-1)^j x^j) = (-1)^i x^i (-1)^j f(x^j) = (-1)^(i+j) h(x).
// So h is even when i + j is even and odd when i + j is odd. Parts (i) and
// (ii) are the cases (i, j) = (1, 1), (2, 1), (0, 2), (0, 3); the argument
// works for i = 0 as well.

// h(x) = x^i f(x^j).
template <std::invocable<double> F>
[[nodiscard]] double ex01_h(F&& f, unsigned i, unsigned j, double x) {
    return std::pow(x, static_cast<double>(i)) * f(std::pow(x, static_cast<double>(j)));
}

// The parity predicted by (iii): (-1)^(i+j).
[[nodiscard]] constexpr Parity ex01_predicted_parity(unsigned i, unsigned j) noexcept {
    return (i + j) % 2 == 0 ? Parity::Even : Parity::Odd;
}

}  // namespace p4mfe::ch10::ex
