#pragma once
// Exercise 4 (section 6.11): Taylor approximations of e^(g(x)), and of
// e^((x+1)^2) in two ways.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 4 ---
// (i) Let g(x) be infinitely differentiable. Find the linear and quadratic
// Taylor approximations of e^(g(x)) around 0.
//
// With f = e^g: f' = g' e^g, f'' = (g'' + (g')^2) e^g, so
//   e^(g(x)) ~ e^(g(0)) (1 + g'(0) x),
//   e^(g(x)) ~ e^(g(0)) (1 + g'(0) x + (g''(0) + g'(0)^2) x^2 / 2).
//
// (ii) The quadratic approximation of e^((x+1)^2): g(0) = 1, g'(0) = 2,
// g''(0) = 2, so
//   e^((x+1)^2) ~ e (1 + 2x + 3x^2).
//
// (iii) The same from the series of e^x and e^(x^2): e^((x+1)^2) = e e^(2x) e^(x^2),
// with e^(2x) = 1 + 2x + 2x^2 + O(x^3) and e^(x^2) = 1 + x^2 + O(x^4). Their
// product, dropping the x^3 and higher terms, is
//   e (1 + 2x + 2x^2)(1 + x^2) = e (1 + 2x + 3x^2) + O(x^3).
// Both routes agree.
using Quadratic = std::array<double, 3>;  // c0 + c1 x + c2 x^2

// (i): coefficients from g(0), g'(0), g''(0).
[[nodiscard]] inline Quadratic ex04_quadratic(double g0, double g1, double g2) {
    const double e = std::exp(g0);
    return {e, e * g1, 0.5 * e * (g2 + g1 * g1)};
}
[[nodiscard]] inline Quadratic ex04_linear(double g0, double g1) { return {std::exp(g0), std::exp(g0) * g1, 0.0}; }

// Product of two polynomials truncated to degree 2.
[[nodiscard]] constexpr Quadratic ex04_multiply(const Quadratic& a, const Quadratic& b) noexcept {
    Quadratic c{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; i + j < 3; ++j) c[i + j] += a[i] * b[j];
    return c;
}

// (ii): g(x) = (x + 1)^2.
[[nodiscard]] inline Quadratic ex04_part_ii() { return ex04_quadratic(1.0, 2.0, 2.0); }

// (iii): e * [1 + 2x + (2x)^2/2] * [1 + x^2], truncated.
[[nodiscard]] inline Quadratic ex04_part_iii() {
    const Quadratic e2x{1.0, 2.0, 2.0};
    const Quadratic ex2{1.0, 0.0, 1.0};
    auto c = ex04_multiply(e2x, ex2);
    for (double& v : c) v *= std::numbers::e;
    return c;
}

[[nodiscard]] constexpr double ex04_eval(const Quadratic& c, double x) noexcept {
    return c[0] + x * (c[1] + x * c[2]);
}

}  // namespace p4mfe::ch06::ex
