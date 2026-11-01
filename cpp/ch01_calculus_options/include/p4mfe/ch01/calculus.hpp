#pragma once
// Chapter 1, sections 1.1-1.5: differentiation, integration, differentiating
// definite integrals, limits, L'Hopital's rule.
//
// The book treats these analytically. The functions below are the numerical
// counterparts used to check closed-form answers: finite-difference
// derivatives, Riemann sums, the Leibniz rule for d/dt of an integral with
// variable limits, and compounding limits.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace p4mfe::ch01 {

namespace detail {
// Step sizes that balance truncation and round-off error:
// eps^(1/3) for the central first difference, eps^(1/4) for the second.
inline double first_diff_step(double x) noexcept {
    return std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(x));
}
inline double second_diff_step(double x) noexcept {
    return std::pow(std::numeric_limits<double>::epsilon(), 0.25) * std::max(1.0, std::abs(x));
}
}  // namespace detail

// ---------------------------------------------------------------- 1.1 ------
// f'(x) ~ (f(x+h) - f(x-h)) / (2h), error O(h^2).
template <std::invocable<double> F>
[[nodiscard]] double derivative(F&& f, double x, double h = 0.0) {
    if (h <= 0.0) h = detail::first_diff_step(x);
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// f''(x) ~ (f(x+h) - 2f(x) + f(x-h)) / h^2, error O(h^2).
template <std::invocable<double> F>
[[nodiscard]] double second_derivative(F&& f, double x, double h = 0.0) {
    if (h <= 0.0) h = detail::second_diff_step(x);
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

// ---------------------------------------------------------------- 1.2 ------
// The definite integral as the limit of Riemann sums over n equal intervals.
enum class RiemannRule { Left, Right, Midpoint };

template <std::invocable<double> F>
[[nodiscard]] double riemann_sum(F&& f, double a, double b, std::size_t n,
                                 RiemannRule rule = RiemannRule::Midpoint) {
    if (n == 0) throw std::invalid_argument("riemann_sum: n must be positive");
    const double h = (b - a) / static_cast<double>(n);
    const double offset = rule == RiemannRule::Left    ? 0.0
                          : rule == RiemannRule::Right ? 1.0
                                                       : 0.5;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += f(a + (static_cast<double>(i) + offset) * h);
    return h * sum;
}

// ---------------------------------------------------------------- 1.3 ------
// Leibniz rule. For I(t) = integral from a(t) to b(t) of f(x, t) dx,
//   I'(t) = integral from a(t) to b(t) of df/dt(x, t) dx
//           + f(b(t), t) b'(t) - f(a(t), t) a'(t).
// df/dt, a' and b' are taken by central differences; the integral by an
// n-interval midpoint sum.
template <std::invocable<double, double> F, std::invocable<double> A, std::invocable<double> B>
[[nodiscard]] double integral_derivative(F&& f, A&& a, B&& b, double t, std::size_t n = 1000) {
    const double lo = a(t);
    const double hi = b(t);
    const auto df_dt = [&](double x) { return derivative([&](double s) { return f(x, s); }, t); };
    return riemann_sum(df_dt, lo, hi, n) + f(hi, t) * derivative(b, t) - f(lo, t) * derivative(a, t);
}

// ---------------------------------------------------------------- 1.4 ------
// (1 + x/n)^n -> e^x as n -> infinity: n-times-a-year compounding converges
// to continuous compounding. Returns the growth factor of one unit over time
// t at rate r compounded n times per unit time, (1 + r/n)^(n t).
[[nodiscard]] inline double compounding_factor(double r, double t, double n) {
    if (n <= 0.0) throw std::invalid_argument("compounding_factor: n must be positive");
    return std::pow(1.0 + r / n, n * t);
}

// The limit n -> infinity of compounding_factor: e^(r t).
[[nodiscard]] inline double continuous_compounding_factor(double r, double t) noexcept {
    return std::exp(r * t);
}

// ---------------------------------------------------------------- 1.5 ------
// L'Hopital's rule for a 0/0 limit of f(x)/g(x) at x0: returns f'(x0)/g'(x0),
// or f''(x0)/g''(x0) if g'(x0) vanishes (e.g. (1 - cos x)/x^2 at 0).
// f and g themselves must be defined and smooth at x0; only the quotient is
// indeterminate there.
template <std::invocable<double> F, std::invocable<double> G>
[[nodiscard]] double lhopital_limit(F&& f, G&& g, double x0, double tol = 1e-6) {
    const double dg = derivative(g, x0);
    if (std::abs(dg) > tol) return derivative(f, x0) / dg;
    const double d2g = second_derivative(g, x0);
    if (std::abs(d2g) > tol)
        return second_derivative(f, x0) / d2g;
    throw std::domain_error("lhopital_limit: g' and g'' both vanish at x0");
}

}  // namespace p4mfe::ch01
