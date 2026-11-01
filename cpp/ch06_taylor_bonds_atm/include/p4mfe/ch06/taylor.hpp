#pragma once
// Chapter 6, sections 6.1-6.2.1: Taylor's formula for functions of one and
// of several variables.
//
// One variable (6.1). If f has n+1 continuous derivatives around a,
//   f(x) = P_n(x) + R_n(x),   P_n(x) = sum_{k=0}^{n} f^(k)(a) (x-a)^k / k!,
// with the remainder in integral form
//   R_n(x) = integral_a^x (x-t)^n / n! f^(n+1)(t) dt,
// or in Lagrange form R_n(x) = f^(n+1)(c) (x-a)^(n+1) / (n+1)! for some c
// between a and x, so |R_n(x)| <= M |x-a|^(n+1) / (n+1)! when
// |f^(n+1)| <= M there.
//
// Several variables (6.2). With h = x - a,
//   f(x) = f(a) + grad f(a) . h + 1/2 h^T Hess f(a) h + O(|h|^3).
// For two variables (6.2.1), with dx = x - a, dy = y - b,
//   f(x, y) ~ f + f_x dx + f_y dy + 1/2 (f_xx dx^2 + 2 f_xy dx dy + f_yy dy^2).
//
// Derivatives may be given (closed form) or computed by chapter 1's finite
// differences.

#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/multivariable.hpp>
#include <p4mfe/ch02/integration.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace p4mfe::ch06 {

using ch01::Matrix;
using ch01::Vector;

// ---------------------------------------------------------------- 6.1 ------
// P_n(x) from the derivatives f(a), f'(a), ..., f^(n)(a) (n = derivs.size()-1).
// Horner-like: accumulates (x-a)^k / k! term by term.
[[nodiscard]] inline double taylor_polynomial(std::span<const double> derivs, double a, double x) {
    if (derivs.empty()) throw std::invalid_argument("taylor_polynomial: need at least f(a)");
    const double h = x - a;
    double term = 1.0;  // (x-a)^k / k!
    double sum = 0.0;
    for (std::size_t k = 0; k < derivs.size(); ++k) {
        if (k > 0) term *= h / static_cast<double>(k);
        sum += derivs[k] * term;
    }
    return sum;
}

// Linear and quadratic Taylor approximations with finite-difference
// derivatives (chapter 1):
//   f(x) ~ f(a) + f'(a)(x-a),   f(x) ~ f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2.
template <std::invocable<double> F>
[[nodiscard]] double taylor_linear(F&& f, double a, double x) {
    return f(a) + ch01::derivative(f, a) * (x - a);
}
template <std::invocable<double> F>
[[nodiscard]] double taylor_quadratic(F&& f, double a, double x) {
    const double h = x - a;
    return f(a) + ch01::derivative(f, a) * h + 0.5 * ch01::second_derivative(f, a) * h * h;
}

// Lagrange bound: |R_n(x)| <= M |x-a|^(n+1) / (n+1)!.
[[nodiscard]] inline double taylor_remainder_bound(double M, double a, double x, unsigned n) {
    return M * std::pow(std::abs(x - a), n + 1) / std::tgamma(static_cast<double>(n) + 2.0);
}

// Integral form of the remainder, R_n(x) = int_a^x (x-t)^n / n! f^(n+1)(t) dt,
// computed with chapter 2's Simpson's rule to a tolerance. `dn1` is f^(n+1).
template <std::invocable<double> G>
[[nodiscard]] double taylor_remainder_integral(G&& dn1, double a, double x, unsigned n, double tol = 1e-12) {
    if (x == a) return 0.0;
    const double nfact = std::tgamma(static_cast<double>(n) + 1.0);
    const auto integrand = [&](double t) { return std::pow(x - t, n) / nfact * dn1(t); };
    // Simpson on [min, max]; int_a^x = -int_x^a.
    const double lo = std::min(a, x), hi = std::max(a, x);
    const double v = ch02::integrate_to_tolerance(integrand, lo, hi, tol).value;
    return x > a ? v : -v;
}

// ---------------------------------------------------------------- 6.2 ------
// f(a) + g . h + 1/2 h^T H h for given f(a), gradient g and Hessian H.
[[nodiscard]] inline double taylor_quadratic_nd(double fa, const Vector& grad, const Matrix& H, const Vector& h) {
    const std::size_t n = h.size();
    if (grad.size() != n || H.size() != n) throw std::invalid_argument("taylor_quadratic_nd: size mismatch");
    double lin = 0.0, quad = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        lin += grad[i] * h[i];
        for (std::size_t j = 0; j < n; ++j) quad += h[i] * H[i][j] * h[j];
    }
    return fa + lin + 0.5 * quad;
}

// Same, with gradient and Hessian of f at a from chapter 1's central differences.
template <typename F>
[[nodiscard]] double taylor_quadratic_nd(F&& f, const Vector& a, const Vector& x) {
    if (a.size() != x.size()) throw std::invalid_argument("taylor_quadratic_nd: size mismatch");
    Vector h(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) h[i] = x[i] - a[i];
    return taylor_quadratic_nd(f(a), ch01::gradient(f, a), ch01::hessian(f, a), h);
}

// First-order (linear) multivariable Taylor approximation, f(a) + grad f(a) . h.
template <typename F>
[[nodiscard]] double taylor_linear_nd(F&& f, const Vector& a, const Vector& x) {
    if (a.size() != x.size()) throw std::invalid_argument("taylor_linear_nd: size mismatch");
    const Vector g = ch01::gradient(f, a);
    double v = f(a);
    for (std::size_t i = 0; i < a.size(); ++i) v += g[i] * (x[i] - a[i]);
    return v;
}

// -------------------------------------------------------------- 6.2.1 ------
// The partial derivatives of f(x, y) at (a, b) used by the two-variable formula.
struct Partials2D {
    double f, fx, fy, fxx, fxy, fyy;
};

// Partials from chapter 1's central differences.
template <std::invocable<double, double> F>
[[nodiscard]] Partials2D partials_2d(F&& f, double a, double b) {
    const auto g = [&](const Vector& v) { return f(v[0], v[1]); };
    const Vector p{a, b};
    return {f(a, b),
            ch01::partial(g, p, 0),
            ch01::partial(g, p, 1),
            ch01::second_partial(g, p, 0, 0),
            ch01::second_partial(g, p, 0, 1),
            ch01::second_partial(g, p, 1, 1)};
}

// Linear and quadratic two-variable Taylor approximations at (a + dx, b + dy).
[[nodiscard]] constexpr double taylor_linear_2d(const Partials2D& d, double dx, double dy) noexcept {
    return d.f + d.fx * dx + d.fy * dy;
}
[[nodiscard]] constexpr double taylor_quadratic_2d(const Partials2D& d, double dx, double dy) noexcept {
    return taylor_linear_2d(d, dx, dy) + 0.5 * (d.fxx * dx * dx + 2.0 * d.fxy * dx * dy + d.fyy * dy * dy);
}

}  // namespace p4mfe::ch06
