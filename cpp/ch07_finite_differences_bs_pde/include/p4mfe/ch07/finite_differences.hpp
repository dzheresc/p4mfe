#pragma once
// Chapter 7, section 7.1: forward, backward and central finite differences.
//
// From Taylor's formula (chapter 6) around a, with h > 0:
//   forward    f'(a) ~ (f(a+h) - f(a)) / h              error  h/2 |f''|       O(h)
//   backward   f'(a) ~ (f(a) - f(a-h)) / h              error  h/2 |f''|       O(h)
//   central    f'(a) ~ (f(a+h) - f(a-h)) / (2h)         error  h^2/6 |f'''|    O(h^2)
//   central    f''(a) ~ (f(a+h) - 2f(a) + f(a-h)) / h^2 error  h^2/12 |f''''| O(h^2)
// The central first and second differences are chapter 1's `derivative` and
// `second_derivative` and are reused here, not rewritten.
//
// In floating point, f is only known to about eps |f|, so a difference
// quotient also carries a round-off error of order eps |f| / h (first
// derivatives) or eps |f| / h^2 (second derivative). Minimizing
// truncation + round-off gives the optimal steps returned by `optimal_step`:
// h ~ eps^(1/2) forward/backward, eps^(1/3) central, eps^(1/4) second order.

#include <p4mfe/ch01/calculus.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch07 {

enum class FdScheme { Forward, Backward, Central };

// ---------------------------------------------------------------- 7.1 ------
template <std::invocable<double> F>
[[nodiscard]] double forward_difference(F&& f, double a, double h) {
    return (f(a + h) - f(a)) / h;
}

template <std::invocable<double> F>
[[nodiscard]] double backward_difference(F&& f, double a, double h) {
    return (f(a) - f(a - h)) / h;
}

template <std::invocable<double> F>
[[nodiscard]] double central_difference(F&& f, double a, double h) {
    if (!(h > 0.0)) throw std::invalid_argument("central_difference: need h > 0");
    return ch01::derivative(f, a, h);
}

// Central approximation of f''(a).
template <std::invocable<double> F>
[[nodiscard]] double second_central_difference(F&& f, double a, double h) {
    if (!(h > 0.0)) throw std::invalid_argument("second_central_difference: need h > 0");
    return ch01::second_derivative(f, a, h);
}

// f'(a) by the chosen scheme.
template <std::invocable<double> F>
[[nodiscard]] double first_difference(F&& f, double a, double h, FdScheme scheme) {
    switch (scheme) {
        case FdScheme::Forward: return forward_difference(f, a, h);
        case FdScheme::Backward: return backward_difference(f, a, h);
        case FdScheme::Central: return central_difference(f, a, h);
    }
    throw std::invalid_argument("first_difference: unknown scheme");
}

// Order of the truncation error: 1 for one-sided, 2 for central differences.
[[nodiscard]] constexpr int truncation_order(FdScheme scheme) noexcept { return scheme == FdScheme::Central ? 2 : 1; }

// -------------------------------------------------------------- 7.3.1 ------
// Step minimizing truncation + round-off for f'(a) (derivative_order = 1)
// or f''(a) (derivative_order = 2), scaled by max(1, |a|):
//   one-sided f'  : sqrt(eps)    ~ 1.5e-8
//   central   f'  : eps^(1/3)    ~ 6.1e-6
//   central   f'' : eps^(1/4)    ~ 1.2e-4
[[nodiscard]] inline double optimal_step(FdScheme scheme, int derivative_order = 1, double a = 0.0) {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const double scale = std::max(1.0, std::abs(a));
    if (derivative_order == 2) return std::pow(eps, 0.25) * scale;
    if (derivative_order != 1) throw std::invalid_argument("optimal_step: derivative_order must be 1 or 2");
    return (scheme == FdScheme::Central ? std::cbrt(eps) : std::sqrt(eps)) * scale;
}

// One row of an accuracy study: the step, the approximation and its error.
struct FdErrorRow {
    double h;
    double approx;
    double error;  // approx - exact
};

// Approximations of f'(a) (or f''(a)) for h = h0, h0/ratio, h0/ratio^2, ...
// (n rows), with their errors against `exact`. Shows the error falling like
// h^p until round-off takes over and it grows again.
template <std::invocable<double> F>
[[nodiscard]] std::vector<FdErrorRow> fd_error_table(F&& f, double a, double exact, FdScheme scheme, double h0,
                                                     std::size_t n, double ratio = 10.0,
                                                     int derivative_order = 1) {
    if (!(h0 > 0.0 && ratio > 1.0)) throw std::invalid_argument("fd_error_table: need h0 > 0 and ratio > 1");
    std::vector<FdErrorRow> rows;
    rows.reserve(n);
    double h = h0;
    for (std::size_t i = 0; i < n; ++i, h /= ratio) {
        const double v = derivative_order == 2 ? second_central_difference(f, a, h) : first_difference(f, a, h, scheme);
        rows.push_back({h, v, v - exact});
    }
    return rows;
}

// Observed order log(|e(h)| / |e(h/2)|) / log 2, for steps where truncation
// dominates (≈ 1 for one-sided, 2 for central differences).
template <std::invocable<double> F>
[[nodiscard]] double fd_observed_order(F&& f, double a, double exact, FdScheme scheme, double h,
                                       int derivative_order = 1) {
    const auto rows = fd_error_table(f, a, exact, scheme, h, 2, 2.0, derivative_order);
    return std::log2(std::abs(rows[0].error) / std::abs(rows[1].error));
}

}  // namespace p4mfe::ch07
