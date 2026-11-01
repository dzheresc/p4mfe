#pragma once
// Chapter 2, sections 2.1-2.4: improper integrals, differentiating improper
// integrals, the midpoint, trapezoidal and Simpson's rules, and how they
// converge.
//
// The composite rules split [a, b] into n intervals of length h = (b - a)/n.
// The tolerance-driven driver (2.4.1) doubles n until two consecutive
// approximations agree to within tol. Infinite limits (2.1) are handled by a
// change of variables onto a finite interval, integrated with the midpoint
// rule, which never evaluates the endpoints.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace p4mfe::ch02 {

inline constexpr double infinity = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------- 2.3 ------
// M_n = h * sum_{i=1}^{n} f(a + (i - 1/2) h),                   error O(h^2)
template <std::invocable<double> F>
[[nodiscard]] double midpoint_rule(F&& f, double a, double b, std::size_t n) {
    if (n == 0) throw std::invalid_argument("midpoint_rule: n must be positive");
    const double h = (b - a) / static_cast<double>(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += f(a + (static_cast<double>(i) + 0.5) * h);
    return h * sum;
}

// T_n = h * (f(a)/2 + sum_{i=1}^{n-1} f(a + i h) + f(b)/2),     error O(h^2)
template <std::invocable<double> F>
[[nodiscard]] double trapezoidal_rule(F&& f, double a, double b, std::size_t n) {
    if (n == 0) throw std::invalid_argument("trapezoidal_rule: n must be positive");
    const double h = (b - a) / static_cast<double>(n);
    double sum = 0.5 * (f(a) + f(b));
    for (std::size_t i = 1; i < n; ++i) sum += f(a + static_cast<double>(i) * h);
    return h * sum;
}

// Simpson's rule on each of the n intervals, using its midpoint:
// S_n = h/6 * (f(a) + 2 sum_{i=1}^{n-1} f(a + i h)
//              + 4 sum_{i=1}^{n} f(a + (i - 1/2) h) + f(b)),     error O(h^4)
// Equivalently S_n = (T_n + 2 M_n) / 3.
template <std::invocable<double> F>
[[nodiscard]] double simpson_rule(F&& f, double a, double b, std::size_t n) {
    if (n == 0) throw std::invalid_argument("simpson_rule: n must be positive");
    const double h = (b - a) / static_cast<double>(n);
    double ends = f(a) + f(b);
    double nodes = 0.0;
    double mids = 0.0;
    for (std::size_t i = 1; i < n; ++i) nodes += f(a + static_cast<double>(i) * h);
    for (std::size_t i = 0; i < n; ++i) mids += f(a + (static_cast<double>(i) + 0.5) * h);
    return h / 6.0 * (ends + 2.0 * nodes + 4.0 * mids);
}

enum class Rule { Midpoint, Trapezoidal, Simpson };

template <std::invocable<double> F>
[[nodiscard]] double integrate(F&& f, double a, double b, std::size_t n, Rule rule = Rule::Simpson) {
    switch (rule) {
        case Rule::Midpoint: return midpoint_rule(f, a, b, n);
        case Rule::Trapezoidal: return trapezoidal_rule(f, a, b, n);
        case Rule::Simpson: break;
    }
    return simpson_rule(f, a, b, n);
}

// ---------------------------------------------------------------- 2.4 ------
struct IntegrationResult {
    double value;
    std::size_t intervals;  // n of the accepted approximation
};

namespace detail {
// Doubles n, starting from n0, until |I_{2n} - I_n| < tol; returns I_{2n}.
template <std::invocable<std::size_t> Approx>
IntegrationResult refine(Approx&& approx, double tol, std::size_t n0, std::size_t max_n) {
    if (tol <= 0.0) throw std::invalid_argument("tolerance must be positive");
    std::size_t n = std::max<std::size_t>(n0, 1);
    double previous = approx(n);
    while (2 * n <= max_n) {
        n *= 2;
        const double current = approx(n);
        if (std::abs(current - previous) < tol) return {current, n};
        previous = current;
    }
    throw std::runtime_error("integration did not reach the tolerance");
}
}  // namespace detail

// 2.4.1: the book's implementation. Start with n0 intervals and double n until
// two consecutive approximations differ by less than tol.
template <std::invocable<double> F>
[[nodiscard]] IntegrationResult integrate_to_tolerance(F&& f, double a, double b, double tol,
                                                       Rule rule = Rule::Simpson, std::size_t n0 = 4,
                                                       std::size_t max_n = std::size_t{1} << 24) {
    return detail::refine([&](std::size_t n) { return integrate(f, a, b, n, rule); }, tol, n0, max_n);
}

// Observed order of convergence from the errors with n and 2n intervals:
// log2(|err_n| / |err_2n|). Close to 2 for midpoint and trapezoidal, 4 for
// Simpson, when f is smooth.
template <std::invocable<double> F>
[[nodiscard]] double observed_order(F&& f, double a, double b, double exact, std::size_t n, Rule rule) {
    const double e1 = std::abs(integrate(f, a, b, n, rule) - exact);
    const double e2 = std::abs(integrate(f, a, b, 2 * n, rule) - exact);
    return std::log2(e1 / e2);
}

// ---------------------------------------------------------------- 2.1 ------
// Improper integral of f over [a, b], where a may be -infinity and b may be
// +infinity. A finite limit is used as is. An infinite one is removed by
//   x = a + t / (1 - t),  dx = dt / (1 - t)^2,  t in [0, 1)      (b = +inf)
//   x = b - t / (1 - t)                                           (a = -inf)
// and the whole real line is split at 0. The integrand must decay fast enough
// for the integral to converge; f is never evaluated at an infinite point.
namespace detail {
template <std::invocable<double> F>
double improper_midpoint(F&& f, double a, double b, std::size_t n) {
    const bool lo_inf = std::isinf(a);
    const bool hi_inf = std::isinf(b);
    if (lo_inf && hi_inf) return improper_midpoint(f, a, 0.0, n) + improper_midpoint(f, 0.0, b, n);
    if (hi_inf) {
        return midpoint_rule(
            [&](double t) {
                const double s = 1.0 - t;
                return f(a + t / s) / (s * s);
            },
            0.0, 1.0, n);
    }
    if (lo_inf) {
        return midpoint_rule(
            [&](double t) {
                const double s = 1.0 - t;
                return f(b - t / s) / (s * s);
            },
            0.0, 1.0, n);
    }
    return midpoint_rule(f, a, b, n);
}
}  // namespace detail

template <std::invocable<double> F>
[[nodiscard]] IntegrationResult improper_integral(F&& f, double a, double b, double tol = 1e-9,
                                                  std::size_t n0 = 8,
                                                  std::size_t max_n = std::size_t{1} << 24) {
    if (!(a < b) || a == infinity || b == -infinity)
        throw std::invalid_argument("improper_integral: need a < b, a != +inf, b != -inf");
    return detail::refine([&](std::size_t n) { return detail::improper_midpoint(f, a, b, n); }, tol,
                          n0, max_n);
}

// The Gamma function, the book's example of an improper integral:
//   Gamma(x) = integral_0^inf t^(x-1) e^(-t) dt,  x > 0,
// with Gamma(n) = (n-1)! for positive integers. Accurate for x >= 1; for
// 0 < x < 1 the integrand is singular at 0 and the midpoint rule converges
// slowly.
[[nodiscard]] inline double gamma_function(double x, double tol = 1e-9) {
    if (x <= 0.0) throw std::domain_error("gamma_function: x must be positive");
    return improper_integral([x](double t) { return std::pow(t, x - 1.0) * std::exp(-t); }, 0.0, infinity,
                             tol)
        .value;
}

// ---------------------------------------------------------------- 2.2 ------
// Differentiating an improper integral with constant limits under the
// integral sign: for I(t) = integral_a^b f(x, t) dx,
//   I'(t) = integral_a^b df/dt(x, t) dx,
// valid when df/dt is continuous and its integral converges uniformly in t.
// df/dt is taken by a central difference in t.
template <std::invocable<double, double> F>
[[nodiscard]] double improper_integral_derivative(F&& f, double a, double b, double t, double tol = 1e-9) {
    const double h = std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(t));
    const auto df_dt = [&](double x) { return (f(x, t + h) - f(x, t - h)) / (2.0 * h); };
    return improper_integral(df_dt, a, b, tol).value;
}

}  // namespace p4mfe::ch02
