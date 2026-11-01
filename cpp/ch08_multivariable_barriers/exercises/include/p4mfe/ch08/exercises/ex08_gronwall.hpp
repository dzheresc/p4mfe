#pragma once
// Exercise 8 (section 8.12): Gronwall's inequality.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <vector>

namespace p4mfe::ch08::ex {

// ------------------------------------------------------------ Exercise 8 ---
// Let u, v : [0, inf) -> [0, inf) be continuous, and M > 0 with
//   u(x) <= M + integral_0^x u(t) v(t) dt   for all x >= 0.
// Show that u(x) <= M exp(integral_0^x v(t) dt) for all x >= 0.
//
// Proof. Let W(x) = M + integral_0^x u v dt and V(x) = integral_0^x v dt. Both
// are C^1 (u v and v are continuous), with W' = u v, V' = v, W(0) = M, V(0) = 0.
// Following the hint, let g(x) = W(x) e^(-V(x)). Then
//   g'(x) = (W'(x) - v(x) W(x)) e^(-V(x)) = v(x) (u(x) - W(x)) e^(-V(x)) <= 0,
// because v >= 0 and u <= W by hypothesis. So g is non-increasing and
// g(x) <= g(0) = M, i.e. W(x) <= M e^(V(x)). Finally
//   u(x) <= W(x) <= M exp(integral_0^x v(t) dt).
//
// The bound is sharp: u = M e^V satisfies the hypothesis with equality
// (M + integral_0^x M e^V v dt = M + M (e^V - 1) = M e^V), and then g = M.
//
// Numerical check: on a grid 0 = x_0 < ... < x_n = X, with the integrals by
// the cumulative trapezoidal rule, report
//   the largest u - W (<= 0 if the hypothesis holds),
//   the largest u - M e^V (<= 0 by the conclusion),
//   the largest increase g(x_{i+1}) - g(x_i) (<= 0 by the proof).
// The discrete integrals have O(h^2) errors, so these are <= 0 up to O(h^2)
// when the hypothesis holds with equality.
struct Ex08Check {
    double hypothesis_gap;   // max (u - W): <= 0 when the hypothesis holds
    double conclusion_gap;   // max (u - M e^V), relative to M e^V
    double g_increase;       // max (g_{i+1} - g_i), relative to M
};

template <std::invocable<double> U, std::invocable<double> Vf>
[[nodiscard]] Ex08Check ex08_check(U&& u, Vf&& v, double M, double X, std::size_t n = 2000) {
    const double h = X / static_cast<double>(n);
    double W = M, V = 0.0;
    double ux = u(0.0), vx = v(0.0);
    Ex08Check c{ux - W, (ux - M) / M, -std::numeric_limits<double>::infinity()};
    double g = M;
    for (std::size_t i = 1; i <= n; ++i) {
        const double xn = static_cast<double>(i) * h;
        const double un = u(xn), vn = v(xn);
        W += 0.5 * h * (ux * vx + un * vn);
        V += 0.5 * h * (vx + vn);
        const double gn = W * std::exp(-V);
        const double bound = M * std::exp(V);
        c.hypothesis_gap = std::max(c.hypothesis_gap, un - W);
        c.conclusion_gap = std::max(c.conclusion_gap, (un - bound) / bound);
        c.g_increase = std::max(c.g_increase, (gn - g) / M);
        ux = un;
        vx = vn;
        g = gn;
    }
    return c;
}

// Test functions on [0, inf): v(t) = 1 + sin t, V(x) = x + 1 - cos x.
[[nodiscard]] inline double ex08_v(double t) { return 1.0 + std::sin(t); }
[[nodiscard]] inline double ex08_V(double x) { return x + 1.0 - std::cos(x); }

}  // namespace p4mfe::ch08::ex
