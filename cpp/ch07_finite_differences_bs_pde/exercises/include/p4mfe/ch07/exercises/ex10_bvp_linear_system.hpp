#pragma once
// Exercise 10 (section 7.6): central difference discretization of a second
// order boundary value problem, and its 5 x 5 linear system for n = 6.

#include <p4mfe/ch07/ode.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch07::ex {

// ----------------------------------------------------------- Exercise 10 ---
//   3x^2 y''(x) - x y'(x) + y(x) = 0 on [0, 1],  y(0) = 1,  y(1) = 1/2.
//
// (i) Nodes x_i = i h, h = 1/n. At x_i, i = 1 : n - 1, replace
//   y''(x_i) by (y_{i+1} - 2y_i + y_{i-1}) / h^2,  y'(x_i) by (y_{i+1} - y_{i-1}) / (2h):
//   3 x_i^2 (y_{i+1} - 2y_i + y_{i-1}) / h^2 - x_i (y_{i+1} - y_{i-1}) / (2h) + y_i = 0.
// With x_i = i h the h's cancel (x_i^2/h^2 = i^2, x_i/h = i):
//   (3i^2 + i/2) y_{i-1} + (1 - 6i^2) y_i + (3i^2 - i/2) y_{i+1} = 0,  i = 1 : n - 1,
// with y_0 = 1 and y_n = 1/2.
//
// (ii) n = 6: unknowns Y = (y_1, ..., y_5); move y_0 = 1 and y_6 = 1/2 to the
// right-hand side. A Y = b with the tridiagonal
//        | -5    2.5                       |        | -3.5   |
//        | 13   -23    11                  |        |  0     |
//   A =  |      28.5  -53    25.5          |,  b =  |  0     |
//        |             50   -95    46      |        |  0     |
//        |                  77.5  -149     |        | -36.25 |
// (b_1 = -(3 + 1/2) y_0, b_5 = -(75 - 5/2) y_6 = -72.5/2). Its solution is
//   Y = (1.385311, 1.370622, 1.228661, 1.021815, 0.774770).
//
// Note on the problem itself. The ODE is an Euler (equidimensional) equation:
// y = x^m solves it iff 3m(m - 1) - m + 1 = (3m - 1)(m - 1) = 0, so every
// solution on (0, 1] is y = c1 x + c2 x^(1/3), and every solution tends to 0
// at x = 0. The condition y(0) = 1 cannot be met: the boundary value problem
// has no solution (x = 0 is a singular point, the coefficient of y'' vanishes
// there). The discrete system is still uniquely solvable for each n, but its
// solutions do not converge as n grows: y at x = 1/2 is 1.23, 1.48, 1.80,
// 2.20, 2.70 for n = 6, 12, 24, 48, 96. The exercise only asks for A and b.
struct Ex10System {
    std::vector<double> lower, diag, upper, rhs;  // tridiagonal A (lower[0], upper[n-2] unused), b
};

// Coefficients of y_{i-1}, y_i, y_{i+1} in the equation at node i.
struct Ex10Row {
    double lower, diag, upper;
};
[[nodiscard]] constexpr Ex10Row ex10_row(double i) noexcept {
    return {3.0 * i * i + 0.5 * i, 1.0 - 6.0 * i * i, 3.0 * i * i - 0.5 * i};
}

[[nodiscard]] inline Ex10System ex10_system(std::size_t n, double y0 = 1.0, double yn = 0.5) {
    if (n < 2) throw std::invalid_argument("ex10_system: need n >= 2");
    const std::size_t m = n - 1;
    Ex10System s{std::vector<double>(m), std::vector<double>(m), std::vector<double>(m), std::vector<double>(m)};
    for (std::size_t k = 0; k < m; ++k) {
        const auto row = ex10_row(static_cast<double>(k + 1));
        s.lower[k] = row.lower;
        s.diag[k] = row.diag;
        s.upper[k] = row.upper;
    }
    s.rhs[0] -= s.lower[0] * y0;
    s.rhs[m - 1] -= s.upper[m - 1] * yn;
    s.lower[0] = 0.0;
    s.upper[m - 1] = 0.0;
    return s;
}

// Solution y_0..y_n of the discrete system (Thomas algorithm, section 7.2).
[[nodiscard]] inline OdeSolution ex10_solve(std::size_t n, double y0 = 1.0, double yn = 0.5) {
    const auto s = ex10_system(n, y0, yn);
    const auto inner = solve_tridiagonal(s.lower, s.diag, s.upper, s.rhs);
    OdeSolution sol{std::vector<double>(n + 1), std::vector<double>(n + 1)};
    for (std::size_t i = 0; i <= n; ++i) sol.x[i] = static_cast<double>(i) / static_cast<double>(n);
    sol.y[0] = y0;
    sol.y[n] = yn;
    for (std::size_t i = 1; i < n; ++i) sol.y[i] = inner[i - 1];
    return sol;
}

}  // namespace p4mfe::ch07::ex
