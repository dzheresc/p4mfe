#pragma once
// Exercise 11 (section 7.6): a second order linear ODE as a first order
// system Y' = f(x, Y).

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch07::ex {

// ----------------------------------------------------------- Exercise 11 ---
// Show that y''(x) - 2y'(x) + x^2 y(x) = 0 can be written as Y'(x) = f(x, Y(x)),
// where
//   Y(x) = ( y(x)  ),    f(x, Y(x)) = (  0    1 ) Y(x).
//          ( y'(x) )                  ( -x^2  2 )
//
// Proof. Y'(x) = (y'(x), y''(x))^T. The first component of f(x, Y) is
// 0 * y + 1 * y' = y', which is the first component of Y' by definition. The
// second is -x^2 y + 2 y', and solving the ODE for y'' gives
// y'' = 2y' - x^2 y. So Y' = f(x, Y) holds iff y solves the ODE (the first row
// is the definition of the new unknown y', the second row is the ODE).
// Conversely, if Y = (u, v) solves Y' = f(x, Y), then u' = v and
// v' = 2v - x^2 u, so u'' - 2u' + x^2 u = 0. Every n-th order ODE reduces to
// a first order system this way, so the schemes of section 7.2 (forward Euler
// Y_{k+1} = Y_k + h f(x_k, Y_k), etc.) apply componentwise.
using Vec2 = std::array<double, 2>;

[[nodiscard]] constexpr Vec2 ex11_f(double x, const Vec2& Y) noexcept {
    return {Y[1], -x * x * Y[0] + 2.0 * Y[1]};
}

struct Ex11Solution {
    std::vector<double> x;
    std::vector<Vec2> Y;
};

// Forward Euler for the system on [x0, x_end] (first order).
[[nodiscard]] inline Ex11Solution ex11_euler(const Vec2& Y0, double x0, double x_end, std::size_t n) {
    if (n == 0) throw std::invalid_argument("ex11_euler: need n >= 1");
    const double h = (x_end - x0) / static_cast<double>(n);
    Ex11Solution s{std::vector<double>(n + 1), std::vector<Vec2>(n + 1)};
    s.Y[0] = Y0;
    for (std::size_t k = 0; k <= n; ++k) s.x[k] = x0 + static_cast<double>(k) * h;
    for (std::size_t k = 0; k < n; ++k) {
        const Vec2 d = ex11_f(s.x[k], s.Y[k]);
        s.Y[k + 1] = {s.Y[k][0] + h * d[0], s.Y[k][1] + h * d[1]};
    }
    return s;
}

// Classical Runge-Kutta (fourth order), as an accurate reference.
[[nodiscard]] inline Ex11Solution ex11_rk4(const Vec2& Y0, double x0, double x_end, std::size_t n) {
    if (n == 0) throw std::invalid_argument("ex11_rk4: need n >= 1");
    const double h = (x_end - x0) / static_cast<double>(n);
    const auto axpy = [](const Vec2& Y, double a, const Vec2& d) { return Vec2{Y[0] + a * d[0], Y[1] + a * d[1]}; };
    Ex11Solution s{std::vector<double>(n + 1), std::vector<Vec2>(n + 1)};
    s.Y[0] = Y0;
    for (std::size_t k = 0; k <= n; ++k) s.x[k] = x0 + static_cast<double>(k) * h;
    for (std::size_t k = 0; k < n; ++k) {
        const double x = s.x[k];
        const Vec2& Y = s.Y[k];
        const Vec2 k1 = ex11_f(x, Y);
        const Vec2 k2 = ex11_f(x + 0.5 * h, axpy(Y, 0.5 * h, k1));
        const Vec2 k3 = ex11_f(x + 0.5 * h, axpy(Y, 0.5 * h, k2));
        const Vec2 k4 = ex11_f(x + h, axpy(Y, h, k3));
        s.Y[k + 1] = {Y[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                      Y[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])};
    }
    return s;
}

// Residual y'' - 2y' + x^2 y at interior grid node k of a solution, with y'
// and y'' by central differences of the first component (section 7.1).
[[nodiscard]] inline double ex11_ode_residual(const Ex11Solution& s, std::size_t k) {
    const double h = s.x[1] - s.x[0];
    const double ym = s.Y[k - 1][0], y = s.Y[k][0], yp = s.Y[k + 1][0];
    const double d1 = (yp - ym) / (2.0 * h);
    const double d2 = (yp - 2.0 * y + ym) / (h * h);
    return d2 - 2.0 * d1 + s.x[k] * s.x[k] * y;
}

}  // namespace p4mfe::ch07::ex
