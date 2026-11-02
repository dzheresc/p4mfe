#pragma once
// Chapter 7, section 7.2: finite difference solutions of ODEs.
//
// Replace the derivatives in an ODE by finite differences on a grid
// x_k = x0 + k h and solve for the grid values y_k ~ y(x_k).
//
// Initial value problem y' = f(x, y), y(x0) = y0:
//   forward  (explicit Euler)  y_{k+1} = y_k + h f(x_k, y_k) O(h) backward
//   (implicit Euler)  y_{k+1} = y_k + h f(x_{k+1}, y_{k+1})          O(h)
//            each step solves a scalar equation for y_{k+1}; chapter 5's
//            Newton's method is used for it.
//   central  (leapfrog)        y_{k+1} = y_{k-1} + 2h f(x_k, y_k) O(h^2)
//            two-step; y_1 comes from one forward step. Second order, but for
//            decaying solutions (y' = -a y) it has a growing parasitic mode,
//            while the backward scheme is stable for every h.
//
// Boundary value problem y'' + p(x) y' + q(x) y = r(x) on [a, b] with
// y(a) = ya, y(b) = yb: central differences for y'' and y' at the n-1 interior
// nodes give a tridiagonal linear system, solved in O(n) by the Thomas
// algorithm. The error is O(h^2).

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch05/solvers.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch07 {

// Grid values: x[k] and the approximation y[k] ~ y(x[k]), k = 0..n.
struct OdeSolution {
  std::vector<double> x;
  std::vector<double> y;
};

enum class OdeScheme { Forward, Backward, Central };

namespace detail {
inline OdeSolution make_grid(double x0, double x_end, std::size_t n) {
  if (n == 0) throw std::invalid_argument("ODE solver: need n >= 1 steps");
  OdeSolution s{std::vector<double>(n + 1), std::vector<double>(n + 1)};
  const double h = (x_end - x0) / static_cast<double>(n);
  for (std::size_t k = 0; k <= n; ++k) s.x[k] = x0 + static_cast<double>(k) * h;
  s.x[n] = x_end;
  return s;
}
}  // namespace detail

// ---------------------------------------------------------------- 7.2 ------
template <std::invocable<double, double> F>
[[nodiscard]] OdeSolution euler_forward(F&& f, double x0, double y0,
                                        double x_end, std::size_t n) {
  auto s = detail::make_grid(x0, x_end, n);
  const double h = (x_end - x0) / static_cast<double>(n);
  s.y[0] = y0;
  for (std::size_t k = 0; k < n; ++k)
    s.y[k + 1] = s.y[k] + h * f(s.x[k], s.y[k]);
  return s;
}

// Implicit Euler: solves g(y) = y - y_k - h f(x_{k+1}, y) = 0 for y_{k+1}
// by Newton's method (chapter 5) from the explicit Euler guess, with dg/dy by
// a central difference (chapter 1).
template <std::invocable<double, double> F>
[[nodiscard]] OdeSolution euler_backward(F&& f, double x0, double y0,
                                         double x_end, std::size_t n) {
  auto s = detail::make_grid(x0, x_end, n);
  const double h = (x_end - x0) / static_cast<double>(n);
  s.y[0] = y0;
  for (std::size_t k = 0; k < n; ++k) {
    const double yk = s.y[k];
    const double x1 = s.x[k + 1];
    const auto g = [&](double y) { return y - yk - h * f(x1, y); };
    const auto dg = [&](double y) { return ch01::Derivative(g, y); };
    const double scale = std::max(1.0, std::abs(yk));
    const ch05::Tolerances tol{1e-12 * scale, 1e-12 * scale, 100};
    s.y[k + 1] = ch05::newton(g, dg, yk + h * f(s.x[k], yk), tol).x;
  }
  return s;
}

template <std::invocable<double, double> F>
[[nodiscard]] OdeSolution leapfrog(F&& f, double x0, double y0, double x_end,
                                   std::size_t n) {
  auto s = detail::make_grid(x0, x_end, n);
  const double h = (x_end - x0) / static_cast<double>(n);
  s.y[0] = y0;
  s.y[1] = y0 + h * f(x0, y0);
  for (std::size_t k = 1; k < n; ++k)
    s.y[k + 1] = s.y[k - 1] + 2.0 * h * f(s.x[k], s.y[k]);
  return s;
}

template <std::invocable<double, double> F>
[[nodiscard]] OdeSolution solve_ivp(F&& f, double x0, double y0, double x_end,
                                    std::size_t n, OdeScheme scheme) {
  switch (scheme) {
    case OdeScheme::Forward:
      return euler_forward(f, x0, y0, x_end, n);
    case OdeScheme::Backward:
      return euler_backward(f, x0, y0, x_end, n);
    case OdeScheme::Central:
      return leapfrog(f, x0, y0, x_end, n);
  }
  throw std::invalid_argument("solve_ivp: unknown scheme");
}

// Thomas algorithm for a tridiagonal system
//   lower[i] x[i-1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i],  i = 0..n-1
// (lower[0] and upper[n-1] are ignored). Gaussian elimination without
// pivoting; fine for diagonally dominant systems such as those below.
[[nodiscard]] inline std::vector<double> solve_tridiagonal(
    const std::vector<double>& lower, std::vector<double> diag,
    const std::vector<double>& upper, std::vector<double> rhs) {
  const std::size_t n = diag.size();
  if (n == 0 || lower.size() != n || upper.size() != n || rhs.size() != n)
    throw std::invalid_argument("solve_tridiagonal: size mismatch");
  for (std::size_t i = 1; i < n; ++i) {
    if (diag[i - 1] == 0.0)
      throw std::runtime_error("solve_tridiagonal: zero pivot");
    const double m = lower[i] / diag[i - 1];
    diag[i] -= m * upper[i - 1];
    rhs[i] -= m * rhs[i - 1];
  }
  if (diag[n - 1] == 0.0)
    throw std::runtime_error("solve_tridiagonal: zero pivot");
  std::vector<double> x(n);
  x[n - 1] = rhs[n - 1] / diag[n - 1];
  for (std::size_t i = n - 1; i-- > 0;)
    x[i] = (rhs[i] - upper[i] * x[i + 1]) / diag[i];
  return x;
}

// y'' + p(x) y' + q(x) y = r(x), y(a) = ya, y(b) = yb, on n equal intervals.
// At interior node i:
//   (1/h^2 - p_i/(2h)) y_{i-1} + (q_i - 2/h^2) y_i + (1/h^2 + p_i/(2h)) y_{i+1}
//   = r_i.
template <std::invocable<double> P, std::invocable<double> Q,
          std::invocable<double> R>
[[nodiscard]] OdeSolution solve_bvp(P&& p, Q&& q, R&& r, double a, double b,
                                    double ya, double yb, std::size_t n) {
  if (n < 2) throw std::invalid_argument("solve_bvp: need n >= 2 intervals");
  auto s = detail::make_grid(a, b, n);
  const double h = (b - a) / static_cast<double>(n);
  const std::size_t m = n - 1;  // unknowns y_1..y_{n-1}
  std::vector<double> lo(m), di(m), up(m), rhs(m);
  for (std::size_t j = 0; j < m; ++j) {
    const double x = s.x[j + 1];
    const double pi = p(x);
    lo[j] = 1.0 / (h * h) - pi / (2.0 * h);
    di[j] = q(x) - 2.0 / (h * h);
    up[j] = 1.0 / (h * h) + pi / (2.0 * h);
    rhs[j] = r(x);
  }
  rhs[0] -= lo[0] * ya;
  rhs[m - 1] -= up[m - 1] * yb;
  const auto inner = solve_tridiagonal(lo, std::move(di), up, std::move(rhs));
  s.y[0] = ya;
  s.y[n] = yb;
  std::copy(inner.begin(), inner.end(), s.y.begin() + 1);
  return s;
}

// Largest |y_k - y(x_k)| over the grid.
template <std::invocable<double> Exact>
[[nodiscard]] double max_error(const OdeSolution& s, Exact&& exact) {
  double e = 0.0;
  for (std::size_t k = 0; k < s.x.size(); ++k)
    e = std::max(e, std::abs(s.y[k] - exact(s.x[k])));
  return e;
}

}  // namespace p4mfe::ch07
