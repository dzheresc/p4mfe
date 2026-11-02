#pragma once
// Chapter 1, section 1.6: functions of several variables.
//
// Partial derivatives, the gradient and the Hessian by central differences,
// for f : R^n -> R taking its argument as const std::vector<double>&.
// Section 1.6.1 (two variables) is the n = 2 case.

#include <cstddef>
#include <p4mfe/ch01/calculus.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch01 {

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// df/dx_i at x.
template <typename F>
[[nodiscard]] double partial(F&& f, const Vector& x, std::size_t i,
                             double h = 0.0) {
  if (i >= x.size()) throw std::out_of_range("partial: index out of range");
  Vector y = x;
  return derivative(
      [&](double xi) {
        y[i] = xi;
        return f(y);
      },
      x[i], h);
}

// d^2 f / dx_i dx_j at x. Diagonal terms use the 3-point second difference,
// mixed terms the 4-point cross difference
//   (f(+h,+h) - f(+h,-h) - f(-h,+h) + f(-h,-h)) / (4 h^2).
template <typename F>
[[nodiscard]] double second_partial(F&& f, const Vector& x, std::size_t i,
                                    std::size_t j, double h = 0.0) {
  if (i >= x.size() || j >= x.size())
    throw std::out_of_range("second_partial: index out of range");
  Vector y = x;
  if (i == j) {
    return second_derivative(
        [&](double xi) {
          y[i] = xi;
          return f(y);
        },
        x[i], h);
  }
  const double hi = h > 0.0 ? h : detail::second_diff_step(x[i]);
  const double hj = h > 0.0 ? h : detail::second_diff_step(x[j]);
  const auto at = [&](double si, double sj) {
    y[i] = x[i] + si * hi;
    y[j] = x[j] + sj * hj;
    return f(y);
  };
  return (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (4.0 * hi * hj);
}

// Gradient: the vector of first partial derivatives.
template <typename F>
[[nodiscard]] Vector gradient(F&& f, const Vector& x) {
  Vector g(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) g[i] = partial(f, x, i);
  return g;
}

// Hessian: the symmetric matrix of second partial derivatives.
template <typename F>
[[nodiscard]] Matrix hessian(F&& f, const Vector& x) {
  const std::size_t n = x.size();
  Matrix H(n, Vector(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j)
      H[i][j] = H[j][i] = second_partial(f, x, i, j);
  }
  return H;
}

}  // namespace p4mfe::ch01
