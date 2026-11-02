#pragma once
// Chapter 10, section 10.2: polynomial interpolation.
//
// Given n+1 distinct nodes x_0, ..., x_n and values y_0, ..., y_n there is a
// unique polynomial p of degree at most n with p(x_i) = y_i. Three ways to
// write it:
//   Lagrange:  p(x) = sum_i y_i L_i(x),  L_i(x) = prod_{j != i} (x - x_j) /
//   (x_i - x_j) Newton:    p(x) = c_0 + c_1 (x - x_0) + ... + c_n (x -
//   x_0)...(x - x_{n-1}),
//              c_k = f[x_0, ..., x_k] (divided differences)
//   monomial:  p(x) = sum a_k x^k, from the Vandermonde system V a = y
//              (solved with chapter 5's Gaussian elimination).
// If y_i = f(x_i) with f n+1 times differentiable, the error is
//   f(x) - p(x) = f^(n+1)(xi) / (n+1)! prod_i (x - x_i),
// so |f(x) - p(x)| <= M / (n+1)! |prod_i (x - x_i)| when |f^(n+1)| <= M.

#include <cmath>
#include <cstddef>
#include <p4mfe/ch05/newton_nd.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch10 {

using Vector = std::vector<double>;

namespace detail {
inline void check_nodes(std::span<const double> xs, std::size_t ny) {
  if (xs.empty() || xs.size() != ny)
    throw std::invalid_argument("interpolation: need as many values as nodes");
  for (std::size_t i = 0; i < xs.size(); ++i)
    for (std::size_t j = i + 1; j < xs.size(); ++j)
      if (xs[i] == xs[j])
        throw std::invalid_argument("interpolation: nodes must be distinct");
}
}  // namespace detail

// sum_k a_k x^k by Horner's rule (a_0 first).
[[nodiscard]] inline double polynomial_value(std::span<const double> a,
                                             double x) noexcept {
  double v = 0.0;
  for (std::size_t k = a.size(); k-- > 0;) v = v * x + a[k];
  return v;
}

// The Lagrange basis polynomial L_i(x): 1 at x_i, 0 at the other nodes.
[[nodiscard]] inline double lagrange_basis(std::span<const double> xs,
                                           std::size_t i, double x) {
  if (i >= xs.size())
    throw std::out_of_range("lagrange_basis: index out of range");
  double v = 1.0;
  for (std::size_t j = 0; j < xs.size(); ++j)
    if (j != i) v *= (x - xs[j]) / (xs[i] - xs[j]);
  return v;
}

[[nodiscard]] inline double lagrange_interpolate(std::span<const double> xs,
                                                 std::span<const double> ys,
                                                 double x) {
  detail::check_nodes(xs, ys.size());
  double v = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i)
    v += ys[i] * lagrange_basis(xs, i, x);
  return v;
}

// Newton coefficients c_k = f[x_0, ..., x_k], computed in place:
// f[x_i..x_{i+k}] = (f[x_{i+1}..x_{i+k}] - f[x_i..x_{i+k-1}]) / (x_{i+k} -
// x_i).
[[nodiscard]] inline Vector divided_differences(std::span<const double> xs,
                                                std::span<const double> ys) {
  detail::check_nodes(xs, ys.size());
  Vector c(ys.begin(), ys.end());
  for (std::size_t k = 1; k < xs.size(); ++k)
    for (std::size_t i = xs.size() - 1; i >= k; --i)
      c[i] = (c[i] - c[i - 1]) / (xs[i] - xs[i - k]);
  return c;
}

// Evaluates the Newton form by nested multiplication.
[[nodiscard]] inline double newton_interpolate(std::span<const double> xs,
                                               std::span<const double> c,
                                               double x) {
  if (c.empty() || c.size() > xs.size())
    throw std::invalid_argument("newton_interpolate: size mismatch");
  double v = c.back();
  for (std::size_t k = c.size() - 1; k-- > 0;) v = v * (x - xs[k]) + c[k];
  return v;
}

// Monomial coefficients a_0..a_n of the interpolating polynomial (Vandermonde
// system). Fine for small n; the Vandermonde matrix is ill-conditioned.
[[nodiscard]] inline Vector interpolating_polynomial(
    std::span<const double> xs, std::span<const double> ys) {
  detail::check_nodes(xs, ys.size());
  const std::size_t n = xs.size();
  ch05::Matrix V(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    double p = 1.0;
    for (std::size_t j = 0; j < n; ++j, p *= xs[i]) V(i, j) = p;
  }
  return ch05::SolveLinear(std::move(V), Vector(ys.begin(), ys.end()));
}

// M / (n+1)! |prod_i (x - x_i)|, the interpolation error bound when |f^(n+1)|
// <= M.
[[nodiscard]] inline double interpolation_error_bound(
    double M, std::span<const double> xs, double x) {
  double prod = 1.0;
  for (double xi : xs) prod *= x - xi;
  return M * std::abs(prod) / std::tgamma(static_cast<double>(xs.size()) + 1.0);
}

}  // namespace p4mfe::ch10
