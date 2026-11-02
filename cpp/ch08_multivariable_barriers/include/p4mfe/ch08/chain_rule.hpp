#pragma once
// Chapter 8, section 8.1: the chain rule for functions of several variables.
//
// For f : R^n -> R and a curve g : R -> R^n,
//   d/dt f(g(t)) = sum_i df/dx_i(g(t)) g_i'(t) = grad f(g(t)) . g'(t).
// For G : R^m -> R^n,
//   grad (f o G)(x) = J_G(x)^T grad f(G(x)),
// and for F : R^n -> R^k,  J_(F o G)(x) = J_F(G(x)) J_G(x).
// Gradients come from chapter 1, Jacobians from chapter 5 (central
// differences); these functions let you check a hand-derived chain rule.

#include <concepts>
#include <cstddef>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/multivariable.hpp>
#include <p4mfe/ch05/newton_nd.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch08 {

using Vector = std::vector<double>;
using ch05::Matrix;

// ---------------------------------------------------------------- 8.1 ------
// g'(t), componentwise central differences.
template <std::invocable<double> G>
[[nodiscard]] Vector CurveDerivative(G&& g, double t) {
  const std::size_t n = g(t).size();
  Vector d(n);
  for (std::size_t i = 0; i < n; ++i)
    d[i] = ch01::Derivative([&](double s) { return g(s)[i]; }, t);
  return d;
}

// d/dt f(g(t)) = grad f(g(t)) . g'(t).
template <class F, std::invocable<double> G>
[[nodiscard]] double ChainRuleDerivative(F&& f, G&& g, double t) {
  const Vector x = g(t);
  const Vector grad = ch01::Gradient(f, x);
  const Vector dg = CurveDerivative(g, t);
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += grad[i] * dg[i];
  return s;
}

// grad (f o G)(x) = J_G(x)^T grad f(G(x)).
template <class F, ch05::VectorFunction G>
[[nodiscard]] Vector ChainRuleGradient(F&& f, G&& g, const Vector& x) {
  const Matrix jacobian =
      ch05::FiniteDifferenceJacobian(g, x, 1e-6, ch05::Difference::kCentral);
  const Vector grad = ch01::Gradient(f, g(x));
  Vector out(x.size(), 0.0);
  for (std::size_t j = 0; j < jacobian.Cols(); ++j)
    for (std::size_t i = 0; i < jacobian.Rows(); ++i)
      out[j] += jacobian(i, j) * grad[i];
  return out;
}

[[nodiscard]] inline Matrix Multiply(const Matrix& lhs, const Matrix& rhs) {
  if (lhs.Cols() != rhs.Rows())
    throw std::invalid_argument("multiply: inner dimensions differ");
  Matrix product(lhs.Rows(), rhs.Cols());
  for (std::size_t i = 0; i < lhs.Rows(); ++i)
    for (std::size_t k = 0; k < lhs.Cols(); ++k)
      for (std::size_t j = 0; j < rhs.Cols(); ++j)
        product(i, j) += lhs(i, k) * rhs(k, j);
  return product;
}

// J_(F o G)(x) = J_F(G(x)) J_G(x).
template <ch05::VectorFunction F, ch05::VectorFunction G>
[[nodiscard]] Matrix ChainRuleJacobian(F&& f, G&& g, const Vector& x) {
  return Multiply(
      ch05::FiniteDifferenceJacobian(f, g(x), 1e-6, ch05::Difference::kCentral),
      ch05::FiniteDifferenceJacobian(g, x, 1e-6, ch05::Difference::kCentral));
}

}  // namespace p4mfe::ch08
