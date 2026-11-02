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
[[nodiscard]] Vector curve_derivative(G&& g, double t) {
  const std::size_t n = g(t).size();
  Vector d(n);
  for (std::size_t i = 0; i < n; ++i)
    d[i] = ch01::Derivative([&](double s) { return g(s)[i]; }, t);
  return d;
}

// d/dt f(g(t)) = grad f(g(t)) . g'(t).
template <class F, std::invocable<double> G>
[[nodiscard]] double chain_rule_derivative(F&& f, G&& g, double t) {
  const Vector x = g(t);
  const Vector grad = ch01::Gradient(f, x);
  const Vector dg = curve_derivative(g, t);
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += grad[i] * dg[i];
  return s;
}

// grad (f o G)(x) = J_G(x)^T grad f(G(x)).
template <class F, ch05::VectorFunction G>
[[nodiscard]] Vector chain_rule_gradient(F&& f, G&& G_, const Vector& x) {
  const Matrix J =
      ch05::FiniteDifferenceJacobian(G_, x, 1e-6, ch05::Difference::kCentral);
  const Vector grad = ch01::Gradient(f, G_(x));
  Vector out(x.size(), 0.0);
  for (std::size_t j = 0; j < J.Cols(); ++j)
    for (std::size_t i = 0; i < J.Rows(); ++i) out[j] += J(i, j) * grad[i];
  return out;
}

[[nodiscard]] inline Matrix multiply(const Matrix& A, const Matrix& B) {
  if (A.Cols() != B.Rows())
    throw std::invalid_argument("multiply: inner dimensions differ");
  Matrix C(A.Rows(), B.Cols());
  for (std::size_t i = 0; i < A.Rows(); ++i)
    for (std::size_t k = 0; k < A.Cols(); ++k)
      for (std::size_t j = 0; j < B.Cols(); ++j) C(i, j) += A(i, k) * B(k, j);
  return C;
}

// J_(F o G)(x) = J_F(G(x)) J_G(x).
template <ch05::VectorFunction F, ch05::VectorFunction G>
[[nodiscard]] Matrix chain_rule_jacobian(F&& F_, G&& G_, const Vector& x) {
  return multiply(
      ch05::FiniteDifferenceJacobian(F_, G_(x), 1e-6,
                                     ch05::Difference::kCentral),
      ch05::FiniteDifferenceJacobian(G_, x, 1e-6, ch05::Difference::kCentral));
}

}  // namespace p4mfe::ch08
