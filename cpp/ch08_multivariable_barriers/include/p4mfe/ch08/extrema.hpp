#pragma once
// Chapter 8, section 8.4: relative extrema of functions of several variables.
//
// A critical point x0 has grad f(x0) = 0. With H the Hessian at x0:
//   H positive definite  -> local minimum
//   H negative definite  -> local maximum
//   H nonsingular, neither -> saddle point
//   H singular            -> the second-order test is inconclusive.
// Definiteness is decided by Sylvester's criterion on the leading principal
// minors D_1, ..., D_n: all D_k > 0 (positive definite), or (-1)^k D_k > 0
// (negative definite). For two variables this is the book's test with
// D_2 = f_xx f_yy - f_xy^2 and D_1 = f_xx.
//
// Critical points are found by Newton's method (chapter 5) on grad f = 0,
// whose Jacobian is the Hessian (both by finite differences, chapter 1).

#include <cmath>
#include <cstddef>
#include <p4mfe/ch01/multivariable.hpp>
#include <p4mfe/ch05/newton_nd.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch08 {

enum class CriticalPointType { kMinimum, kMaximum, kSaddle, kDegenerate };

// Determinant by Gaussian elimination with partial pivoting.
[[nodiscard]] inline double Determinant(ch01::Matrix matrix) {
  const std::size_t n = matrix.size();
  double det = 1.0;
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::abs(matrix[i][k]) > std::abs(matrix[p][k])) p = i;
    if (matrix[p][k] == 0.0) return 0.0;
    if (p != k) {
      std::swap(matrix[p], matrix[k]);
      det = -det;
    }
    det *= matrix[k][k];
    for (std::size_t i = k + 1; i < n; ++i) {
      const double m = matrix[i][k] / matrix[k][k];
      for (std::size_t j = k; j < n; ++j) matrix[i][j] -= m * matrix[k][j];
    }
  }
  return det;
}

// D_1, ..., D_n: determinants of the upper-left k x k blocks.
[[nodiscard]] inline std::vector<double> LeadingPrincipalMinors(
    const ch01::Matrix& hessian) {
  std::vector<double> minors;
  for (std::size_t k = 1; k <= hessian.size(); ++k) {
    ch01::Matrix block(k, ch01::Vector(k));
    for (std::size_t i = 0; i < k; ++i)
      for (std::size_t j = 0; j < k; ++j) block[i][j] = hessian[i][j];
    minors.push_back(Determinant(std::move(block)));
  }
  return minors;
}

// ---------------------------------------------------------------- 8.4 ------
// `tol` treats |D_k| <= tol as zero (Hessians from finite differences are
// only accurate to ~1e-7).
[[nodiscard]] inline CriticalPointType ClassifyHessian(
    const ch01::Matrix& hessian, double tol = 1e-6) {
  if (hessian.empty())
    throw std::invalid_argument("classify_hessian: empty matrix");
  const auto minors = LeadingPrincipalMinors(hessian);
  if (std::abs(minors.back()) <= tol) return CriticalPointType::kDegenerate;
  bool pos = true, neg = true;
  for (std::size_t k = 0; k < minors.size(); ++k) {
    pos = pos && minors[k] > tol;
    neg = neg && (k % 2 == 0 ? minors[k] < -tol : minors[k] > tol);
  }
  if (pos) return CriticalPointType::kMinimum;
  if (neg) return CriticalPointType::kMaximum;
  return CriticalPointType::kSaddle;
}

template <class F>
[[nodiscard]] CriticalPointType ClassifyCriticalPoint(F&& f,
                                                      const ch01::Vector& x,
                                                      double tol = 1e-6) {
  return ClassifyHessian(ch01::Hessian(f, x), tol);
}

// Solves grad f(x) = 0 by Newton's method from x0.
template <class F>
[[nodiscard]] ch01::Vector FindCriticalPoint(F&& f, ch01::Vector x0,
                                             const ch05::Tolerances& tol = {
                                                 1e-8, 1e-10, 100}) {
  const auto grad = [&](const ch05::Vector& x) { return ch01::Gradient(f, x); };
  const auto hess = [&](const ch05::Vector& x) {
    const auto hessian = ch01::Hessian(f, x);
    ch05::Matrix hessian_matrix(hessian.size(), hessian.size());
    for (std::size_t i = 0; i < hessian.size(); ++i)
      for (std::size_t j = 0; j < hessian.size(); ++j)
        hessian_matrix(i, j) = hessian[i][j];
    return hessian_matrix;
  };
  return ch05::NewtonNd(grad, hess, std::move(x0), tol).x;
}

}  // namespace p4mfe::ch08
