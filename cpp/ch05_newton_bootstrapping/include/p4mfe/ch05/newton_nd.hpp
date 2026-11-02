#pragma once
// Chapter 5, section 5.2: numerical methods for N-dimensional problems
// F(x) = 0, with F: R^N -> R^N.
//
// 5.2.1 Newton's method linearizes F around x_k,
//   F(x) ~ F(x_k) + D_F(x_k) (x - x_k),
// and sets the linearization to zero:
//   x_{k+1} = x_k - D_F(x_k)^{-1} F(x_k).
// D_F is the Jacobian, (D_F)_{ij} = dF_i / dx_j. The inverse is never formed:
// each step solves the linear system D_F(x_k) s = -F(x_k) (Gaussian
// elimination with partial pivoting) and sets x_{k+1} = x_k + s.
//
// 5.2.2 The approximate Newton's method replaces D_F by finite differences,
//   (D_F)_{ij} ~ (F_i(x + h e_j) - F_i(x)) / h,
// costing N extra evaluations of F per step, for when the partial derivatives
// are unknown or expensive.
//
// Stopping mirrors section 5.1 with Euclidean norms:
//   ||F(x_k)|| <= tol.approx  and  ||x_k - x_{k-1}|| <= tol.consec.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <p4mfe/ch05/solvers.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch05 {

using Vector = std::vector<double>;

// Dense row-major N x M matrix; just enough for Jacobians.
class Matrix {
 public:
  Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
  Matrix(std::initializer_list<std::initializer_list<double>> rows)
      : rows_(rows.size()), cols_(0) {
    if (rows_ > 0) cols_ = rows.begin()->size();
    for (const auto& r : rows) {
      if (r.size() != cols_) throw std::invalid_argument("Matrix: ragged rows");
      data_.insert(data_.end(), r.begin(), r.end());
    }
  }
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept {
    return data_[i * cols_ + j];
  }
  [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept {
    return data_[i * cols_ + j];
  }

 private:
  std::size_t rows_, cols_;
  std::vector<double> data_;
};

[[nodiscard]] inline double norm2(const Vector& v) noexcept {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

// Solves A x = b by Gaussian elimination with partial pivoting.
[[nodiscard]] inline Vector solve_linear(Matrix A, Vector b) {
  const std::size_t n = A.rows();
  if (A.cols() != n || b.size() != n)
    throw std::invalid_argument("solve_linear: need a square system");
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::abs(A(i, k)) > std::abs(A(p, k))) p = i;
    if (A(p, k) == 0.0)
      throw std::runtime_error("solve_linear: singular matrix");
    if (p != k) {
      for (std::size_t j = k; j < n; ++j) std::swap(A(k, j), A(p, j));
      std::swap(b[k], b[p]);
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      const double m = A(i, k) / A(k, k);
      for (std::size_t j = k; j < n; ++j) A(i, j) -= m * A(k, j);
      b[i] -= m * b[k];
    }
  }
  Vector x(n);
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= A(i, j) * x[j];
    x[i] = s / A(i, i);
  }
  return x;
}

template <class F>
concept VectorFunction =
    std::invocable<F, const Vector&> &&
    std::convertible_to<std::invoke_result_t<F, const Vector&>, Vector>;

template <class J>
concept JacobianFunction =
    std::invocable<J, const Vector&> &&
    std::convertible_to<std::invoke_result_t<J, const Vector&>, Matrix>;

struct NdRootResult {
  Vector x;
  std::size_t iterations;
};

// --------------------------------------------------------------- 5.2.1 -----
template <VectorFunction F, JacobianFunction J>
[[nodiscard]] NdRootResult newton_nd(F&& F_, J&& jacobian, Vector x0,
                                     const Tolerances& tol = {}) {
  NdRootResult res{std::move(x0), 0};
  Vector fx = F_(res.x);
  if (fx.size() != res.x.size())
    throw std::invalid_argument("newton_nd: F must map R^N to R^N");
  while (res.iterations < tol.max_iter) {
    Vector rhs(fx.size());
    std::transform(fx.begin(), fx.end(), rhs.begin(),
                   [](double v) { return -v; });
    const Vector s = solve_linear(jacobian(res.x), std::move(rhs));
    for (std::size_t i = 0; i < s.size(); ++i) res.x[i] += s[i];
    ++res.iterations;
    fx = F_(res.x);
    const double step = norm2(s);
    if ((norm2(fx) <= tol.approx && step <= tol.consec) || step == 0.0)
      return res;
  }
  throw std::runtime_error("newton_nd: no convergence within max_iter");
}

// --------------------------------------------------------------- 5.2.2 -----
enum class Difference { Forward, Central };

// Finite difference Jacobian of F at x with step h: forward differences are
// O(h) accurate (the book's choice), central ones O(h^2) at twice the cost.
template <VectorFunction F>
[[nodiscard]] Matrix finite_difference_jacobian(
    F&& F_, const Vector& x, double h = 1e-6,
    Difference scheme = Difference::Forward) {
  const Vector f0 = F_(x);
  Matrix J(f0.size(), x.size());
  Vector xp = x;
  for (std::size_t j = 0; j < x.size(); ++j) {
    xp[j] = x[j] + h;
    const Vector fp = F_(xp);
    if (scheme == Difference::Forward) {
      for (std::size_t i = 0; i < f0.size(); ++i) J(i, j) = (fp[i] - f0[i]) / h;
    } else {
      xp[j] = x[j] - h;
      const Vector fm = F_(xp);
      for (std::size_t i = 0; i < f0.size(); ++i)
        J(i, j) = (fp[i] - fm[i]) / (2.0 * h);
    }
    xp[j] = x[j];
  }
  return J;
}

template <VectorFunction F>
[[nodiscard]] NdRootResult approximate_newton_nd(
    F&& F_, Vector x0, const Tolerances& tol = {}, double h = 1e-6,
    Difference scheme = Difference::Forward) {
  return newton_nd(
      F_,
      [&](const Vector& x) {
        return finite_difference_jacobian(F_, x, h, scheme);
      },
      std::move(x0), tol);
}

}  // namespace p4mfe::ch05
