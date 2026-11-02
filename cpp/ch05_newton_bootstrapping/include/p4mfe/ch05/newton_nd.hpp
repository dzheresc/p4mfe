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
  [[nodiscard]] std::size_t Rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t Cols() const noexcept { return cols_; }
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

[[nodiscard]] inline double Norm2(const Vector& v) noexcept {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

// Solves A x = b by Gaussian elimination with partial pivoting.
[[nodiscard]] inline Vector SolveLinear(Matrix a, Vector b) {
  const std::size_t n = a.Rows();
  if (a.Cols() != n || b.size() != n)
    throw std::invalid_argument("solve_linear: need a square system");
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::abs(a(i, k)) > std::abs(a(p, k))) p = i;
    if (a(p, k) == 0.0)
      throw std::runtime_error("solve_linear: singular matrix");
    if (p != k) {
      for (std::size_t j = k; j < n; ++j) std::swap(a(k, j), a(p, j));
      std::swap(b[k], b[p]);
    }
    for (std::size_t i = k + 1; i < n; ++i) {
      const double m = a(i, k) / a(k, k);
      for (std::size_t j = k; j < n; ++j) a(i, j) -= m * a(k, j);
      b[i] -= m * b[k];
    }
  }
  Vector x(n);
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) s -= a(i, j) * x[j];
    x[i] = s / a(i, i);
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
[[nodiscard]] NdRootResult NewtonNd(F&& func, J&& jacobian, Vector x0,
                                    const Tolerances& tol = {}) {
  NdRootResult res{std::move(x0), 0};
  Vector fx = func(res.x);
  if (fx.size() != res.x.size())
    throw std::invalid_argument("newton_nd: F must map R^N to R^N");
  while (res.iterations < tol.max_iter) {
    Vector rhs(fx.size());
    std::transform(fx.begin(), fx.end(), rhs.begin(),
                   [](double v) { return -v; });
    const Vector s = SolveLinear(jacobian(res.x), std::move(rhs));
    for (std::size_t i = 0; i < s.size(); ++i) res.x[i] += s[i];
    ++res.iterations;
    fx = func(res.x);
    const double step = Norm2(s);
    if ((Norm2(fx) <= tol.approx && step <= tol.consec) || step == 0.0)
      return res;
  }
  throw std::runtime_error("newton_nd: no convergence within max_iter");
}

// --------------------------------------------------------------- 5.2.2 -----
enum class Difference { kForward, kCentral };

// Finite difference Jacobian of F at x with step h: forward differences are
// O(h) accurate (the book's choice), central ones O(h^2) at twice the cost.
template <VectorFunction F>
[[nodiscard]] Matrix FiniteDifferenceJacobian(
    F&& func, const Vector& x, double h = 1e-6,
    Difference scheme = Difference::kForward) {
  const Vector f0 = func(x);
  Matrix jacobian(f0.size(), x.size());
  Vector xp = x;
  for (std::size_t j = 0; j < x.size(); ++j) {
    xp[j] = x[j] + h;
    const Vector fp = func(xp);
    if (scheme == Difference::kForward) {
      for (std::size_t i = 0; i < f0.size(); ++i)
        jacobian(i, j) = (fp[i] - f0[i]) / h;
    } else {
      xp[j] = x[j] - h;
      const Vector fm = func(xp);
      for (std::size_t i = 0; i < f0.size(); ++i)
        jacobian(i, j) = (fp[i] - fm[i]) / (2.0 * h);
    }
    xp[j] = x[j];
  }
  return jacobian;
}

template <VectorFunction F>
[[nodiscard]] NdRootResult ApproximateNewtonNd(
    F&& func, Vector x0, const Tolerances& tol = {}, double h = 1e-6,
    Difference scheme = Difference::kForward) {
  return NewtonNd(
      func,
      [&](const Vector& x) {
        return FiniteDifferenceJacobian(func, x, h, scheme);
      },
      std::move(x0), tol);
}

}  // namespace p4mfe::ch05
