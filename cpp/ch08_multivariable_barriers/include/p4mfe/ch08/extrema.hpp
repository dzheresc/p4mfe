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

#include <p4mfe/ch01/multivariable.hpp>
#include <p4mfe/ch05/newton_nd.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch08 {

enum class CriticalPointType { Minimum, Maximum, Saddle, Degenerate };

// Determinant by Gaussian elimination with partial pivoting.
[[nodiscard]] inline double determinant(ch01::Matrix A) {
    const std::size_t n = A.size();
    double det = 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::abs(A[i][k]) > std::abs(A[p][k])) p = i;
        if (A[p][k] == 0.0) return 0.0;
        if (p != k) {
            std::swap(A[p], A[k]);
            det = -det;
        }
        det *= A[k][k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const double m = A[i][k] / A[k][k];
            for (std::size_t j = k; j < n; ++j) A[i][j] -= m * A[k][j];
        }
    }
    return det;
}

// D_1, ..., D_n: determinants of the upper-left k x k blocks.
[[nodiscard]] inline std::vector<double> leading_principal_minors(const ch01::Matrix& H) {
    std::vector<double> D;
    for (std::size_t k = 1; k <= H.size(); ++k) {
        ch01::Matrix B(k, ch01::Vector(k));
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) B[i][j] = H[i][j];
        D.push_back(determinant(std::move(B)));
    }
    return D;
}

// ---------------------------------------------------------------- 8.4 ------
// `tol` treats |D_k| <= tol as zero (Hessians from finite differences are
// only accurate to ~1e-7).
[[nodiscard]] inline CriticalPointType classify_hessian(const ch01::Matrix& H, double tol = 1e-6) {
    if (H.empty()) throw std::invalid_argument("classify_hessian: empty matrix");
    const auto D = leading_principal_minors(H);
    if (std::abs(D.back()) <= tol) return CriticalPointType::Degenerate;
    bool pos = true, neg = true;
    for (std::size_t k = 0; k < D.size(); ++k) {
        pos = pos && D[k] > tol;
        neg = neg && (k % 2 == 0 ? D[k] < -tol : D[k] > tol);
    }
    if (pos) return CriticalPointType::Minimum;
    if (neg) return CriticalPointType::Maximum;
    return CriticalPointType::Saddle;
}

template <class F>
[[nodiscard]] CriticalPointType classify_critical_point(F&& f, const ch01::Vector& x, double tol = 1e-6) {
    return classify_hessian(ch01::hessian(f, x), tol);
}

// Solves grad f(x) = 0 by Newton's method from x0.
template <class F>
[[nodiscard]] ch01::Vector find_critical_point(F&& f, ch01::Vector x0, const ch05::Tolerances& tol = {1e-8, 1e-10, 100}) {
    const auto grad = [&](const ch05::Vector& x) { return ch01::gradient(f, x); };
    const auto hess = [&](const ch05::Vector& x) {
        const auto H = ch01::hessian(f, x);
        ch05::Matrix M(H.size(), H.size());
        for (std::size_t i = 0; i < H.size(); ++i)
            for (std::size_t j = 0; j < H.size(); ++j) M(i, j) = H[i][j];
        return M;
    };
    return ch05::newton_nd(grad, hess, std::move(x0), tol).x;
}

}  // namespace p4mfe::ch08
