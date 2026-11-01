#pragma once
// Chapter 9, section 9.1: Lagrange multipliers.
//
// To find the extrema of f : R^n -> R subject to m constraints g(x) = 0,
// g : R^n -> R^m (m < n), form the Lagrangian
//   L(x, lambda) = f(x) + lambda . g(x).
// At a constrained extremum where the constraint gradients are linearly
// independent there is a lambda with
//   grad f(x) + sum_j lambda_j grad g_j(x) = 0,   g(x) = 0,
// n + m equations in n + m unknowns. They are solved here by Newton's method
// (chapter 5); the Jacobian of the system is the bordered matrix
//   [ H_L(x, lambda)   grad g(x)^T ]
//   [ grad g(x)        0           ],   H_L = H_f + sum_j lambda_j H_(g_j).
//
// Second-order test: restrict H_L to the tangent space of the constraints,
// {v : grad g_j(x) . v = 0}. If Z is a basis of it, Z^T H_L Z positive
// (negative) definite gives a constrained minimum (maximum); chapter 8's
// classify_hessian does the rest.

#include <p4mfe/ch01/multivariable.hpp>
#include <p4mfe/ch05/newton_nd.hpp>
#include <p4mfe/ch08/extrema.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch09 {

using Vector = std::vector<double>;
using ch08::CriticalPointType;

namespace detail {
template <class G>
auto component(G& g, std::size_t j) {
    return [&g, j](const Vector& x) { return g(x)[j]; };
}

// Orthonormal basis of {v : A v = 0} for the m x n matrix A given as rows
// (Gram-Schmidt: orthonormalize the rows, then the unit vectors against them).
inline std::vector<Vector> null_space(const std::vector<Vector>& rows, std::size_t n, double tol = 1e-10) {
    const auto dot = [](const Vector& a, const Vector& b) {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    };
    std::vector<Vector> basis;  // orthonormal, row space first
    const auto add = [&](Vector v) {
        for (const auto& b : basis) {
            const double c = dot(v, b);
            for (std::size_t i = 0; i < n; ++i) v[i] -= c * b[i];
        }
        const double nv = std::sqrt(dot(v, v));
        if (nv <= tol) return false;
        for (double& vi : v) vi /= nv;
        basis.push_back(std::move(v));
        return true;
    };
    for (const auto& r : rows)
        if (!add(r)) throw std::invalid_argument("lagrange: constraint gradients are linearly dependent");
    const std::size_t m = basis.size();
    for (std::size_t k = 0; k < n && basis.size() < n; ++k) {
        Vector e(n, 0.0);
        e[k] = 1.0;
        add(std::move(e));
    }
    return {basis.begin() + static_cast<std::ptrdiff_t>(m), basis.end()};
}
}  // namespace detail

struct LagrangeResult {
    Vector x;
    Vector lambda;
    std::size_t iterations;
};

// ---------------------------------------------------------------- 9.1 ------
// Gradient of the Lagrangian in (x, lambda): (grad f + sum lambda_j grad g_j, g).
template <class F, ch05::VectorFunction G>
[[nodiscard]] Vector lagrangian_gradient(F&& f, G&& g, const Vector& x, const Vector& lambda) {
    Vector out = ch01::gradient(f, x);
    const Vector gx = g(x);
    for (std::size_t j = 0; j < gx.size(); ++j) {
        const Vector gj = ch01::gradient(detail::component(g, j), x);
        for (std::size_t i = 0; i < x.size(); ++i) out[i] += lambda[j] * gj[i];
    }
    out.insert(out.end(), gx.begin(), gx.end());
    return out;
}

// H_L = H_f + sum_j lambda_j H_(g_j).
template <class F, ch05::VectorFunction G>
[[nodiscard]] ch01::Matrix lagrangian_hessian(F&& f, G&& g, const Vector& x, const Vector& lambda) {
    auto H = ch01::hessian(f, x);
    for (std::size_t j = 0; j < lambda.size(); ++j) {
        const auto Hj = ch01::hessian(detail::component(g, j), x);
        for (std::size_t r = 0; r < x.size(); ++r)
            for (std::size_t c = 0; c < x.size(); ++c) H[r][c] += lambda[j] * Hj[r][c];
    }
    return H;
}

// Multipliers that best satisfy grad f + sum lambda_j grad g_j = 0 at x in
// the least squares sense: (Jg Jg^T) lambda = -Jg grad f. The default
// starting lambda (lambda = 0 makes the bordered Jacobian singular when f is
// linear).
template <class F, ch05::VectorFunction G>
[[nodiscard]] Vector least_squares_multipliers(F&& f, G&& g, const Vector& x) {
    const std::size_t m = g(x).size();
    const Vector gf = ch01::gradient(f, x);
    std::vector<Vector> Jg;
    for (std::size_t j = 0; j < m; ++j) Jg.push_back(ch01::gradient(detail::component(g, j), x));
    ch05::Matrix M(m, m);
    Vector rhs(m, 0.0);
    for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t i = 0; i < x.size(); ++i) rhs[a] -= Jg[a][i] * gf[i];
        for (std::size_t b = 0; b < m; ++b)
            for (std::size_t i = 0; i < x.size(); ++i) M(a, b) += Jg[a][i] * Jg[b][i];
    }
    return ch05::solve_linear(M, rhs);
}

// Solves the Lagrange system from (x0, lambda0) by Newton's method; an empty
// lambda0 starts from least_squares_multipliers(f, g, x0).
template <class F, ch05::VectorFunction G>
[[nodiscard]] LagrangeResult lagrange_stationary_point(F&& f, G&& g, Vector x0, Vector lambda0 = {},
                                                       const ch05::Tolerances& tol = {1e-9, 1e-8, 100}) {
    const std::size_t n = x0.size();
    const std::size_t m = g(x0).size();
    if (m == 0 || m >= n) throw std::invalid_argument("lagrange_stationary_point: need 0 < m < n constraints");
    if (lambda0.empty()) lambda0 = least_squares_multipliers(f, g, x0);
    if (lambda0.size() != m) throw std::invalid_argument("lagrange_stationary_point: lambda0 has the wrong size");
    const auto split = [n](const Vector& z) {
        return std::pair{Vector(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(n)),
                         Vector(z.begin() + static_cast<std::ptrdiff_t>(n), z.end())};
    };
    const auto F_ = [&](const Vector& z) {
        const auto [x, l] = split(z);
        return lagrangian_gradient(f, g, x, l);
    };
    const auto J = [&](const Vector& z) {
        const auto [x, l] = split(z);
        const auto H = lagrangian_hessian(f, g, x, l);
        ch05::Matrix M(n + m, n + m);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c) M(r, c) = H[r][c];
        for (std::size_t j = 0; j < m; ++j) {
            const Vector gj = ch01::gradient(detail::component(g, j), x);
            for (std::size_t i = 0; i < n; ++i) M(n + j, i) = M(i, n + j) = gj[i];
        }
        return M;
    };
    Vector z = std::move(x0);
    z.insert(z.end(), lambda0.begin(), lambda0.end());
    const auto res = ch05::newton_nd(F_, J, std::move(z), tol);
    auto [x, l] = split(res.x);
    return {std::move(x), std::move(l), res.iterations};
}

// Second-order test on the tangent space of the constraints at x.
template <class F, ch05::VectorFunction G>
[[nodiscard]] CriticalPointType classify_constrained(F&& f, G&& g, const Vector& x, const Vector& lambda,
                                                     double tol = 1e-6) {
    const auto H = lagrangian_hessian(f, g, x, lambda);
    std::vector<Vector> rows;
    for (std::size_t j = 0; j < lambda.size(); ++j) rows.push_back(ch01::gradient(detail::component(g, j), x));
    const auto Z = detail::null_space(rows, x.size());
    const std::size_t k = Z.size();
    ch01::Matrix R(k, ch01::Vector(k, 0.0));
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b)
            for (std::size_t r = 0; r < x.size(); ++r)
                for (std::size_t c = 0; c < x.size(); ++c) R[a][b] += Z[a][r] * H[r][c] * Z[b][c];
    return ch08::classify_hessian(R, tol);
}

}  // namespace p4mfe::ch09
