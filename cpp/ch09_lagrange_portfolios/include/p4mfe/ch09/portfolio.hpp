#pragma once
// Chapter 9, sections 9.2-9.4: optimal investment portfolios.
//
// 9.2 Assets with expected returns mu and covariance matrix Sigma; a
//     portfolio with weights w (sum w = 1) has
//       mean mu_P = mu . w,   variance sigma_P^2 = w^T Sigma w.
//     The problems below are solved with Lagrange multipliers in closed form.
//     With A = 1^T Sigma^-1 1, B = 1^T Sigma^-1 mu, C = mu^T Sigma^-1 mu and
//     D = A C - B^2 > 0:
//
// 9.3 Minimum variance for a target mean mu_P (constraints mu . w = mu_P,
//     sum w = 1):
//       w = Sigma^-1 ((C - B mu_P) 1 + (A mu_P - B) mu) / D,
//       sigma_P^2 = (A mu_P^2 - 2 B mu_P + C) / D          (the frontier).
//     Without a target mean: w = Sigma^-1 1 / A, sigma^2 = 1 / A.
//     With a risk-free asset at rate r_f, the excess returns m = mu - r_f 1
//     give w_risky = (mu_P - r_f) Sigma^-1 m / (m^T Sigma^-1 m) (the rest in
//     cash): every efficient portfolio mixes cash with the tangency
//     portfolio Sigma^-1 m / (1^T Sigma^-1 m).
//
// 9.4 Maximum mean for a target standard deviation sigma_P: the upper branch
//     of the frontier,
//       mu_P = (B + sqrt(D (A sigma_P^2 - 1))) / A,  sigma_P^2 >= 1 / A,
//     and with a risk-free asset the capital market line
//       mu_P = r_f + sigma_P sqrt(m^T Sigma^-1 m).
//
// Linear systems are solved with chapter 5's solve_linear; Sigma is never
// inverted explicitly.

#include <cmath>
#include <cstddef>
#include <p4mfe/ch05/newton_nd.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch09 {

using Vector = std::vector<double>;
using ch05::Matrix;

[[nodiscard]] inline double dot(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) throw std::invalid_argument("dot: size mismatch");
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

[[nodiscard]] inline Vector multiply(const Matrix& A, const Vector& x) {
  if (A.Cols() != x.size())
    throw std::invalid_argument("multiply: size mismatch");
  Vector y(A.Rows(), 0.0);
  for (std::size_t i = 0; i < A.Rows(); ++i)
    for (std::size_t j = 0; j < A.Cols(); ++j) y[i] += A(i, j) * x[j];
  return y;
}

// ---------------------------------------------------------------- 9.2 ------
// Sigma_ij = rho_ij sigma_i sigma_j.
[[nodiscard]] inline Matrix covariance_matrix(const Vector& sigma,
                                              const Matrix& corr) {
  const std::size_t n = sigma.size();
  if (corr.Rows() != n || corr.Cols() != n)
    throw std::invalid_argument("covariance_matrix: size mismatch");
  Matrix S(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      S(i, j) = corr(i, j) * sigma[i] * sigma[j];
  return S;
}

[[nodiscard]] inline double portfolio_mean(const Vector& w, const Vector& mu) {
  return dot(w, mu);
}
[[nodiscard]] inline double portfolio_variance(const Vector& w,
                                               const Matrix& Sigma) {
  return dot(w, multiply(Sigma, w));
}
[[nodiscard]] inline double portfolio_std_dev(const Vector& w,
                                              const Matrix& Sigma) {
  return std::sqrt(portfolio_variance(w, Sigma));
}

// The scalars A, B, C, D of the frontier.
struct FrontierConstants {
  double A, B, C, D;
};
[[nodiscard]] inline FrontierConstants frontier_constants(const Vector& mu,
                                                          const Matrix& Sigma) {
  const Vector ones(mu.size(), 1.0);
  const Vector Si1 = ch05::SolveLinear(Sigma, ones);
  const Vector Simu = ch05::SolveLinear(Sigma, mu);
  const double A = dot(ones, Si1), B = dot(ones, Simu), C = dot(mu, Simu);
  const double D = A * C - B * B;
  if (!(D > 0.0))
    throw std::invalid_argument(
        "frontier_constants: need non-constant expected returns");
  return {A, B, C, D};
}

// ---------------------------------------------------------------- 9.3 ------
[[nodiscard]] inline Vector global_min_variance_portfolio(const Matrix& Sigma) {
  Vector w = ch05::SolveLinear(Sigma, Vector(Sigma.Rows(), 1.0));
  double s = 0.0;
  for (double x : w) s += x;
  for (double& x : w) x /= s;
  return w;
}

[[nodiscard]] inline Vector min_variance_portfolio(const Vector& mu,
                                                   const Matrix& Sigma,
                                                   double target_mean) {
  const auto [A, B, C, D] = frontier_constants(mu, Sigma);
  Vector rhs(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i)
    rhs[i] = ((C - B * target_mean) + (A * target_mean - B) * mu[i]) / D;
  return ch05::SolveLinear(Sigma, rhs);
}

// sigma_P^2 on the frontier for mean mu_P.
[[nodiscard]] inline double frontier_variance(const Vector& mu,
                                              const Matrix& Sigma,
                                              double target_mean) {
  const auto [A, B, C, D] = frontier_constants(mu, Sigma);
  return (A * target_mean * target_mean - 2.0 * B * target_mean + C) / D;
}

// Risky weights of the minimum variance portfolio with mean mu_P when cash
// (rate r_f) is available; cash weight is 1 - sum(w).
[[nodiscard]] inline Vector min_variance_portfolio_with_cash(
    const Vector& mu, const Matrix& Sigma, double r_f, double target_mean) {
  Vector m(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i) m[i] = mu[i] - r_f;
  Vector w = ch05::SolveLinear(Sigma, m);
  const double k = (target_mean - r_f) / dot(m, w);
  for (double& x : w) x *= k;
  return w;
}

// The tangency (maximum Sharpe ratio) portfolio: Sigma^-1 m / (1^T Sigma^-1 m).
[[nodiscard]] inline Vector tangency_portfolio(const Vector& mu,
                                               const Matrix& Sigma,
                                               double r_f) {
  Vector m(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i) m[i] = mu[i] - r_f;
  Vector w = ch05::SolveLinear(Sigma, m);
  double s = 0.0;
  for (double x : w) s += x;
  if (s == 0.0)
    throw std::invalid_argument(
        "tangency_portfolio: undefined (r_f equals the GMV mean)");
  for (double& x : w) x /= s;
  return w;
}

[[nodiscard]] inline double sharpe_ratio(const Vector& w, const Vector& mu,
                                         const Matrix& Sigma, double r_f) {
  return (portfolio_mean(w, mu) - r_f) / portfolio_std_dev(w, Sigma);
}

// ---------------------------------------------------------------- 9.4 ------
// Maximum mean among fully invested portfolios with standard deviation sigma_P.
[[nodiscard]] inline double max_return_for_std_dev(const Vector& mu,
                                                   const Matrix& Sigma,
                                                   double sigma_P) {
  const auto [A, B, C, D] = frontier_constants(mu, Sigma);
  const double disc = A * sigma_P * sigma_P - 1.0;
  if (disc < 0.0)
    throw std::invalid_argument(
        "max_return_for_std_dev: sigma_P below the minimum 1/sqrt(A)");
  return (B + std::sqrt(D * disc)) / A;
}

[[nodiscard]] inline Vector max_return_portfolio(const Vector& mu,
                                                 const Matrix& Sigma,
                                                 double sigma_P) {
  return min_variance_portfolio(mu, Sigma,
                                max_return_for_std_dev(mu, Sigma, sigma_P));
}

// With cash: the capital market line, mu_P = r_f + sigma_P sqrt(m^T Sigma^-1
// m).
[[nodiscard]] inline double max_return_with_cash(const Vector& mu,
                                                 const Matrix& Sigma,
                                                 double r_f, double sigma_P) {
  Vector m(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i) m[i] = mu[i] - r_f;
  return r_f + sigma_P * std::sqrt(dot(m, ch05::SolveLinear(Sigma, m)));
}

[[nodiscard]] inline Vector max_return_portfolio_with_cash(const Vector& mu,
                                                           const Matrix& Sigma,
                                                           double r_f,
                                                           double sigma_P) {
  return min_variance_portfolio_with_cash(
      mu, Sigma, r_f, max_return_with_cash(mu, Sigma, r_f, sigma_P));
}

}  // namespace p4mfe::ch09
