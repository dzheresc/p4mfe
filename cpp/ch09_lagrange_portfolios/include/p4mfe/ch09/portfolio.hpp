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

[[nodiscard]] inline double Dot(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) throw std::invalid_argument("dot: size mismatch");
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

[[nodiscard]] inline Vector Multiply(const Matrix& matrix, const Vector& x) {
  if (matrix.Cols() != x.size())
    throw std::invalid_argument("multiply: size mismatch");
  Vector y(matrix.Rows(), 0.0);
  for (std::size_t i = 0; i < matrix.Rows(); ++i)
    for (std::size_t j = 0; j < matrix.Cols(); ++j) y[i] += matrix(i, j) * x[j];
  return y;
}

// ---------------------------------------------------------------- 9.2 ------
// Sigma_ij = rho_ij sigma_i sigma_j.
[[nodiscard]] inline Matrix CovarianceMatrix(const Vector& sigma,
                                             const Matrix& corr) {
  const std::size_t n = sigma.size();
  if (corr.Rows() != n || corr.Cols() != n)
    throw std::invalid_argument("covariance_matrix: size mismatch");
  Matrix covariance(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      covariance(i, j) = corr(i, j) * sigma[i] * sigma[j];
  return covariance;
}

[[nodiscard]] inline double PortfolioMean(const Vector& w, const Vector& mu) {
  return Dot(w, mu);
}
[[nodiscard]] inline double PortfolioVariance(const Vector& w,
                                              const Matrix& covariance) {
  return Dot(w, Multiply(covariance, w));
}
[[nodiscard]] inline double PortfolioStdDev(const Vector& w,
                                            const Matrix& covariance) {
  return std::sqrt(PortfolioVariance(w, covariance));
}

// The scalars A, B, C, D of the frontier.
struct FrontierConstants {
  double a, b, c, d;
};
[[nodiscard]] inline FrontierConstants ComputeFrontierConstants(
    const Vector& mu, const Matrix& covariance) {
  const Vector ones(mu.size(), 1.0);
  const Vector cov_inv_ones = ch05::SolveLinear(covariance, ones);
  const Vector cov_inv_mu = ch05::SolveLinear(covariance, mu);
  const double a = Dot(ones, cov_inv_ones), b = Dot(ones, cov_inv_mu),
               c = Dot(mu, cov_inv_mu);
  const double d = a * c - b * b;
  if (!(d > 0.0))
    throw std::invalid_argument(
        "frontier_constants: need non-constant expected returns");
  return {a, b, c, d};
}

// ---------------------------------------------------------------- 9.3 ------
[[nodiscard]] inline Vector GlobalMinVariancePortfolio(
    const Matrix& covariance) {
  Vector w = ch05::SolveLinear(covariance, Vector(covariance.Rows(), 1.0));
  double s = 0.0;
  for (double x : w) s += x;
  for (double& x : w) x /= s;
  return w;
}

[[nodiscard]] inline Vector MinVariancePortfolio(const Vector& mu,
                                                 const Matrix& covariance,
                                                 double target_mean) {
  const auto [A, B, C, D] = ComputeFrontierConstants(mu, covariance);
  Vector rhs(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i)
    rhs[i] = ((C - B * target_mean) + (A * target_mean - B) * mu[i]) / D;
  return ch05::SolveLinear(covariance, rhs);
}

// sigma_P^2 on the frontier for mean mu_P.
[[nodiscard]] inline double FrontierVariance(const Vector& mu,
                                             const Matrix& covariance,
                                             double target_mean) {
  const auto [A, B, C, D] = ComputeFrontierConstants(mu, covariance);
  return (A * target_mean * target_mean - 2.0 * B * target_mean + C) / D;
}

// Risky weights of the minimum variance portfolio with mean mu_P when cash
// (rate r_f) is available; cash weight is 1 - sum(w).
[[nodiscard]] inline Vector MinVariancePortfolioWithCash(
    const Vector& mu, const Matrix& covariance, double r_f,
    double target_mean) {
  Vector m(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i) m[i] = mu[i] - r_f;
  Vector w = ch05::SolveLinear(covariance, m);
  const double k = (target_mean - r_f) / Dot(m, w);
  for (double& x : w) x *= k;
  return w;
}

// The tangency (maximum Sharpe ratio) portfolio: Sigma^-1 m / (1^T Sigma^-1 m).
[[nodiscard]] inline Vector TangencyPortfolio(const Vector& mu,
                                              const Matrix& covariance,
                                              double r_f) {
  Vector m(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i) m[i] = mu[i] - r_f;
  Vector w = ch05::SolveLinear(covariance, m);
  double s = 0.0;
  for (double x : w) s += x;
  if (s == 0.0)
    throw std::invalid_argument(
        "tangency_portfolio: undefined (r_f equals the GMV mean)");
  for (double& x : w) x /= s;
  return w;
}

[[nodiscard]] inline double SharpeRatio(const Vector& w, const Vector& mu,
                                        const Matrix& covariance, double r_f) {
  return (PortfolioMean(w, mu) - r_f) / PortfolioStdDev(w, covariance);
}

// ---------------------------------------------------------------- 9.4 ------
// Maximum mean among fully invested portfolios with standard deviation sigma_P.
[[nodiscard]] inline double MaxReturnForStdDev(const Vector& mu,
                                               const Matrix& covariance,
                                               double sigma_p) {
  const auto [A, B, C, D] = ComputeFrontierConstants(mu, covariance);
  const double disc = A * sigma_p * sigma_p - 1.0;
  if (disc < 0.0)
    throw std::invalid_argument(
        "max_return_for_std_dev: sigma_P below the minimum 1/sqrt(A)");
  return (B + std::sqrt(D * disc)) / A;
}

[[nodiscard]] inline Vector MaxReturnPortfolio(const Vector& mu,
                                               const Matrix& covariance,
                                               double sigma_p) {
  return MinVariancePortfolio(mu, covariance,
                              MaxReturnForStdDev(mu, covariance, sigma_p));
}

// With cash: the capital market line, mu_P = r_f + sigma_P sqrt(m^T Sigma^-1
// m).
[[nodiscard]] inline double MaxReturnWithCash(const Vector& mu,
                                              const Matrix& covariance,
                                              double r_f, double sigma_p) {
  Vector m(mu.size());
  for (std::size_t i = 0; i < mu.size(); ++i) m[i] = mu[i] - r_f;
  return r_f + sigma_p * std::sqrt(Dot(m, ch05::SolveLinear(covariance, m)));
}

[[nodiscard]] inline Vector MaxReturnPortfolioWithCash(const Vector& mu,
                                                       const Matrix& covariance,
                                                       double r_f,
                                                       double sigma_p) {
  return MinVariancePortfolioWithCash(
      mu, covariance, r_f, MaxReturnWithCash(mu, covariance, r_f, sigma_p));
}

}  // namespace p4mfe::ch09
