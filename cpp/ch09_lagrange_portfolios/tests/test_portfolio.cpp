#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch09/lagrange.hpp>
#include <p4mfe/ch09/portfolio.hpp>
#include <random>

using namespace p4mfe::ch09;

namespace {
const Vector mu{0.08, 0.12, 0.15};
const Matrix Sigma = covariance_matrix(
    {0.15, 0.20, 0.30},
    Matrix{{1.0, 0.3, 0.1}, {0.3, 1.0, 0.4}, {0.1, 0.4, 1.0}});
double sum(const Vector& w) {
  double s = 0;
  for (double x : w) s += x;
  return s;
}
}  // namespace

// ---------------------------------------------------------------- 9.2 -----
TEST(Ch09Portfolio, Basics) {
  EXPECT_DOUBLE_EQ(Sigma(0, 0), 0.0225);
  EXPECT_DOUBLE_EQ(Sigma(1, 2), 0.4 * 0.2 * 0.3);
  const Vector w{0.5, 0.3, 0.2};
  EXPECT_NEAR(portfolio_mean(w, mu), 0.04 + 0.036 + 0.03, 1e-15);
  double var = 0;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) var += w[i] * w[j] * Sigma(i, j);
  EXPECT_NEAR(portfolio_variance(w, Sigma), var, 1e-15);
  EXPECT_NEAR(portfolio_std_dev(w, Sigma), std::sqrt(var), 1e-15);
}

// ---------------------------------------------------------------- 9.3 -----
TEST(Ch09Portfolio, MinimumVariance) {
  const auto [A, B, C, D] = frontier_constants(mu, Sigma);
  const auto gmv = global_min_variance_portfolio(Sigma);
  EXPECT_NEAR(sum(gmv), 1.0, 1e-14);
  EXPECT_NEAR(portfolio_variance(gmv, Sigma), 1 / A, 1e-14);
  EXPECT_NEAR(portfolio_mean(gmv, mu), B / A, 1e-14);
  // Any fully invested perturbation increases the variance.
  std::mt19937_64 rng(5);
  std::normal_distribution<double> z;
  for (int k = 0; k < 20; ++k) {
    Vector d{z(rng), z(rng), 0};
    d[2] = -d[0] - d[1];
    Vector w = gmv;
    for (std::size_t i = 0; i < 3; ++i) w[i] += 0.05 * d[i];
    EXPECT_GT(portfolio_variance(w, Sigma), 1 / A);
  }

  for (double target : {0.06, 0.10, 0.14, 0.20}) {
    const auto w = min_variance_portfolio(mu, Sigma, target);
    EXPECT_NEAR(sum(w), 1.0, 1e-12);
    EXPECT_NEAR(portfolio_mean(w, mu), target, 1e-12);
    EXPECT_NEAR(portfolio_variance(w, Sigma),
                frontier_variance(mu, Sigma, target), 1e-12);
    // Same answer from the general Lagrange multiplier solver.
    const auto var = [&](const Vector& x) {
      return portfolio_variance(x, Sigma);
    };
    const auto cons = [&](const Vector& x) {
      return Vector{portfolio_mean(x, mu) - target, sum(x) - 1};
    };
    const auto r =
        lagrange_stationary_point(var, cons, {1.0 / 3, 1.0 / 3, 1.0 / 3});
    for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(r.x[i], w[i], 1e-7);
    EXPECT_EQ(classify_constrained(var, cons, r.x, r.lambda),
              CriticalPointType::Minimum);
  }
  // Two-fund separation: frontier portfolios are affine in the target mean.
  const auto w1 = min_variance_portfolio(mu, Sigma, 0.1),
             w2 = min_variance_portfolio(mu, Sigma, 0.2);
  const auto w3 = min_variance_portfolio(mu, Sigma, 0.13);
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_NEAR(w3[i], 0.7 * w1[i] + 0.3 * w2[i], 1e-12);
  // The frontier's vertex is the GMV portfolio.
  EXPECT_NEAR(frontier_variance(mu, Sigma, B / A), 1 / A, 1e-14);
  EXPECT_GT(D, 0);
}

TEST(Ch09Portfolio, WithCash) {
  const double rf = 0.03;
  const auto t = tangency_portfolio(mu, Sigma, rf);
  EXPECT_NEAR(sum(t), 1.0, 1e-14);
  // The tangency portfolio is on the frontier and has the highest Sharpe ratio.
  EXPECT_NEAR(portfolio_variance(t, Sigma),
              frontier_variance(mu, Sigma, portfolio_mean(t, mu)), 1e-12);
  for (double target : {0.06, 0.1, 0.14, 0.2})
    EXPECT_LE(
        sharpe_ratio(min_variance_portfolio(mu, Sigma, target), mu, Sigma, rf),
        sharpe_ratio(t, mu, Sigma, rf) + 1e-12);
  for (double target : {0.05, 0.1, 0.2}) {
    const auto w = min_variance_portfolio_with_cash(mu, Sigma, rf, target);
    EXPECT_NEAR(rf * (1 - sum(w)) + portfolio_mean(w, mu), target, 1e-12);
    // Risky part proportional to the tangency portfolio.
    for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(w[i] / sum(w), t[i], 1e-12);
    // Cash helps: lower variance than the all-risky frontier.
    EXPECT_LT(portfolio_variance(w, Sigma),
              frontier_variance(mu, Sigma, target) + 1e-15);
  }
}

// ---------------------------------------------------------------- 9.4 -----
TEST(Ch09Portfolio, MaximumReturn) {
  const auto [A, B, C, D] = frontier_constants(mu, Sigma);
  for (double s : {0.14, 0.2, 0.3}) {
    const double m = max_return_for_std_dev(mu, Sigma, s);
    const auto w = max_return_portfolio(mu, Sigma, s);
    EXPECT_NEAR(portfolio_std_dev(w, Sigma), s, 1e-12);
    EXPECT_NEAR(portfolio_mean(w, mu), m, 1e-12);
    EXPECT_GE(m, B / A);  // the efficient (upper) branch
    // Lagrange: maximize mu.w subject to w^T Sigma w = s^2 and sum w = 1.
    const auto ret = [&](const Vector& x) { return portfolio_mean(x, mu); };
    const auto cons = [&](const Vector& x) {
      return Vector{portfolio_variance(x, Sigma) - s * s, sum(x) - 1};
    };
    Vector start = w;
    start[0] += 0.02;
    start[1] -= 0.02;
    const auto r = lagrange_stationary_point(ret, cons, start);
    EXPECT_NEAR(ret(r.x), m, 1e-8);
    EXPECT_EQ(classify_constrained(ret, cons, r.x, r.lambda),
              CriticalPointType::Maximum);
  }
  EXPECT_THROW((void)max_return_for_std_dev(mu, Sigma, 0.5 / std::sqrt(A)),
               std::invalid_argument);

  const double rf = 0.03;
  const auto t = tangency_portfolio(mu, Sigma, rf);
  for (double s : {0.05, 0.15, 0.3}) {
    const auto w = max_return_portfolio_with_cash(mu, Sigma, rf, s);
    EXPECT_NEAR(portfolio_std_dev(w, Sigma), s, 1e-12);
    const double m = rf * (1 - sum(w)) + portfolio_mean(w, mu);
    EXPECT_NEAR(m, max_return_with_cash(mu, Sigma, rf, s), 1e-12);
    // The capital market line's slope is the tangency Sharpe ratio, and it
    // dominates the frontier.
    EXPECT_NEAR((m - rf) / s, sharpe_ratio(t, mu, Sigma, rf), 1e-12);
    if (s * s >= 1 / A) {
      EXPECT_GE(m, max_return_for_std_dev(mu, Sigma, s) - 1e-12);
    }
  }
}
