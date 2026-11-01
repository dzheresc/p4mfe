#pragma once
// Exercise 3 (section 9.6): minimum variance and maximum return portfolios of
// five assets, long and short positions allowed.

#include <p4mfe/ch09/portfolio.hpp>

#include <cmath>
#include <vector>

namespace p4mfe::ch09::ex {

// ------------------------------------------------------------ Exercise 3 ---
// Five tradable assets have expected returns and standard deviations
//   mu    = (0.08, 0.10, 0.13, 0.15, 0.20),
//   sigma = (0.14, 0.18, 0.23, 0.25, 0.35),
// and correlation matrix
//         (  1     -0.3    0.4    0.25  -0.2  )
//         ( -0.3    1     -0.1   -0.2    0.15 )
//   Omega=(  0.4   -0.1    1      0.35   0.25 )
//         (  0.25  -0.2    0.35   1     -0.15 )
//         ( -0.2    0.15   0.25  -0.15   1    ).
// Long and short positions of any size are allowed.
// (i)  Find the minimal variance portfolio with 15% expected return and its
//      standard deviation.
// (ii) Find the maximum expected return portfolio with 25% standard deviation
//      and its expected return.
//
// Solution. The covariance matrix is Sigma_ij = Omega_ij sigma_i sigma_j.
// Omega is positive definite (its leading principal minors are positive;
// the smallest eigenvalue is about 0.370), so Sigma is too and the problems
// of sections 9.3-9.4 apply with weights w, sum w = 1, any signs.
//
// (i) Minimize w^T Sigma w subject to mu . w = 0.15 and 1 . w = 1. The
// Lagrange system (L = w^T Sigma w + l1 (mu.w - mu_P) + l2 (1.w - 1))
//   2 Sigma w + l1 mu + l2 1 = 0,  mu . w = mu_P,  1 . w = 1
// gives w = Sigma^-1 (a mu + b 1) and, with
//   A = 1^T Sigma^-1 1 = 144.98195,  B = 1^T Sigma^-1 mu = 15.053812,
//   C = mu^T Sigma^-1 mu = 1.7264770, D = A C - B^2 = 23.690759,
// the closed form of section 9.3
//   w = Sigma^-1 ((C - B mu_P) 1 + (A mu_P - B) mu) / D,
//   sigma_P^2 = (A mu_P^2 - 2 B mu_P + C) / D.
// Since w^T Sigma w is strictly convex and the constraints are linear, the
// stationary point is the global minimum. For mu_P = 0.15:
//   w = (0.067354, 0.217299, -0.002244, 0.406893, 0.310697)
//   (asset 3 is shorted, 0.22% of the capital),
//   sigma_P = sqrt(0.019941) = 0.141214, i.e. 14.12%.
//
// (ii) Maximize mu . w subject to w^T Sigma w = 0.25^2 and 1 . w = 1. The
// portfolios with standard deviation sigma_P lie on or inside the frontier
// sigma^2 = (A mu^2 - 2 B mu + C)/D; the largest mean with sigma = sigma_P
// is the upper root (section 9.4)
//   mu_P = (B + sqrt(D (A sigma_P^2 - 1))) / A,
// which needs sigma_P >= 1/sqrt(A) = 0.083051 (the global minimum variance
// portfolio's risk): 0.25 qualifies. The maximal portfolio is the minimum
// variance portfolio for that mu_P. For sigma_P = 0.25:
//   mu_P = 0.199151, i.e. 19.92%,
//   w = (-0.395503, 0.096062, 0.045839, 0.709880, 0.543722)
//   (asset 1 is shorted, 39.55% of the capital).
//
// Checks (in the tests): both portfolios satisfy their constraints; the
// general Lagrange solver of section 9.1 started from equal weights finds the
// same weights and classifies them as a constrained minimum (i) and
// maximum (ii); Omega passes Sylvester's criterion.

struct Ex03Data {
    Vector mu{0.08, 0.10, 0.13, 0.15, 0.20};
    Vector sigma{0.14, 0.18, 0.23, 0.25, 0.35};
    Matrix corr{{1.0, -0.3, 0.4, 0.25, -0.2},
                {-0.3, 1.0, -0.1, -0.2, 0.15},
                {0.4, -0.1, 1.0, 0.35, 0.25},
                {0.25, -0.2, 0.35, 1.0, -0.15},
                {-0.2, 0.15, 0.25, -0.15, 1.0}};
    [[nodiscard]] Matrix covariance() const { return covariance_matrix(sigma, corr); }
};

struct Ex03Portfolio {
    Vector w;
    double mean;
    double std_dev;
};

// (i) Minimum variance portfolio with expected return target_mean.
[[nodiscard]] inline Ex03Portfolio ex03_min_variance(double target_mean = 0.15, const Ex03Data& d = {}) {
    const Matrix S = d.covariance();
    Vector w = min_variance_portfolio(d.mu, S, target_mean);
    return {w, portfolio_mean(w, d.mu), portfolio_std_dev(w, S)};
}

// (ii) Maximum expected return portfolio with standard deviation target_std.
[[nodiscard]] inline Ex03Portfolio ex03_max_return(double target_std = 0.25, const Ex03Data& d = {}) {
    const Matrix S = d.covariance();
    Vector w = max_return_portfolio(d.mu, S, target_std);
    return {w, portfolio_mean(w, d.mu), portfolio_std_dev(w, S)};
}

}  // namespace p4mfe::ch09::ex
