// Sections 9.2-9.4: minimum variance and maximum return portfolios, with and
// without a risk-free asset.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch09/portfolio.hpp>

int main() {
  using namespace p4mfe::ch09;
  const Vector mu{0.08, 0.12, 0.15};
  const Vector sig{0.15, 0.20, 0.30};
  const Matrix Sigma = covariance_matrix(
      sig, Matrix{{1.0, 0.3, 0.1}, {0.3, 1.0, 0.4}, {0.1, 0.4, 1.0}});
  const double rf = 0.03;

  const auto [A, B, C, D] = frontier_constants(mu, Sigma);
  const auto gmv = global_min_variance_portfolio(Sigma);
  std::printf("Assets: mu = (%.2f, %.2f, %.2f), sigma = (%.2f, %.2f, %.2f)\n",
              mu[0], mu[1], mu[2], sig[0], sig[1], sig[2]);
  std::printf("A = %.4f, B = %.4f, C = %.4f, D = %.4f\n", A, B, C, D);
  std::printf(
      "Global minimum variance: w = (%.4f, %.4f, %.4f), mean %.4f, std "
      "%.4f\n\n",
      gmv[0], gmv[1], gmv[2], portfolio_mean(gmv, mu),
      portfolio_std_dev(gmv, Sigma));

  std::printf(
      "Minimum variance frontier (9.3)\n%8s  %8s  %8s  %8s  %8s  %10s\n",
      "mean", "std", "w1", "w2", "w3", "std w/ cash");
  for (double m = 0.06; m <= 0.2001; m += 0.02) {
    const auto w = min_variance_portfolio(mu, Sigma, m);
    const auto wc = min_variance_portfolio_with_cash(mu, Sigma, rf, m);
    std::printf("%8.3f  %8.4f  %8.4f  %8.4f  %8.4f  %10.4f\n", m,
                portfolio_std_dev(w, Sigma), w[0], w[1], w[2],
                portfolio_std_dev(wc, Sigma));
  }

  const auto t = tangency_portfolio(mu, Sigma, rf);
  std::printf(
      "\nTangency portfolio (r_f = %.2f): w = (%.4f, %.4f, %.4f), mean %.4f, "
      "std %.4f, Sharpe %.4f\n",
      rf, t[0], t[1], t[2], portfolio_mean(t, mu), portfolio_std_dev(t, Sigma),
      sharpe_ratio(t, mu, Sigma, rf));

  std::printf("\nMaximum mean for a given std (9.4)\n%8s  %12s  %12s\n", "std",
              "risky only", "with cash");
  for (double s : {0.12, 0.15, 0.2, 0.25, 0.3})
    std::printf(
        "%8.3f  %12.5f  %12.5f\n", s,
        s * s >= 1 / A ? max_return_for_std_dev(mu, Sigma, s) : std::nan(""),
        max_return_with_cash(mu, Sigma, rf, s));
}
