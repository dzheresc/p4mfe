// Sections 9.2-9.4: minimum variance and maximum return portfolios, with and
// without a risk-free asset.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch09/portfolio.hpp>

int main() {
  using namespace p4mfe::ch09;
  const Vector mu{0.08, 0.12, 0.15};
  const Vector sig{0.15, 0.20, 0.30};
  const Matrix covariance = CovarianceMatrix(
      sig, Matrix{{1.0, 0.3, 0.1}, {0.3, 1.0, 0.4}, {0.1, 0.4, 1.0}});
  const double rf = 0.03;

  const auto [A, B, C, D] = ComputeFrontierConstants(mu, covariance);
  const auto gmv = GlobalMinVariancePortfolio(covariance);
  std::printf("Assets: mu = (%.2f, %.2f, %.2f), sigma = (%.2f, %.2f, %.2f)\n",
              mu[0], mu[1], mu[2], sig[0], sig[1], sig[2]);
  std::printf("A = %.4f, B = %.4f, C = %.4f, D = %.4f\n", A, B, C, D);
  std::printf(
      "Global minimum variance: w = (%.4f, %.4f, %.4f), mean %.4f, std "
      "%.4f\n\n",
      gmv[0], gmv[1], gmv[2], PortfolioMean(gmv, mu),
      PortfolioStdDev(gmv, covariance));

  std::printf(
      "Minimum variance frontier (9.3)\n%8s  %8s  %8s  %8s  %8s  %10s\n",
      "mean", "std", "w1", "w2", "w3", "std w/ cash");
  for (double m = 0.06; m <= 0.2001; m += 0.02) {
    const auto w = MinVariancePortfolio(mu, covariance, m);
    const auto wc = MinVariancePortfolioWithCash(mu, covariance, rf, m);
    std::printf("%8.3f  %8.4f  %8.4f  %8.4f  %8.4f  %10.4f\n", m,
                PortfolioStdDev(w, covariance), w[0], w[1], w[2],
                PortfolioStdDev(wc, covariance));
  }

  const auto t = TangencyPortfolio(mu, covariance, rf);
  std::printf(
      "\nTangency portfolio (r_f = %.2f): w = (%.4f, %.4f, %.4f), mean %.4f, "
      "std %.4f, Sharpe %.4f\n",
      rf, t[0], t[1], t[2], PortfolioMean(t, mu),
      PortfolioStdDev(t, covariance), SharpeRatio(t, mu, covariance, rf));

  std::printf("\nMaximum mean for a given std (9.4)\n%8s  %12s  %12s\n", "std",
              "risky only", "with cash");
  for (double s : {0.12, 0.15, 0.2, 0.25, 0.3})
    std::printf(
        "%8.3f  %12.5f  %12.5f\n", s,
        s * s >= 1 / A ? MaxReturnForStdDev(mu, covariance, s) : std::nan(""),
        MaxReturnWithCash(mu, covariance, rf, s));
}
