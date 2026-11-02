// Sections 6.1-6.4: Taylor polynomials of e^x and their remainder bounds, the
// series of N(x), and log vs. percentage returns (r = R - R^2/2 + ...).

#include <cmath>
#include <cstdio>
#include <p4mfe/ch06/returns.hpp>
#include <p4mfe/ch06/taylor.hpp>
#include <p4mfe/ch06/taylor_series.hpp>
#include <p4mfe/normal.hpp>
#include <vector>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch06;

  std::printf("Taylor polynomials of e^x around 0 at x = 0.5\n");
  std::printf("%3s  %20s  %12s  %12s\n", "n", "P_n(0.5)", "error", "bound");
  for (unsigned n = 1; n <= 8; ++n) {
    const std::vector<double> d(n + 1, 1.0);
    const double p = TaylorPolynomial(d, 0.0, 0.5);
    std::printf("%3u  %20.15f  %12.3e  %12.3e\n", n, p, std::exp(0.5) - p,
                TaylorRemainderBound(std::exp(0.5), 0.0, 0.5, n));
  }

  std::printf("\nN(x) from its Taylor series (n + 1 terms)\n");
  std::printf("%6s  %4s  %18s  %12s\n", "x", "n", "series", "error");
  for (double x : {0.5, 1.0, 2.0, 3.0})
    for (unsigned n : {4u, 10u, 20u})
      std::printf("%6.2f  %4u  %18.15f  %12.3e\n", x, n, NormCdfSeries(x, n),
                  NormCdfSeries(x, n) - NormCdf(x));

  std::printf("\nPercentage vs. log returns: r = ln(1 + R) ~ R - R^2/2\n");
  std::printf("%8s  %12s  %12s  %12s\n", "R", "r", "R - R^2/2", "error");
  for (double pct_return : {-0.2, -0.05, -0.01, 0.01, 0.05, 0.2}) {
    const double r = LogFromPercentage(pct_return);
    std::printf("%8.3f  %12.8f  %12.8f  %12.3e\n", pct_return, r,
                LogReturnTaylor(pct_return, 2),
                r - LogReturnTaylor(pct_return, 2));
  }
}
