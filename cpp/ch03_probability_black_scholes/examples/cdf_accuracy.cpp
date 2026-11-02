// Section 3.9: accuracy of the two ways of computing N(x) used to implement
// the Black-Scholes formula, and their effect on a call price.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/cdf_approx.hpp>

int main() {
  using namespace p4mfe::ch03;
  std::printf("%6s  %18s  %12s  %12s\n", "x", "N(x) (erfc)", "poly err",
              "simpson err");
  for (double x : {-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0}) {
    const double ref = p4mfe::norm_cdf(x);
    std::printf("%6.2f  %18.15f  %12.2e  %12.2e\n", x, ref,
                norm_cdf_poly(x) - ref, norm_cdf_simpson(x) - ref);
  }

  double worst = 0.0;
  for (double x = -10.0; x <= 10.0; x += 1e-3)
    worst = std::fmax(worst, std::abs(norm_cdf_poly(x) - p4mfe::norm_cdf(x)));
  std::printf("\nmax |poly error| on [-10, 10]: %.2e (A&S bound 7.5e-8)\n",
              worst);

  // Black-Scholes call with each N(x).
  const double S = 42, K = 40, T = 0.5, sigma = 0.2, r = 0.1;
  const auto [d1, d2] = bs_d1_d2(S, K, T, sigma, r);
  const auto call = [&](auto N) {
    return S * N(d1) - K * std::exp(-r * T) * N(d2);
  };
  std::printf("\ncall (S=42, K=40, T=0.5, sigma=0.2, r=0.1):\n");
  std::printf("  erfc     %.12f\n",
              call([](double x) { return p4mfe::norm_cdf(x); }));
  std::printf("  poly     %.12f\n",
              call([](double x) { return norm_cdf_poly(x); }));
  std::printf("  simpson  %.12f\n",
              call([](double x) { return norm_cdf_simpson(x); }));
}
