// Section 3.7: implied volatility by Newton's method started at the
// inflection point sigma_0 = sqrt(2 |ln(S/K) + (r - q) T| / T), compared with
// bisection.

#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>

int main() {
  using namespace p4mfe::ch03;
  const double S = 30, K = 30, T = 0.5, r = 0.03, q = 0.01, C = 2.5;
  std::printf("Call price %g, S = %g, K = %g, T = %g, r = %g, q = %g\n", C, S,
              K, T, r, q);
  std::printf("initial guess sigma_0 = %.6f\n\n",
              implied_vol_initial_guess(S, K, T, r, q));

  // Show the Newton iterates themselves.
  double sigma = implied_vol_initial_guess(S, K, T, r, q);
  for (int k = 1; k <= 5; ++k) {
    sigma -= (bs_call(S, K, T, sigma, r, q) - C) /
             bs_greeks(OptionType::Call, S, K, T, sigma, r, q).vega;
    std::printf("  iteration %d: sigma = %.12f\n", k, sigma);
  }

  const auto n = implied_vol_newton(OptionType::Call, C, S, K, T, r, q);
  const auto b = implied_vol_bisection(OptionType::Call, C, S, K, T, r, q);
  std::printf("\nNewton:    sigma = %.12f in %zu iterations\n", n.sigma,
              n.iterations);
  std::printf("Bisection: sigma = %.12f in %zu iterations\n", b.sigma,
              b.iterations);
  std::printf("check: BS price at Newton sigma = %.12f\n",
              bs_call(S, K, T, n.sigma, r, q));
}
