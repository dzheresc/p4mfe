// Section 3.7: implied volatility by Newton's method started at the
// inflection point sigma_0 = sqrt(2 |ln(S/K) + (r - q) T| / T), compared with
// bisection.

#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>

int main() {
  using namespace p4mfe::ch03;
  const double spot = 30, strike = 30, maturity = 0.5, r = 0.03, q = 0.01,
               call = 2.5;
  std::printf("Call price %g, S = %g, K = %g, T = %g, r = %g, q = %g\n", call,
              spot, strike, maturity, r, q);
  std::printf("initial guess sigma_0 = %.6f\n\n",
              ImpliedVolInitialGuess(spot, strike, maturity, r, q));

  // Show the Newton iterates themselves.
  double sigma = ImpliedVolInitialGuess(spot, strike, maturity, r, q);
  for (int k = 1; k <= 5; ++k) {
    sigma -=
        (BsCall(spot, strike, maturity, sigma, r, q) - call) /
        BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r, q).vega;
    std::printf("  iteration %d: sigma = %.12f\n", k, sigma);
  }

  const auto n =
      ImpliedVolNewton(OptionType::kCall, call, spot, strike, maturity, r, q);
  const auto b = ImpliedVolBisection(OptionType::kCall, call, spot, strike,
                                     maturity, r, q);
  std::printf("\nNewton:    sigma = %.12f in %zu iterations\n", n.sigma,
              n.iterations);
  std::printf("Bisection: sigma = %.12f in %zu iterations\n", b.sigma,
              b.iterations);
  std::printf("check: BS price at Newton sigma = %.12f\n",
              BsCall(spot, strike, maturity, n.sigma, r, q));
}
