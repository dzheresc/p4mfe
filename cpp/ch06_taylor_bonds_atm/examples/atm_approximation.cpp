// Sections 6.8-6.9: how good is C ~ 0.4 sigma S sqrt(T) at the money, and
// the Taylor expansion of an option's value through its Greeks.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/greeks_taylor.hpp>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch06;
  const double spot = 100;

  std::printf(
      "ATM, r = q = 0: relative error of sigma S sqrt(T/(2 pi)) vs. sigma^2 T "
      "/ 24\n");
  std::printf("%6s  %6s  %12s  %12s  %12s  %12s\n", "T", "sigma", "exact",
              "approx", "rel. error", "bound");
  for (double maturity : {1.0 / 12, 0.5, 1.0, 2.0})
    for (double sigma : {0.1, 0.3, 0.6})
      std::printf("%6.3f  %6.2f  %12.6f  %12.6f  %12.3e  %12.3e\n", maturity,
                  sigma, AtmPriceExact(spot, maturity, sigma),
                  AtmPriceApprox(spot, maturity, sigma),
                  AtmRelativeError(maturity, sigma),
                  AtmRelativeErrorBound(maturity, sigma));

  std::printf("\nATM with rates, T = 0.5, sigma = 0.25\n");
  std::printf("%6s  %6s  %10s  %10s  %10s  %10s\n", "r", "q", "call", "approx",
              "put", "approx");
  for (double r : {0.0, 0.03, 0.06})
    for (double q : {0.0, 0.02})
      std::printf("%6.3f  %6.3f  %10.5f  %10.5f  %10.5f  %10.5f\n", r, q,
                  ch03::BsCall(spot, spot, 0.5, 0.25, r, q),
                  AtmCallApprox(spot, 0.5, 0.25, r, q),
                  ch03::BsPut(spot, spot, 0.5, 0.25, r, q),
                  AtmPutApprox(spot, 0.5, 0.25, r, q));

  const double strike = 105, maturity = 0.5, sigma = 0.3, r = 0.04, q = 0.01;
  const double value = ch03::BsCall(spot, strike, maturity, sigma, r, q);
  const auto g = ch03::BsGreeks(ch03::OptionType::kCall, spot, strike, maturity,
                                sigma, r, q);
  std::printf(
      "\nCall S = %.0f, K = %.0f, T = %.1f: V = %.6f, Delta = %.4f, Gamma = "
      "%.5f, Theta = %.4f\n",
      spot, strike, maturity, value, g.delta, g.gamma, g.theta);
  std::printf("%6s  %12s  %12s  %12s  %12s\n", "dS", "exact", "Delta",
              "Delta+Gamma", "+Theta (1d)");
  const double dt = 1.0 / 252;
  for (double d_spot : {-5.0, -1.0, 1.0, 5.0})
    std::printf(
        "%6.1f  %12.6f  %12.6f  %12.6f  %12.6f\n", d_spot,
        ch03::BsCall(spot + d_spot, strike, maturity - dt, sigma, r, q) - value,
        OptionChangeDelta(g, d_spot), OptionChangeDeltaGamma(g, d_spot),
        OptionChangeTaylor(g, d_spot, dt));
}
