// Section 8.8: Black-Scholes through the heat equation, in closed form (heat
// kernel) and by explicit finite differences, including the alpha <= 1/2
// stability limit, and by Crank-Nicolson, which has no such limit.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch08/heat_equation.hpp>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch08;
  const double strike = 40, maturity = 0.5, sigma = 0.25, r = 0.05, q = 0.02;
  const auto c = MakeHeatTransform(sigma, r, q);
  std::printf(
      "K = %.0f, T = %.1f, sigma = %.2f, r = %.2f, q = %.2f: a = %.6f, b = "
      "%.6f, tau_final = %.6f\n",
      strike, maturity, sigma, r, q, c.a, c.b, 0.5 * sigma * sigma * maturity);

  std::printf("\n%6s  %12s  %12s  %12s  %12s\n", "S", "Black-Scholes",
              "heat kernel", "FD nx=200", "FD nx=800");
  for (double spot : {30.0, 36.0, 40.0, 44.0, 50.0}) {
    const auto h = [&](double s) { return std::max(s - strike, 0.0); };
    std::printf("%6.1f  %12.7f  %12.7f  %12.7f  %12.7f\n", spot,
                ch03::BsCall(spot, strike, maturity, sigma, r, q),
                BsPriceViaHeat(h, spot, strike, maturity, sigma, r, q),
                EuropeanHeatFd(OptionType::kCall, spot, strike, maturity, sigma,
                               r, q, 200),
                EuropeanHeatFd(OptionType::kCall, spot, strike, maturity, sigma,
                               r, q, 800));
  }

  std::printf(
      "\nStability: put at S = 40 with nx = 200 and nt chosen for alpha\n");
  std::printf("%8s  %8s  %14s\n", "alpha", "nt", "price");
  const double half_width = 6.0 * sigma * std::sqrt(maturity);
  const double dx = 2 * half_width / 200,
               tau_final = 0.5 * sigma * sigma * maturity;
  for (double alpha : {0.25, 0.45, 0.5, 0.52, 0.6}) {
    const auto nt =
        static_cast<std::size_t>(std::ceil(tau_final / (alpha * dx * dx)));
    std::printf("%8.2f  %8zu  %14.6g\n", alpha, nt,
                EuropeanHeatFd(OptionType::kPut, 40, strike, maturity, sigma, r,
                               q, 200, nt));
  }
  std::printf("%8s  %8s  %14.6g\n", "exact", "",
              ch03::BsPut(40, strike, maturity, sigma, r, q));

  std::printf("\nCrank-Nicolson: the same put, nx = 200, far fewer steps\n");
  std::printf("%8s  %8s  %14s\n", "alpha", "nt", "price");
  for (std::size_t nt : {10, 25, 50, 200}) {
    const double alpha = tau_final / static_cast<double>(nt) / (dx * dx);
    std::printf("%8.2f  %8zu  %14.6g\n", alpha, nt,
                EuropeanHeatCrankNicolson(OptionType::kPut, 40, strike,
                                          maturity, sigma, r, q, 200, nt));
  }
}
