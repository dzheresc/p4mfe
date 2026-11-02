// Sections 7.1 and 7.3-7.3.1: forward, backward and central differences,
// finite difference Greeks, and why making h smaller stops helping.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/fd_greeks.hpp>
#include <p4mfe/ch07/finite_differences.hpp>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch07;

  const auto f = [](double x) { return std::exp(-x * x / 2) * std::sin(x); };
  const double a = 1.0;
  const double df = std::exp(-0.5) * (std::cos(1.0) - std::sin(1.0));
  std::printf("f(x) = e^(-x^2/2) sin x, f'(1) = %.15f\n", df);
  std::printf("%8s  %14s  %14s  %14s\n", "h", "forward err", "backward err",
              "central err");
  for (double h = 0.1; h > 1e-13; h /= 10)
    std::printf(
        "%8.0e  %14.3e  %14.3e  %14.3e\n", h, forward_difference(f, a, h) - df,
        backward_difference(f, a, h) - df, central_difference(f, a, h) - df);
  std::printf(
      "optimal steps: one-sided %.1e, central %.1e, second order %.1e\n\n",
      optimal_step(FdScheme::Forward), optimal_step(FdScheme::Central),
      optimal_step(FdScheme::Central, 2));

  const double S = 100, K = 100, T = 1, sigma = 0.2, r = 0.05;
  const auto exact = ch03::BsGreeks(OptionType::kCall, S, K, T, sigma, r);
  const auto call_S = [&](double s) { return ch03::BsCall(s, K, T, sigma, r); };
  std::printf("ATM call S = K = 100, T = 1, sigma = 0.2, r = 0.05\n");
  std::printf("Delta = %.12f, Gamma = %.12f\n", exact.delta, exact.gamma);
  std::printf("%8s  %14s  %14s  %14s\n", "dS", "fwd Delta err", "cen Delta err",
              "Gamma err");
  for (double h = 1.0; h > 1e-9; h /= 10)
    std::printf("%8.0e  %14.3e  %14.3e  %14.3e\n", h,
                forward_difference(call_S, S, h) - exact.delta,
                central_difference(call_S, S, h) - exact.delta,
                second_central_difference(call_S, S, h) - exact.gamma);

  const auto fd = bs_fd_greeks(OptionType::kCall, S, K, T, sigma, r);
  std::printf("\n%8s  %16s  %16s\n", "Greek", "closed form", "central FD");
  std::printf("%8s  %16.10f  %16.10f\n", "Delta", exact.delta, fd.delta);
  std::printf("%8s  %16.10f  %16.10f\n", "Gamma", exact.gamma, fd.gamma);
  std::printf("%8s  %16.10f  %16.10f\n", "Vega", exact.vega, fd.vega);
  std::printf("%8s  %16.10f  %16.10f\n", "Theta", exact.theta, fd.theta);
  std::printf("%8s  %16.10f  %16.10f\n", "Rho", exact.rho, fd.rho);
}
