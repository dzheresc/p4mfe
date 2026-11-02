// Section 7.4: the Black-Scholes PDE checked by finite differences, and its
// financial meaning: a Delta-hedged short option replicates the payoff, with
// an error that shrinks like 1/sqrt(rebalancing frequency) whatever the drift.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/bs_pde.hpp>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch07;
  const double spot = 100, strike = 105, maturity = 0.5, sigma = 0.2, r = 0.03,
               q = 0.01;

  std::printf(
      "PDE residual Theta + 1/2 s^2 S^2 Gamma + (r-q) S Delta - r V by finite "
      "differences, t = 0\n");
  const auto call = [&](double s, double t) {
    return ch03::BsCall(s, strike, maturity - t, sigma, r, q);
  };
  const auto put = [&](double s, double t) {
    return ch03::BsPut(s, strike, maturity - t, sigma, r, q);
  };
  const auto fwd = [&](double s, double t) {
    return ForwardContractValue(s, t, strike, maturity, r, q);
  };
  const auto sq = [&](double s, double) { return s * s; };
  const auto pw = [&](double s, double t) {
    return PowerSolution(2.0, s, t, sigma, r, q);
  };
  std::printf("  call                 %10.2e\n",
              BsPdeResidualFd(call, spot, 0, sigma, r, q));
  std::printf("  put                  %10.2e\n",
              BsPdeResidualFd(put, spot, 0, sigma, r, q));
  std::printf("  forward contract     %10.2e\n",
              BsPdeResidualFd(fwd, spot, 0, sigma, r, q));
  std::printf("  S^2 e^(c t)          %10.2e\n",
              BsPdeResidualFd(pw, spot, 0, sigma, r, q));
  std::printf("  S^2 (not a solution) %10.2e\n",
              BsPdeResidualFd(sq, spot, 0, sigma, r, q));

  const double call_price = ch03::BsCall(spot, strike, maturity, sigma, r, q);
  std::printf(
      "\nShort call (C = %.4f) Delta-hedged n times; P&L over 5000 paths\n",
      call_price);
  std::printf("%6s  %8s  %12s  %12s\n", "n", "mu", "mean", "std dev");
  for (double mu : {-0.1, 0.03, 0.2})
    for (std::size_t n : {4, 16, 64, 256}) {
      const auto st = DeltaHedgeStats(OptionType::kCall, spot, strike, maturity,
                                      sigma, r, q, mu, n, 5000);
      std::printf("%6zu  %8.2f  %12.5f  %12.5f\n", n, mu, st.mean, st.std_dev);
    }
}
