// Section 3.8: a short position in 1000 calls, hedged Delta-neutral with the
// stock and Delta-Gamma neutral with the stock and a second call. Shows how
// each portfolio's value changes when the spot moves.

#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/hedging.hpp>

int main() {
  using namespace p4mfe::ch03;
  const double S = 50, sigma = 0.3, r = 0.05;
  const double K1 = 50, T1 = 0.25;  // the calls we are short
  const double K2 = 55, T2 = 0.5;   // the hedging call
  const double n = -1000;

  const auto g1 = bs_greeks(OptionType::Call, S, K1, T1, sigma, r);
  const auto g2 = bs_greeks(OptionType::Call, S, K2, T2, sigma, r);
  const double delta_shares = delta_hedge_shares(n * g1.delta);
  const auto dg =
      delta_gamma_hedge(n * g1.delta, n * g1.gamma, g2.delta, g2.gamma);
  std::printf("short %g calls: Delta %.2f, Gamma %.4f\n", -n, n * g1.delta,
              n * g1.gamma);
  std::printf("Delta hedge:       buy %.2f shares\n", delta_shares);
  std::printf(
      "Delta-Gamma hedge: buy %.2f calls (K=%g, T=%g) and %.2f shares\n\n",
      dg.hedge_options, K2, T2, dg.shares);

  const auto c1 = [&](double s) { return bs_call(s, K1, T1, sigma, r); };
  const auto c2 = [&](double s) { return bs_call(s, K2, T2, sigma, r); };
  const auto unhedged = [&](double s) { return n * c1(s); };
  const auto delta_hedged = [&](double s) {
    return n * c1(s) + delta_shares * s;
  };
  const auto dg_hedged = [&](double s) {
    return n * c1(s) + dg.hedge_options * c2(s) + dg.shares * s;
  };

  std::printf("%6s  %12s  %12s  %12s  %14s\n", "dS", "unhedged", "Delta",
              "Delta-Gamma", "Delta approx");
  for (double dS : {-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0}) {
    const double s = S + dS;
    std::printf("%6.1f  %12.2f  %12.2f  %12.4f  %14.2f\n", dS,
                unhedged(s) - unhedged(S), delta_hedged(s) - delta_hedged(S),
                dg_hedged(s) - dg_hedged(S),
                portfolio_change_approx(0.0, n * g1.gamma, dS));
  }
  std::printf(
      "\n(Delta approx is 1/2 Gamma dS^2, the second-order change left after "
      "Delta hedging.)\n");
}
