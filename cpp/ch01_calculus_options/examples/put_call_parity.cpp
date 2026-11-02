// Detect a Put-Call parity violation in quoted prices and print the
// arbitrage trade that locks in a riskless profit (section 1.9.1).
// The prices below are illustrative.

#include <cstdio>
#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/options.hpp>

int main() {
  using namespace p4mfe::ch01;
  const double S = 42.0, K = 40.0, T = 0.5, r = 0.05, q = 0.0;
  const double C = 5.0, P = 1.0;

  std::printf("S=%.2f K=%.2f T=%.2f r=%.2f q=%.2f  C=%.4f P=%.4f\n", S, K, T, r,
              q, C, P);
  std::printf("Forward price F           = %.4f\n", forward_price(S, T, r, q));
  std::printf("Put implied by parity     = %.4f\n",
              put_from_call(C, S, K, T, r, q));
  std::printf("Parity violation          = %.4f\n",
              parity_violation(C, P, S, K, T, r, q));

  if (const auto arb = parity_arbitrage(C, P, S, K, T, r, q)) {
    std::printf(
        "Arbitrage at t=0: calls %+.4f, puts %+.4f, shares %+.4f, cash %+.4f\n",
        arb->calls, arb->puts, arb->shares, arb->cash);
    std::printf("Riskless profit at T      = %.4f\n", arb->profit);
  } else {
    std::printf("No arbitrage.\n");
  }
}
