// Detect a Put-Call parity violation in quoted prices and print the
// arbitrage trade that locks in a riskless profit (section 1.9.1).
// The prices below are illustrative.

#include <cstdio>
#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/options.hpp>

int main() {
  using namespace p4mfe::ch01;
  const double spot = 42.0, strike = 40.0, maturity = 0.5, r = 0.05, q = 0.0;
  const double call = 5.0, put = 1.0;

  std::printf("S=%.2f K=%.2f T=%.2f r=%.2f q=%.2f  C=%.4f P=%.4f\n", spot,
              strike, maturity, r, q, call, put);
  std::printf("Forward price F           = %.4f\n",
              ForwardPrice(spot, maturity, r, q));
  std::printf("Put implied by parity     = %.4f\n",
              PutFromCall(call, spot, strike, maturity, r, q));
  std::printf("Parity violation          = %.4f\n",
              ParityViolation(call, put, spot, strike, maturity, r, q));

  if (const auto arb =
          FindParityArbitrage(call, put, spot, strike, maturity, r, q)) {
    std::printf(
        "Arbitrage at t=0: calls %+.4f, puts %+.4f, shares %+.4f, cash %+.4f\n",
        arb->calls, arb->puts, arb->shares, arb->cash);
    std::printf("Riskless profit at T      = %.4f\n", arb->profit);
  } else {
    std::printf("No arbitrage.\n");
  }
}
