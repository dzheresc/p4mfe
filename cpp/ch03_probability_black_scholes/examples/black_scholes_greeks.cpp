// Sections 3.5-3.6: Black-Scholes prices and Greeks of a call and a put, with
// the closed-form Greeks checked against finite differences and the
// Black-Scholes PDE (section 3.6.1).

#include <cstdio>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch03/black_scholes.hpp>

int main() {
  using namespace p4mfe::ch03;
  using p4mfe::ch01::Derivative;
  const double spot = 40, strike = 40, maturity = 0.5, sigma = 0.3, r = 0.05,
               q = 0.02;
  std::printf("S = %g, K = %g, T = %g, sigma = %g, r = %g, q = %g\n\n", spot,
              strike, maturity, sigma, r, q);

  std::printf("%-5s %10s %10s %10s %10s %10s %10s %12s\n", "", "price", "delta",
              "gamma", "vega", "theta", "rho", "PDE resid");
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const auto g = BsGreeks(type, spot, strike, maturity, sigma, r, q);
    std::printf("%-5s %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %12.2e\n",
                type == OptionType::kCall ? "call" : "put",
                BsPrice(type, spot, strike, maturity, sigma, r, q), g.delta,
                g.gamma, g.vega, g.theta, g.rho,
                BsPdeResidual(type, spot, strike, maturity, sigma, r, q));
  }

  const auto g =
      BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r, q);
  const double fd_delta = Derivative(
      [&](double s) { return BsCall(s, strike, maturity, sigma, r, q); }, spot);
  const double fd_vega = Derivative(
      [&](double v) { return BsCall(spot, strike, maturity, v, r, q); }, sigma);
  std::printf("\ncall delta: closed form %.10f, finite difference %.10f\n",
              g.delta, fd_delta);
  std::printf("call vega:  closed form %.10f, finite difference %.10f\n",
              g.vega, fd_vega);
  std::printf("S e^(-qT) phi(d1) - K e^(-rT) phi(d2) = %.2e\n",
              BsPhiIdentityGap(spot, strike, maturity, sigma, r, q));
}
