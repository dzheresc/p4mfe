#pragma once
// Chapter 7, section 7.3: finite difference approximations of the Greeks.
//
// When an option has no closed-form Greeks (or only a numerical price), the
// Greeks are difference quotients of the pricer V(S, T, sigma, r):
//   Delta ~ (V(S+dS) - V(S-dS)) / (2 dS)                 (or one-sided)
//   Gamma ~ (V(S+dS) - 2V(S) + V(S-dS)) / dS^2
//   Vega  ~ (V(sigma+ds) - V(sigma-ds)) / (2 ds)
//   Rho   ~ (V(r+dr) - V(r-dr)) / (2 dr)
//   Theta = dV/dt = -dV/dT ~ (V(T-dT) - V(T+dT)) / (2 dT)
// Central differences are O(h^2) accurate, one-sided ones O(h). Section
// 7.3.1: the error falls with h until round-off, of size eps V / h (eps V /
// h^2 for Gamma), takes over; see `optimal_step` and `fd_error_table` in
// finite_differences.hpp and the example `fd_accuracy`.

#include <concepts>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/finite_differences.hpp>

namespace p4mfe::ch07 {

using ch03::Greeks;
using ch03::OptionType;

// Pricer V(S, T, sigma, r); strike, dividend yield etc. are captured.
template <class P>
concept Pricer =
    std::invocable<P, double, double, double, double> &&
    std::convertible_to<std::invoke_result_t<P, double, double, double, double>,
                        double>;

// Absolute bump sizes for each variable.
struct FdSteps {
  double d_spot = 1e-2;
  double d_t = 1e-4;
  double dsigma = 1e-4;
  double dr = 1e-4;
};

// ---------------------------------------------------------------- 7.3 ------
// All five Greeks of any pricer by finite differences. `scheme` applies to
// the first-order Greeks (Theta's scheme is in calendar time t); Gamma always
// uses the central second difference.
template <Pricer P>
[[nodiscard]] Greeks FdGreeks(P&& pricer, double spot, double maturity,
                              double sigma, double r, const FdSteps& h = {},
                              FdScheme scheme = FdScheme::kCentral) {
  const auto in_spot = [&](double s) { return pricer(s, maturity, sigma, r); };
  const auto in_t = [&](double t) {
    return pricer(spot, maturity - t, sigma, r);
  };  // calendar time t, maturity fixed
  const auto in_sigma = [&](double v) { return pricer(spot, maturity, v, r); };
  const auto in_r = [&](double x) { return pricer(spot, maturity, sigma, x); };
  Greeks g{};
  g.delta = FirstDifference(in_spot, spot, h.d_spot, scheme);
  g.gamma = SecondCentralDifference(in_spot, spot, h.d_spot);
  g.vega = FirstDifference(in_sigma, sigma, h.dsigma, scheme);
  g.theta = FirstDifference(in_t, 0.0, h.d_t, scheme);
  g.rho = FirstDifference(in_r, r, h.dr, scheme);
  return g;
}

// Black-Scholes Greeks by finite differences, to compare with ch03::bs_greeks.
[[nodiscard]] inline Greeks BsFdGreeks(OptionType type, double spot,
                                       double strike, double maturity,
                                       double sigma, double r, double q = 0.0,
                                       const FdSteps& h = {},
                                       FdScheme scheme = FdScheme::kCentral) {
  return FdGreeks(
      [&](double s, double t, double v, double x) {
        return ch03::BsPrice(type, s, strike, t, v, x, q);
      },
      spot, maturity, sigma, r, h, scheme);
}

}  // namespace p4mfe::ch07
