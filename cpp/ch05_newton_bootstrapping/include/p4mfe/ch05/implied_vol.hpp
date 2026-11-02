#pragma once
// Chapter 5, section 5.4: implied volatility with the solvers of 5.1.
//
// sigma_imp solves f(sigma) = V_BS(sigma) - V_market = 0, with f'(sigma) =
// vega > 0. The Black-Scholes price, vega, the no-arbitrage check and the
// initial guess sigma_0 = sqrt(2 |ln(S/K) + (r - q) T| / T) (the inflection
// point of V_BS(sigma), from which Newton converges monotonically) all come
// from chapter 3; this header only plugs them into the generic solvers.
//
// Accuracy note: for deep in- or out-of-the-money options vega is tiny and
// sigma is only determined to about (price rounding) / vega, so a small
// tol.consec may be out of reach. chapter 3's implied_vol_newton handles that
// case by also stopping once the price is matched to rounding error.

#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>
#include <p4mfe/ch05/solvers.hpp>
#include <stdexcept>

namespace p4mfe::ch05 {

using ch01::OptionType;

inline constexpr Tolerances kImpliedVolTolerances{1e-9, 1e-9, 100};

[[nodiscard]] inline RootResult ImpliedVolNewton(
    OptionType type, double price, double spot, double strike, double maturity,
    double r, double q = 0.0, const Tolerances& tol = kImpliedVolTolerances) {
  ch03::detail::CheckPriceInBounds(type, price, spot, strike, maturity, r, q);
  return Newton(
      [&](double s) {
        return ch03::BsPrice(type, spot, strike, maturity, s, r, q) - price;
      },
      [&](double s) {
        return ch03::BsGreeks(type, spot, strike, maturity, s, r, q).vega;
      },
      ch03::ImpliedVolInitialGuess(spot, strike, maturity, r, q), tol);
}

// Secant method started at sigma_0 and sigma_0 + 0.1; no vega needed.
[[nodiscard]] inline RootResult ImpliedVolSecant(
    OptionType type, double price, double spot, double strike, double maturity,
    double r, double q = 0.0, const Tolerances& tol = kImpliedVolTolerances) {
  ch03::detail::CheckPriceInBounds(type, price, spot, strike, maturity, r, q);
  const double s0 = ch03::ImpliedVolInitialGuess(spot, strike, maturity, r, q);
  return Secant(
      [&](double s) {
        return ch03::BsPrice(type, spot, strike, maturity, s, r, q) - price;
      },
      s0 + 0.1, s0, tol);
}

// Bisection on [lo, hi], doubling hi until it brackets the root.
[[nodiscard]] inline RootResult ImpliedVolBisection(
    OptionType type, double price, double spot, double strike, double maturity,
    double r, double q = 0.0, const Tolerances& tol = kImpliedVolTolerances,
    double lo = 1e-6, double hi = 1.0) {
  ch03::detail::CheckPriceInBounds(type, price, spot, strike, maturity, r, q);
  const auto f = [&](double s) {
    return ch03::BsPrice(type, spot, strike, maturity, s, r, q) - price;
  };
  while (f(hi) < 0.0) {
    hi *= 2.0;
    if (hi > 1e3)
      throw std::runtime_error(
          "implied_vol_bisection: cannot bracket the root");
  }
  return Bisection(f, lo, hi, tol);
}

}  // namespace p4mfe::ch05
