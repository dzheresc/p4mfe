#pragma once
// Chapter 3, section 3.7: implied volatility, the sigma for which the
// Black-Scholes price equals a given market price.
//
// The price is strictly increasing in sigma (vega > 0), so the root is unique
// when the price lies strictly inside the no-arbitrage bounds of section 1.8.
// The book solves f(sigma) = V_BS(sigma) - V_market = 0 with Newton's method
//   sigma_{k+1} = sigma_k - f(sigma_k) / vega(sigma_k),
// started at the inflection point of V_BS(sigma),
//   sigma_0 = sqrt(2 |ln(S/K) + (r - q) T| / T),
// from which the iterates converge monotonically. Bisection is given as a
// slower, derivative-free alternative.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <stdexcept>

namespace p4mfe::ch03 {

struct ImpliedVolResult {
  double sigma;
  std::size_t iterations;
};

namespace detail {
inline void CheckPriceInBounds(OptionType type, double price, double spot,
                               double strike, double maturity, double r,
                               double q) {
  const auto b = type == OptionType::kCall
                     ? ch01::CallBounds(spot, strike, maturity, r, q)
                     : ch01::PutBounds(spot, strike, maturity, r, q);
  if (!(price > b.lower && price < b.upper))
    throw std::domain_error(
        "implied volatility: price outside the no-arbitrage bounds");
}
}  // namespace detail

// Inflection point of sigma -> V_BS(sigma); floored so that d1 stays finite
// when the option is at the money forward.
[[nodiscard]] inline double ImpliedVolInitialGuess(double spot, double strike,
                                                   double maturity, double r,
                                                   double q = 0.0) {
  return std::max(
      std::sqrt(2.0 * std::abs(std::log(spot / strike) + (r - q) * maturity) /
                maturity),
      1e-4);
}

// Stops when two consecutive iterates differ by less than tol (the book's
// criterion), or when the price is matched to rounding error: for deep in-
// or out-of-the-money options vega is tiny, sigma is only determined to about
// eps * (S + K) / vega, and tol may be out of reach.
[[nodiscard]] inline ImpliedVolResult ImpliedVolNewton(
    OptionType type, double price, double spot, double strike, double maturity,
    double r, double q = 0.0, double tol = 1e-10, std::size_t max_iter = 100) {
  detail::CheckPriceInBounds(type, price, spot, strike, maturity, r, q);
  const double price_noise =
      8.0 * std::numeric_limits<double>::epsilon() *
      (spot * std::exp(-q * maturity) + strike * std::exp(-r * maturity));
  double sigma = ImpliedVolInitialGuess(spot, strike, maturity, r, q);
  for (std::size_t k = 1; k <= max_iter; ++k) {
    const double f = BsPrice(type, spot, strike, maturity, sigma, r, q) - price;
    if (std::abs(f) <= price_noise) return {sigma, k - 1};
    const double vega =
        BsGreeks(type, spot, strike, maturity, sigma, r, q).vega;
    const double next = sigma - f / vega;
    if (!(next > 0.0) || !std::isfinite(next))
      throw std::runtime_error("implied_vol_newton: iterate left (0, inf)");
    if (std::abs(next - sigma) < tol) return {next, k};
    sigma = next;
  }
  throw std::runtime_error("implied_vol_newton: no convergence");
}

// Bisection on [lo, hi]; widens hi until it brackets the root.
[[nodiscard]] inline ImpliedVolResult ImpliedVolBisection(
    OptionType type, double price, double spot, double strike, double maturity,
    double r, double q = 0.0, double tol = 1e-10, double lo = 1e-6,
    double hi = 1.0) {
  detail::CheckPriceInBounds(type, price, spot, strike, maturity, r, q);
  const auto f = [&](double s) {
    return BsPrice(type, spot, strike, maturity, s, r, q) - price;
  };
  while (f(hi) < 0.0) {
    hi *= 2.0;
    if (hi > 1e3)
      throw std::runtime_error(
          "implied_vol_bisection: cannot bracket the root");
  }
  std::size_t k = 0;
  while (hi - lo >= tol) {
    const double mid = 0.5 * (lo + hi);
    (f(mid) < 0.0 ? lo : hi) = mid;
    ++k;
  }
  return {0.5 * (lo + hi), k};
}

}  // namespace p4mfe::ch03
