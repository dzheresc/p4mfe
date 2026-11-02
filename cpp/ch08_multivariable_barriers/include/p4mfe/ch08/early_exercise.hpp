#pragma once
// Chapter 8, section 8.10: optimality of early exercise.
//
// An American call on an asset paying no dividends should never be exercised
// early: before maturity its value is at least the European value, and
//   C >= S - K e^(-r tau) > S - K   for r > 0, tau > 0,
// so selling the option always beats exercising it (by at least
// K (1 - e^(-r tau))). Hence American call = European call when q = 0.
// With dividends (q > 0), or for puts, early exercise can be optimal and the
// American option is worth strictly more.
//
// The binomial tree below (Cox-Ross-Rubinstein) makes this concrete: it
// prices both styles and shows a zero early exercise premium for calls with
// q = 0 and a positive one otherwise.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch08 {

using ch01::OptionType;

enum class ExerciseStyle { European, American };

// --------------------------------------------------------------- 8.10 ------
// CRR tree with n steps: u = e^(sigma sqrt(dt)), d = 1/u,
// p = (e^((r - q) dt) - d) / (u - d).
[[nodiscard]] inline double binomial_price(OptionType type, double S, double K,
                                           double T, double sigma, double r,
                                           double q, std::size_t n,
                                           ExerciseStyle style) {
  if (n == 0 || !(T > 0.0 && sigma > 0.0))
    throw std::invalid_argument("binomial_price: need n >= 1, T, sigma > 0");
  const double dt = T / static_cast<double>(n);
  const double u = std::exp(sigma * std::sqrt(dt));
  const double d = 1.0 / u;
  const double p = (std::exp((r - q) * dt) - d) / (u - d);
  if (!(p > 0.0 && p < 1.0))
    throw std::invalid_argument(
        "binomial_price: step too large (p not in (0, 1))");
  const double disc = std::exp(-r * dt);
  const double pu = disc * p, pd = disc * (1.0 - p);
  const double u2 = u * u;
  // Values far out of the money decay geometrically down the tree and reach
  // the subnormal range after about a thousand steps. On Intel CPUs each
  // subnormal operation costs ~100 cycles, which made the European tree
  // ~25x slower at n = 10,000. Values below the smallest normal double
  // (about 1e-308, far below any price) are set to 0 instead.
  constexpr double kTiny = std::numeric_limits<double>::min();
  std::vector<double> V(n + 1);
  for (std::size_t j = 0; j <= n; ++j)
    V[j] = ch01::payoff(
        type,
        S * std::pow(u, 2.0 * static_cast<double>(j) - static_cast<double>(n)),
        K);
  if (style == ExerciseStyle::American) {
    for (std::size_t i = n; i-- > 0;) {
      // S u^(2j - i): one pow per row, then a factor u^2 per node.
      double s_ij = S * std::pow(u, -static_cast<double>(i));
      for (std::size_t j = 0; j <= i; ++j, s_ij *= u2) {
        const double v = pu * V[j + 1] + pd * V[j];
        V[j] = std::max(v < kTiny ? 0.0 : v, ch01::payoff(type, s_ij, K));
      }
    }
    return V[0];
  }
  // European: only nodes in [lo, hi] are nonzero (a put's zeros are at the
  // top of the tree, a call's at the bottom). The range widens by at most
  // one node per step and is trimmed once its end values drop below kTiny,
  // so zero nodes are never computed and subnormals never persist.
  std::size_t lo = 0, hi = n;
  while (lo <= n && V[lo] == 0.0) ++lo;
  if (lo > n) return 0.0;
  while (V[hi] == 0.0) --hi;
  for (std::size_t i = n; i-- > 0;) {
    lo = lo > 0 ? lo - 1 : 0;
    hi = std::min(hi, i);
    for (std::size_t j = lo; j <= hi; ++j) V[j] = pu * V[j + 1] + pd * V[j];
    while (hi > lo && V[hi] < kTiny) V[hi--] = 0.0;
    while (lo < hi && V[lo] < kTiny) V[lo++] = 0.0;
    if (V[lo] < kTiny) return 0.0;  // lo == hi and the last value vanished
  }
  return V[0];
}

// American minus European value on the same tree.
[[nodiscard]] inline double early_exercise_premium(OptionType type, double S,
                                                   double K, double T,
                                                   double sigma, double r,
                                                   double q,
                                                   std::size_t n = 500) {
  return binomial_price(type, S, K, T, sigma, r, q, n,
                        ExerciseStyle::American) -
         binomial_price(type, S, K, T, sigma, r, q, n, ExerciseStyle::European);
}

// What is lost by exercising a call now instead of selling it (q = 0):
// C - (S - K) >= K (1 - e^(-r tau)) > 0.
[[nodiscard]] inline double call_exercise_loss(double S, double K, double tau,
                                               double sigma, double r) {
  return ch03::bs_call(S, K, tau, sigma, r) - (S - K);
}
[[nodiscard]] inline double call_exercise_loss_bound(double K, double tau,
                                                     double r) noexcept {
  return -K * std::expm1(-r * tau);
}

}  // namespace p4mfe::ch08
