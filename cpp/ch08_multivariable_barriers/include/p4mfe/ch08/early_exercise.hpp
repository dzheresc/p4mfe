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

enum class ExerciseStyle { kEuropean, kAmerican };

// --------------------------------------------------------------- 8.10 ------
// CRR tree with n steps: u = e^(sigma sqrt(dt)), d = 1/u,
// p = (e^((r - q) dt) - d) / (u - d).
[[nodiscard]] inline double BinomialPrice(OptionType type, double spot,
                                          double strike, double maturity,
                                          double sigma, double r, double q,
                                          std::size_t n, ExerciseStyle style) {
  if (n == 0 || !(maturity > 0.0 && sigma > 0.0))
    throw std::invalid_argument("binomial_price: need n >= 1, T, sigma > 0");
  const double dt = maturity / static_cast<double>(n);
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
  std::vector<double> values(n + 1);
  for (std::size_t j = 0; j <= n; ++j)
    values[j] = ch01::Payoff(type,
                             spot * std::pow(u, 2.0 * static_cast<double>(j) -
                                                    static_cast<double>(n)),
                             strike);
  if (style == ExerciseStyle::kAmerican) {
    for (std::size_t i = n; i-- > 0;) {
      // S u^(2j - i): one pow per row, then a factor u^2 per node.
      double s_ij = spot * std::pow(u, -static_cast<double>(i));
      for (std::size_t j = 0; j <= i; ++j, s_ij *= u2) {
        const double v = pu * values[j + 1] + pd * values[j];
        values[j] =
            std::max(v < kTiny ? 0.0 : v, ch01::Payoff(type, s_ij, strike));
      }
    }
    return values[0];
  }
  // European: only nodes in [lo, hi] are nonzero (a put's zeros are at the
  // top of the tree, a call's at the bottom). The range widens by at most
  // one node per step and is trimmed once its end values drop below kTiny,
  // so zero nodes are never computed and subnormals never persist.
  std::size_t lo = 0, hi = n;
  while (lo <= n && values[lo] == 0.0) ++lo;
  if (lo > n) return 0.0;
  while (values[hi] == 0.0) --hi;
  for (std::size_t i = n; i-- > 0;) {
    lo = lo > 0 ? lo - 1 : 0;
    hi = std::min(hi, i);
    for (std::size_t j = lo; j <= hi; ++j)
      values[j] = pu * values[j + 1] + pd * values[j];
    while (hi > lo && values[hi] < kTiny) values[hi--] = 0.0;
    while (lo < hi && values[lo] < kTiny) values[lo++] = 0.0;
    if (values[lo] < kTiny) return 0.0;  // lo == hi and the last value vanished
  }
  return values[0];
}

// American minus European value on the same tree.
[[nodiscard]] inline double EarlyExercisePremium(OptionType type, double spot,
                                                 double strike, double maturity,
                                                 double sigma, double r,
                                                 double q,
                                                 std::size_t n = 500) {
  return BinomialPrice(type, spot, strike, maturity, sigma, r, q, n,
                       ExerciseStyle::kAmerican) -
         BinomialPrice(type, spot, strike, maturity, sigma, r, q, n,
                       ExerciseStyle::kEuropean);
}

// What is lost by exercising a call now instead of selling it (q = 0):
// C - (S - K) >= K (1 - e^(-r tau)) > 0.
[[nodiscard]] inline double CallExerciseLoss(double spot, double strike,
                                             double tau, double sigma,
                                             double r) {
  return ch03::BsCall(spot, strike, tau, sigma, r) - (spot - strike);
}
[[nodiscard]] inline double CallExerciseLossBound(double strike, double tau,
                                                  double r) noexcept {
  return -strike * std::expm1(-r * tau);
}

}  // namespace p4mfe::ch08
