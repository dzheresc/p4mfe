#pragma once
// Exercises 25-27 (section 1.12): stock vs. calls, replicating a
// piecewise-linear payoff, and synthesizing a range digital with calls.

#include <algorithm>
#include <p4mfe/ch01/exercises/portfolio.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch01::ex {

// ----------------------------------------------------------- Exercise 25 ---
// An asset at $35 is expected to trade in the $40-$45 range in one year.
// One-year ATM calls cost $4. Either buy the asset or buy ATM calls: which is
// better, depending on where the asset is in a year?
//
// Solution. Compare returns on the same amount invested (rates ignored):
//   stock:  R_S = S(T)/35 - 1;
//   calls:  R_C = (S(T) - 35)^+ / 4 - 1.
// For S(T) <= 35 the calls lose everything (R_C = -100%) while the stock
// loses (35 - S(T))/35. For S(T) > 35, R_C > R_S iff
//   (S - 35)/4 - 1 > (S - 35)/35   <=>   31 S > 1225   <=>   S > 39.516.
// So: below 39.52 the stock is better, above it the calls are. In the
// expected 40-45 range the calls win: at 40, R_S = 14.3% vs R_C = 25%; at 45,
// R_S = 28.6% vs R_C = 150%. The calls are the leveraged bet: much higher
// return if the view is right, total loss if S(T) <= 35.
inline constexpr double ex25_spot = 35.0;
inline constexpr double ex25_call_price = 4.0;
[[nodiscard]] inline double ex25_stock_return(double ST) {
  return ST / ex25_spot - 1.0;
}
[[nodiscard]] inline double ex25_call_return(double ST) {
  return std::max(ST - ex25_spot, 0.0) / ex25_call_price - 1.0;
}
// Crossover where both returns agree: S = S0^2 / (S0 - c) = 1225/31.
inline constexpr double ex25_crossover =
    ex25_spot * ex25_spot / (ex25_spot - ex25_call_price);

// ----------------------------------------------------------- Exercise 26 ---
// Build, from plain vanilla options with maturity T, cash, and the asset
// (no dividends, r = 0), a portfolio with payoff
//   V(T) = 2 S(T)       if S(T) < 20,
//          60 - S(T)    if 20 <= S(T) < 40,
//          S(T) - 20    if 40 <= S(T).
//
// Solution. V is continuous (40 at S = 20, 20 at S = 40) and piecewise
// linear with V(0) = 0 and slopes 2, -1, 1. Start with V(0) in cash and the
// first slope in shares, then add (new slope - old slope) calls at each kink:
//   V(T) = 2 S(T) - 3 (S(T) - 20)^+ + 2 (S(T) - 40)^+,
// i.e. long 2 shares, short 3 calls with strike 20, long 2 calls with
// strike 40. Check: S = 30: 60 - 30 = 30 = 60 - 30; S = 50: 100 - 90 + 20 = 30
// = 50 - 20.
//
// replicate_piecewise_linear does this for any continuous piecewise-linear
// payoff on [0, inf): v0 = V(0), slopes[i] on the i-th piece, kinks[i]
// separating piece i from piece i + 1 (so slopes.size() == kinks.size() + 1).
[[nodiscard]] inline Portfolio replicate_piecewise_linear(
    double v0, std::span<const double> slopes, std::span<const double> kinks) {
  if (slopes.size() != kinks.size() + 1)
    throw std::invalid_argument(
        "replicate_piecewise_linear: need one more slope than kinks");
  if (!std::is_sorted(kinks.begin(), kinks.end()))
    throw std::invalid_argument(
        "replicate_piecewise_linear: kinks must be increasing");
  Portfolio p{v0, slopes[0], {}};
  for (std::size_t i = 0; i < kinks.size(); ++i)
    if (const double dq = slopes[i + 1] - slopes[i]; dq != 0.0)
      p.options.push_back(call(kinks[i], dq));
  return p;
}

[[nodiscard]] inline double ex26_target(double ST) {
  if (ST < 20.0) return 2.0 * ST;
  if (ST < 40.0) return 60.0 - ST;
  return ST - 20.0;
}

[[nodiscard]] inline Portfolio ex26_portfolio() {
  const double slopes[] = {2.0, -1.0, 1.0};
  const double kinks[] = {20.0, 40.0};
  return replicate_piecewise_linear(0.0, slopes, kinks);
}

// ----------------------------------------------------------- Exercise 27 ---
// A derivative pays c if K1 <= S(T) <= K2 (0 < K1 < K2) and nothing
// otherwise. Synthesize it almost exactly with plain vanilla calls.
//
// Solution. A call spread (c/eps) [ (S - K)^+ - (S - K - eps)^+ ] is 0 below
// K, c above K + eps, and linear in between: a digital "step up" at K, blurred
// over width eps. The range digital is a step up at K1 minus a step up at
// K2, so with small eps > 0 (2 eps < K2 - K1):
//   long  c/eps calls K1,       short c/eps calls K1 + eps,
//   short c/eps calls K2 - eps, long  c/eps calls K2.
// The payoff equals c on [K1 + eps, K2 - eps], 0 outside [K1, K2], and lies in
// [0, c] on the two ramps of width eps, so it never pays more than the
// derivative and differs from it only on those ramps. (Shifting the ramps
// outward to [K1 - eps, K1] and [K2, K2 + eps] gives a portfolio that never
// pays less.) As eps -> 0 the replication becomes exact.
[[nodiscard]] inline double ex27_target(double ST, double c, double K1,
                                        double K2) {
  return (K1 <= ST && ST <= K2) ? c : 0.0;
}

[[nodiscard]] inline Portfolio ex27_range_digital(double c, double K1,
                                                  double K2, double eps) {
  if (!(0.0 < K1 && K1 < K2))
    throw std::invalid_argument("ex27: need 0 < K1 < K2");
  if (!(eps > 0.0 && 2.0 * eps <= K2 - K1))
    throw std::invalid_argument("ex27: need 0 < 2 eps <= K2 - K1");
  const double n = c / eps;
  return {0.0,
          0.0,
          {call(K1, n), call(K1 + eps, -n), call(K2 - eps, -n), call(K2, n)}};
}

}  // namespace p4mfe::ch01::ex
