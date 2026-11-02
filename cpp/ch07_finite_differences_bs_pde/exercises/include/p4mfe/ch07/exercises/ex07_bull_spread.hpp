#pragma once
// Exercise 7 (section 7.6): bull spreads, the first strike derivative of the
// call price and cash-or-nothing calls.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <p4mfe/ch07/exercises/ex06_butterfly_spread.hpp>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 7 ---
// A bull spread is long one call with strike K and short one call with strike
// K + x, same asset and maturity; C(K), C(K + x) are their values at time t.
//
// (i) The value of 1/x bull spreads is (C(K) - C(K + x)) / x (same argument as
// exercise 6(i)). Show that it tends to -dC/dK (K) as x -> 0.
//   This is minus the forward difference quotient of C in the strike:
//   (C(K) - C(K + x)) / x = -(C(K + x) - C(K)) / x = -C_K(K) - x/2 C_KK(K) +
//   O(x^2)
// for C twice differentiable in K, so the limit is -C_K(K) and the error is
// first order. For Black-Scholes
//   -dC/dK = e^(-rT) N(d2),
// the Black-Scholes price of a cash-or-nothing call paying 1 if S(T) > K.
// Since C_KK > 0 the spread value increases to this limit as x decreases.
//
// (ii) Show that the payoff of 1/x bull spreads tends to the payoff of a
// security paying 1 if S(T) > K and 0 if S(T) < K.
//   The payoff is [(S - K)^+ - (S - K - x)^+] / x =
//     0             for S <= K,
//     (S - K) / x   for K < S < K + x,
//     1             for S >= K + x.
// For S <= K it is always 0; for S > K it equals 1 once x <= S - K. So it
// converges pointwise to 1_{S > K}: 1/x bull spreads with x small are a
// synthetic cash-or-nothing call maturing at T. (At S = K the limit is 0.)
[[nodiscard]] inline double Ex07BullPayoff(double spot_t, double strike,
                                           double x) {
  return std::max(spot_t - strike, 0.0) - std::max(spot_t - strike - x, 0.0);
}
[[nodiscard]] inline double Ex07DigitalPayoff(double spot_t, double strike,
                                              double x) {
  return Ex07BullPayoff(spot_t, strike, x) / x;
}
// (i) value of 1/x bull spreads, and its Black-Scholes limit -C_K.
[[nodiscard]] inline double Ex07ScaledBull(double x,
                                           const SpreadMarket& m = {}) {
  return (ExCall(m.strike, m) - ExCall(m.strike + x, m)) / x;
}
[[nodiscard]] inline double Ex07CashOrNothing(const SpreadMarket& m = {}) {
  const auto [d1, d2] =
      ch03::BsD1D2(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q);
  return std::exp(-m.r * m.maturity) * NormCdf(d2);
}

}  // namespace p4mfe::ch07::ex
