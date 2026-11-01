#pragma once
// Exercise 7 (section 7.6): bull spreads, the first strike derivative of the
// call price and cash-or-nothing calls.

#include <p4mfe/ch07/exercises/ex06_butterfly_spread.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 7 ---
// A bull spread is long one call with strike K and short one call with strike
// K + x, same asset and maturity; C(K), C(K + x) are their values at time t.
//
// (i) The value of 1/x bull spreads is (C(K) - C(K + x)) / x (same argument as
// exercise 6(i)). Show that it tends to -dC/dK (K) as x -> 0.
//   This is minus the forward difference quotient of C in the strike:
//   (C(K) - C(K + x)) / x = -(C(K + x) - C(K)) / x = -C_K(K) - x/2 C_KK(K) + O(x^2)
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
[[nodiscard]] inline double ex07_bull_payoff(double ST, double K, double x) {
    return std::max(ST - K, 0.0) - std::max(ST - K - x, 0.0);
}
[[nodiscard]] inline double ex07_digital_payoff(double ST, double K, double x) {
    return ex07_bull_payoff(ST, K, x) / x;
}
// (i) value of 1/x bull spreads, and its Black-Scholes limit -C_K.
[[nodiscard]] inline double ex07_scaled_bull(double x, const SpreadMarket& m = {}) {
    return (ex_call(m.K, m) - ex_call(m.K + x, m)) / x;
}
[[nodiscard]] inline double ex07_cash_or_nothing(const SpreadMarket& m = {}) {
    const auto [d1, d2] = ch03::bs_d1_d2(m.S, m.K, m.T, m.sigma, m.r, m.q);
    return std::exp(-m.r * m.T) * norm_cdf(d2);
}

}  // namespace p4mfe::ch07::ex
