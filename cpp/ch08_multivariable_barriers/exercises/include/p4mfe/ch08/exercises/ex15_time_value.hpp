#pragma once
// Exercise 15 (section 8.12): the time value of a call is largest at the money.

#include <p4mfe/ch03/black_scholes.hpp>

#include <algorithm>
#include <cmath>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 15 ---
// Show that the maximum of the time value C_BS(S) - max(S - K, 0) of a
// European call is obtained for S = K.
//
// Let f(S) = C(S) - max(S - K, 0), with C the Black-Scholes value (any
// t < T, r, and q >= 0). f is continuous, and differentiable except at S = K:
//   0 < S < K:  f'(S) = Delta(C) = e^(-qT) N(d1) > 0,
//   S > K:      f'(S) = e^(-qT) N(d1) - 1 < 0          (e^(-qT) <= 1, N(d1) < 1).
// So f is strictly increasing on (0, K] and strictly decreasing on [K, inf),
// and its maximum is at S = K, where it equals C(K). (f has a kink there: the
// one-sided derivatives are e^(-qT) N(d1) and e^(-qT) N(d1) - 1.)
// Also f(S) -> 0 as S -> 0, and as S -> infinity
// f(S) - [K (1 - e^(-rT)) - S (1 - e^(-qT))] -> 0: f tends to K (1 - e^(-rT)) for
// q = 0 (exercise 19) and to -infinity for q > 0 (exercise 20).
struct Ex15Market {
    double K = 40.0, T = 0.5, sigma = 0.3, r = 0.05, q = 0.0;
};

[[nodiscard]] inline double ex15_time_value(double S, const Ex15Market& m = {}) {
    return ch03::bs_call(S, m.K, m.T, m.sigma, m.r, m.q) - std::max(S - m.K, 0.0);
}

// f'(S) away from S = K.
[[nodiscard]] inline double ex15_derivative(double S, const Ex15Market& m = {}) {
    const double delta = ch03::bs_greeks(ch03::OptionType::Call, S, m.K, m.T, m.sigma, m.r, m.q).delta;
    return S < m.K ? delta : delta - 1.0;
}

}  // namespace p4mfe::ch08::ex
