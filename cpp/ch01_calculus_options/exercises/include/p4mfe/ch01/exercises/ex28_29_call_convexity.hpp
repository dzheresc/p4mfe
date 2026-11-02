#pragma once
// Exercises 28-29 (section 1.12): convexity of call prices in the strike and
// the butterfly arbitrage. (Solved in the order 29 -> 28.)

#include <array>
#include <optional>
#include <p4mfe/ch01/exercises/portfolio.hpp>
#include <stdexcept>

namespace p4mfe::ch01::ex {

// ----------------------------------------------------------- Exercise 29 ---
// Calls on the same asset and with the same maturity, strikes K1 < K2 < K3,
// trade for C1, C2, C3 (no bid-ask spread) with C1 > C2 > C3. Find necessary
// and sufficient conditions on C1, C2, C3 for no arbitrage with a portfolio
// made of positions in the three options.
//
// Solution. Write lambda = (K3 - K2)/(K3 - K1), so K2 = lambda K1 + (1 -
// lambda) K3. The condition is convexity of the call price in the strike:
//      C2 <= lambda C1 + (1 - lambda) C3,
//   i.e. (K3 - K2) C1 - (K3 - K1) C2 + (K2 - K1) C3 >= 0.
//
// Why. A portfolio a C1 + b C2 + c C3 has a piecewise-linear payoff, zero
// for S <= K1, so it is >= 0 everywhere iff it is >= 0 at K2 and K3 and its
// final slope is >= 0:
//   a >= 0,   a (K3 - K1) + b (K3 - K2) >= 0,   a + b + c >= 0.
// Three independent linear inequalities cut out a cone with three edges
// (set two of them to equality):
//   (0, 0, 1)                                     long the K3 call;
//   (0, 1, -1)                                    bull spread K2/K3;
//   (K3 - K2, -(K3 - K1), K2 - K1) / (K3 - K2)    butterfly.
// Every portfolio with nonnegative payoff is a nonnegative combination of
// these three, so there is no arbitrage iff each of them has nonnegative
// cost: C3 >= 0, C2 >= C3 (given), and the butterfly inequality above.
// (Strictly, every edge payoff is positive somewhere, so a zero price would
// already be a free lottery ticket; the book's inequality is the weak one.)
// Note that C1 > C2 > C3 alone does not guarantee convexity; exercise 28 is a
// counterexample.

struct ThreeCalls {
  std::array<double, 3> K;  // K1 < K2 < K3
  std::array<double, 3> C;  // prices
};

// lambda C1 + (1 - lambda) C3 - C2: no arbitrage iff >= 0.
[[nodiscard]] inline double ex29_convexity_gap(const ThreeCalls& x) {
  const auto& [K1, K2, K3] = x.K;
  if (!(K1 < K2 && K2 < K3))
    throw std::invalid_argument("ex29: need K1 < K2 < K3");
  const double lambda = (K3 - K2) / (K3 - K1);
  return lambda * x.C[0] + (1.0 - lambda) * x.C[2] - x.C[1];
}

[[nodiscard]] inline bool ex29_no_arbitrage(const ThreeCalls& x) {
  return x.C[2] >= 0.0 && x.C[1] >= x.C[2] && ex29_convexity_gap(x) >= 0.0;
}

// If some edge of the cone has negative cost, return that portfolio (with the
// option prices filled in): its payoff is >= 0 and portfolio.cost() < 0, so
// -cost() is collected up front.
[[nodiscard]] inline std::optional<Portfolio> ex29_arbitrage(
    const ThreeCalls& x) {
  const auto& [K1, K2, K3] = x.K;
  const auto& [C1, C2, C3] = x.C;
  if (ex29_convexity_gap(x) < 0.0) {
    const double w = K3 - K2;
    return Portfolio{0.0,
                     0.0,
                     {call(K1, 1.0, C1), call(K2, -(K3 - K1) / w, C2),
                      call(K3, (K2 - K1) / w, C3)}};
  }
  if (C2 < C3)
    return Portfolio{0.0, 0.0, {call(K2, 1.0, C2), call(K3, -1.0, C3)}};
  if (C3 < 0.0) return Portfolio{0.0, 0.0, {call(K3, 1.0, C3)}};
  return std::nullopt;
}

// ----------------------------------------------------------- Exercise 28 ---
// Calls with strikes 100, 120, 130 (same asset and maturity) trade for 8, 5,
// 3. Is there an arbitrage? If yes, how do you make a riskless profit?
//
// Solution. lambda = (130 - 120)/(130 - 100) = 1/3, and
//   lambda C1 + (1 - lambda) C3 = 8/3 + 2 = 4.667 < 5 = C2,
// so convexity fails: the 120-call is too expensive. Sell the butterfly's
// middle: buy 1 call K = 100, sell 3 calls K = 120, buy 2 calls K = 130.
// Cash at 0: -8 + 15 - 6 = +$1. Payoff at T:
//   0 (S <= 100),  S - 100 (100..120),  20 - 2 (S - 120) = 260 - 2S (120..130),
//   0 (S >= 130)  -- never negative.
// Riskless profit: at least $1 (plus up to $20 if 100 < S(T) < 130).
[[nodiscard]] inline ThreeCalls ex28_data() {
  return {{100.0, 120.0, 130.0}, {8.0, 5.0, 3.0}};
}

}  // namespace p4mfe::ch01::ex
