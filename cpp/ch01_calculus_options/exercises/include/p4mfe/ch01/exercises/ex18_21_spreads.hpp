#pragma once
// Exercises 18-21 (section 1.12): butterfly, bull and bear spreads, and the
// convexity of call payoffs. Payoff diagrams are produced with payoff_table()
// (see examples/solutions.cpp and the Python plotting helper).

#include <p4mfe/ch01/exercises/portfolio.hpp>

namespace p4mfe::ch01::ex {

// ----------------------------------------------------------- Exercise 18 ---
// Portfolio: long one call K1 = 30, short two calls K2 = 35, long one call
// K3 = 40, same underlying and maturity T. Draw the payoff diagram V(T)
// against S(T).  (Note: a butterfly spread; a long butterfly bets on
// K1 <= S(T) <= K3.)
//
// Solution.
//   V(T) = 0           for S <= 30,
//          S - 30      for 30 <= S <= 35,
//          40 - S      for 35 <= S <= 40,
//          0           for S >= 40;
// a tent of height 5 peaked at S = 35, never negative.
[[nodiscard]] inline Portfolio ex18_butterfly() {
  return {0.0, 0.0, {call(30, 1), call(35, -2), call(40, 1)}};
}

// ----------------------------------------------------------- Exercise 19 ---
// Draw the payoff diagrams at maturity of
//   - a bull spread: long a call with strike 30, short a call with strike 35;
//   - a bear spread: long a put with strike 20, short a put with strike 15.
//
// Solution.
//   bull:  0 for S <= 30;  S - 30 for 30 <= S <= 35;  5 for S >= 35.
//   bear:  5 for S <= 15;  20 - S for 15 <= S <= 20;  0 for S >= 20.
// Both payoffs lie in [0, 5]; the bull spread profits from a rise, the bear
// spread from a fall.
[[nodiscard]] inline Portfolio ex19_bull_spread() {
  return {0.0, 0.0, {call(30, 1), call(35, -1)}};
}
[[nodiscard]] inline Portfolio ex19_bear_spread() {
  return {0.0, 0.0, {put(20, 1), put(15, -1)}};
}

// ----------------------------------------------------------- Exercise 20 ---
// Calls with strikes 45, 50, 55 (same asset and maturity) cost $4, $6, $9.
// Build a butterfly: long a 45-call and a 55-call, short two 50-calls. What
// are the payoff and the P&L at maturity? When is it profitable? (r = 0.)
//
// Solution. Cost = 4 - 2*6 + 9 = $1 (paid at 0; with r = 0 it is also the
// cost at T).
//   payoff:  0 (S <= 45),  S - 45 (45..50),  55 - S (50..55),  0 (S >= 55);
//   P&L = payoff - 1:  -1 outside (45, 55), up to +4 at S = 50.
// Profitable for 46 < S(T) < 54 (P&L = 0 at S = 46 and S = 54).
[[nodiscard]] inline Portfolio ex20_butterfly() {
  return {0.0, 0.0, {call(45, 1, 4.0), call(50, -2, 6.0), call(55, 1, 9.0)}};
}
inline constexpr double ex20_breakeven_low = 46.0;
inline constexpr double ex20_breakeven_high = 54.0;

// ----------------------------------------------------------- Exercise 21 ---
// Which portfolio would you rather hold (same asset, same maturity)?
//   Portfolio 1: long one call with strike X - 5 and long one call with strike
//   X + 5; Portfolio 2: long two calls with strike X.
//
// Solution. Portfolio 1. Its payoff minus Portfolio 2's is
//   (S - X + 5)^+ - 2 (S - X)^+ + (S - X - 5)^+,
// a butterfly centred at X: zero outside (X - 5, X + 5), positive (up to 5)
// inside. So Portfolio 1 pays at least as much in every state and strictly
// more when |S(T) - X| < 5 (convexity of K -> (S - K)^+). By no-arbitrage it
// must also cost more; if both were offered at the same price, holding
// Portfolio 1 (or buying 1 and selling 2) would be an arbitrage.
[[nodiscard]] inline Portfolio ex21_portfolio1(double X) {
  return {0.0, 0.0, {call(X - 5, 1), call(X + 5, 1)}};
}
[[nodiscard]] inline Portfolio ex21_portfolio2(double X) {
  return {0.0, 0.0, {call(X, 2)}};
}

}  // namespace p4mfe::ch01::ex
