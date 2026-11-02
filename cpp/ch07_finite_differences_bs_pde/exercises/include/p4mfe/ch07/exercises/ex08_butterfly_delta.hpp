#pragma once
// Exercise 8 (section 7.6): the scaled butterfly payoff is a nascent Dirac
// delta at K.

#include <concepts>
#include <cstddef>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch07/exercises/ex06_butterfly_spread.hpp>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 8 ---
// Let f(S) = [C(K + x, T) - 2 C(K, T) + C(K - x, T)] / x^2, where C(K, T) is
// the value of a call with strike K and maturity T on an asset with spot S
// following a lognormal distribution. Show that for any continuous g,
//   lim_{x -> 0} int_{-inf}^{inf} f(S) g(S) dS = g(K).
//
// Reading: f must be a function of S that concentrates at K, so C(., T) here
// is the value at maturity T, i.e. the payoff (S - strike)^+ with S = S(T).
// (With the Black-Scholes value at t < T, f tends to C_KK(K), a function of S
// whose integral against g is not g(K).) Then, by exercise 6,
//   f(S) = (x - |S - K|)^+ / x^2,
// a tent of height 1/x on [K - x, K + x].
//
// Proof. f >= 0, f = 0 outside [K - x, K + x], and
//   int f(S) dS = (1/x^2) * (base 2x * height x / 2) = 1.
// Hence
//   |int f(S) g(S) dS - g(K)| = |int f(S) (g(S) - g(K)) dS|
//                            <= max_{|S - K| <= x} |g(S) - g(K)| * int f(S) dS
//                             = max_{|S - K| <= x} |g(S) - g(K)|,
// which tends to 0 as x -> 0 because g is continuous at K. So f is a nascent
// Dirac delta at K.
// If g is C^2, the rate is O(x^2): with u = S - K,
//   int f g = g(K) + g'(K) int f u + g''(K)/2 int f u^2 + ...,
// int f u = 0 by symmetry and int f u^2 = 2/x^2 int_0^x (x - u) u^2 du = x^2/6,
// so int f g = g(K) + g''(K) x^2/12 + O(x^4).
[[nodiscard]] inline double Ex08F(double spot, double strike, double x) {
  return Ex06ButterflyPayoff(spot, strike, x) / (x * x);
}
// int f(S) g(S) dS by Simpson's rule (chapter 2) on [K - x, K] and [K, K + x],
// where f is smooth; f = 0 elsewhere.
template <std::invocable<double> G>
[[nodiscard]] double Ex08Integral(G&& g, double strike, double x,
                                  std::size_t n = 64) {
  const auto fg = [&](double spot) { return Ex08F(spot, strike, x) * g(spot); };
  return ch02::SimpsonRule(fg, strike - x, strike, n) +
         ch02::SimpsonRule(fg, strike, strike + x, n);
}

}  // namespace p4mfe::ch07::ex
