#pragma once
// Exercise 6 (section 7.6): butterfly spreads, the second strike derivative of
// the call price and Arrow-Debreu securities.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch07::ex {

// Market for the numerical checks of exercises 6-8 (the exercises have no
// numbers): spot, maturity, volatility, rate, dividend yield, center strike.
struct SpreadMarket {
  double spot = 52.0, maturity = 0.5, sigma = 0.25, r = 0.04, q = 0.01,
         strike = 50.0;
};

// Black-Scholes call price as a function of the strike only.
[[nodiscard]] inline double ExCall(double strike, const SpreadMarket& m = {}) {
  return ch03::BsCall(m.spot, strike, m.maturity, m.sigma, m.r, m.q);
}

// ------------------------------------------------------------ Exercise 6 ---
// A butterfly spread is long one call with strike K - x, long one call with
// strike K + x and short two calls with strike K, all on the same asset with
// the same maturity.
//
// (i) Show that its value is C(K + x) - 2 C(K) + C(K - x).
//   At maturity the spread pays (S - K + x)^+ - 2 (S - K)^+ + (S - K - x)^+,
// which is the sum of the payoffs of its parts. Two portfolios with the same
// payoff at T have the same value at every t <= T (otherwise buy the cheap
// one, sell the expensive one, and lock in the difference: an arbitrage), so
// the spread is worth the sum of the values of its parts:
//   V_butterfly = C(K - x) - 2 C(K) + C(K + x).
// Explicitly the payoff is the "tent" (x - |S - K|)^+: 0 for S <= K - x,
// S - (K - x) on [K - x, K], (K + x) - S on [K, K + x], and 0 for S >= K + x.
//
// (ii) Show that the value of 1/x^2 butterflies tends to d2C/dK2 (K) as x -> 0.
//   For a call price that is C^4 in the strike (Black-Scholes with t < T is
// smooth in K), Taylor's formula in K gives
//   C(K +- x) = C +- x C_K + x^2/2 C_KK +- x^3/6 C_KKK + x^4/24 C_KKKK +
//   O(x^5),
// and the odd terms cancel in the sum:
//   [C(K + x) - 2 C(K) + C(K - x)] / x^2 = C_KK(K) + x^2/12 C_KKKK(K) + O(x^4)
//                                        -> C_KK(K).
// This is the central second difference (7.9) applied in the strike. For
// Black-Scholes, dC/dK = -e^(-rT) N(d2) (the terms with phi cancel by
// S e^(-qT) phi(d1) = K e^(-rT) phi(d2)) and d(d2)/dK = -1/(K sigma sqrt(T)),
// so
//   d2C/dK2 = e^(-rT) phi(d2) / (K sigma sqrt(T)) > 0.
// (Breeden-Litzenberger: e^(rT) C_KK(K) is the risk-neutral density of S(T)
// at K.)
//
// (iii) Show that as x -> 0 the payoff of 1/x butterflies tends to the payoff
// of a security paying 1 if S(T) = K and 0 otherwise.
//   The payoff of 1/x butterflies is (x - |S - K|)^+ / x = (1 - |S - K|/x)^+.
// At S = K it equals 1 for every x. For S != K it is 0 as soon as
// x <= |S - K|. So the payoffs converge pointwise to 1_{S = K}, the
// Arrow-Debreu security for the state S(T) = K. (Under the lognormal model
// that single state has probability 0, and indeed the value of 1/x
// butterflies, ~ x C_KK(K), tends to 0; the density of Arrow-Debreu prices is
// C_KK, the limit in (ii).)
template <std::invocable<double> Call>
[[nodiscard]] double Ex06ButterflyValue(Call&& call, double strike, double x) {
  return call(strike + x) - 2.0 * call(strike) + call(strike - x);
}
[[nodiscard]] inline double Ex06ButterflyPayoff(double spot_t, double strike,
                                                double x) {
  return std::max(spot_t - strike + x, 0.0) -
         2.0 * std::max(spot_t - strike, 0.0) +
         std::max(spot_t - strike - x, 0.0);
}
// (ii) value of 1/x^2 butterflies, and its limit for Black-Scholes.
[[nodiscard]] inline double Ex06ScaledButterfly(double x,
                                                const SpreadMarket& m = {}) {
  return Ex06ButterflyValue([&](double k) { return ExCall(k, m); }, m.strike,
                            x) /
         (x * x);
}
[[nodiscard]] inline double Ex06CallKk(const SpreadMarket& m = {}) {
  const auto [d1, d2] =
      ch03::BsD1D2(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q);
  return std::exp(-m.r * m.maturity) * NormPdf(d2) /
         (m.strike * m.sigma * std::sqrt(m.maturity));
}
// (iii) payoff of 1/x butterflies.
[[nodiscard]] inline double Ex06ArrowDebreuPayoff(double spot_t, double strike,
                                                  double x) {
  return Ex06ButterflyPayoff(spot_t, strike, x) / x;
}

}  // namespace p4mfe::ch07::ex
