#pragma once
// Exercises 12-15 (section 3.11): pricing with Black-Scholes, options with
// zero strike, rho of a put from Put-Call parity, and the Delta of an
// asset-or-nothing call.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch03::ex {

// ----------------------------------------------------------- Exercise 12 ---
// Use the Black-Scholes formula to price both a put and a call option with
// strike K = 45 expiring in six months on an underlying asset with spot price
// 50 and volatility 20% paying dividends continuously at 2%, if interest
// rates are constant at 6%.
//
// S = 50, K = 45, T = 0.5, sigma = 0.2, r = 0.06, q = 0.02:
//   d1 = (ln(50/45) + (0.06 - 0.02 + 0.02) * 0.5) / (0.2 sqrt(0.5)) = 0.957143,
//   d2 = d1 - 0.2 sqrt(0.5) = 0.815722,
//   C = 50 e^(-0.01) N(d1) - 45 e^(-0.03) N(d2) = 6.508365,
//   P = 45 e^(-0.03) N(-d2) - 50 e^(-0.01) N(-d1) = 0.675922.
// Check with Put-Call parity: C - P = 50 e^(-0.01) - 45 e^(-0.03) = 5.832443.
struct Ex12Market {
  double spot = 50.0, strike = 45.0, maturity = 0.5, sigma = 0.2, r = 0.06,
         q = 0.02;
};
[[nodiscard]] inline double Ex12Call(const Ex12Market& m = {}) {
  return BsCall(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q);
}
[[nodiscard]] inline double Ex12Put(const Ex12Market& m = {}) {
  return BsPut(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q);
}

// ----------------------------------------------------------- Exercise 13 ---
// What is the value of a European put option with strike K = 0? What is the
// value of a European call option with strike K = 0? How do you hedge a short
// position in such a call option? (No model for the asset is assumed: use the
// Law of One Price, not Black-Scholes.)
//
// Put: its payoff max(0 - S_T, 0) = 0 in every state, so P = 0.
// Call: its payoff max(S_T - 0, 0) = S_T, one unit of the asset at T. A
// position that is worth exactly one share at T is e^(-q(T-t)) shares bought
// today with the dividends reinvested in the asset; by the Law of One Price
//   C = S e^(-q(T-t))      (just S for a non-dividend-paying asset).
// This agrees with Put-Call parity, C - P = S e^(-qT) - 0 e^(-rT).
// Hedge for the short call: buy e^(-q(T-t)) shares with the premium received
// (it costs exactly that) and reinvest every dividend in the asset; at T you
// hold one share and deliver it. For q = 0: buy one share and hold it. This is
// a static hedge (Delta = e^(-q(T-t)) at all times, Gamma = 0).
[[nodiscard]] inline double Ex13PutZeroStrike() { return 0.0; }
[[nodiscard]] inline double Ex13CallZeroStrike(double spot, double maturity,
                                               double q = 0.0) {
  return spot * std::exp(-q * maturity);
}

// ----------------------------------------------------------- Exercise 14 ---
// Use rho(C) = K (T-t) e^(-r(T-t)) N(d2) and the Put-Call parity to obtain
//   rho(P) = -K (T-t) e^(-r(T-t)) N(-d2).
//
// Put-Call parity: P = C - S e^(-q(T-t)) + K e^(-r(T-t)). The term
// S e^(-q(T-t)) does not depend on r, and d/dr K e^(-r(T-t)) = -K (T-t)
// e^(-r(T-t)), so
//   rho(P) = rho(C) - K (T-t) e^(-r(T-t))
//          = K (T-t) e^(-r(T-t)) (N(d2) - 1) = -K (T-t) e^(-r(T-t)) N(-d2),
// using 1 - N(x) = N(-x).
[[nodiscard]] inline double Ex14RhoPutFromParity(double spot, double strike,
                                                 double maturity, double sigma,
                                                 double r, double q = 0.0) {
  return BsGreeks(OptionType::kCall, spot, strike, maturity, sigma, r, q).rho -
         strike * maturity * std::exp(-r * maturity);
}

// ----------------------------------------------------------- Exercise 15 ---
// The value of a European asset-or-nothing call with strike K and maturity T
// on a lognormal asset with volatility 30% paying dividends at rate q is
//   C_AoN = S e^(-qT) N(d1),  d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma
//   sqrt(T)).
// Find the Delta of the asset-or-nothing call.
//
// By the product rule, with d(d1)/dS = 1/(S sigma sqrt(T)):
//   Delta(C_AoN) = e^(-qT) N(d1) + S e^(-qT) phi(d1) / (S sigma sqrt(T))
//                = e^(-qT) (N(d1) + phi(d1) / (sigma sqrt(T))),
//   phi(d1) = e^(-d1^2/2)/sqrt(2 pi). With sigma = 0.3:
//   Delta(C_AoN) = e^(-qT) (N(d1) + phi(d1) / (0.3 sqrt(T))).
// The first term is the Delta of a plain call; the second, always positive,
// comes from the jump in the payoff at K (it is K times the Delta of the
// cash-or- nothing call K 1_{S_T > K}, since C_AoN = C + K * CashOrNothing).
[[nodiscard]] inline double Ex15AonCall(double spot, double strike,
                                        double maturity, double sigma, double r,
                                        double q = 0.0) {
  return spot * std::exp(-q * maturity) *
         norm_cdf(BsD1D2(spot, strike, maturity, sigma, r, q).d1);
}
[[nodiscard]] inline double Ex15AonDelta(double spot, double strike,
                                         double maturity, double sigma,
                                         double r, double q = 0.0) {
  const double d1 = BsD1D2(spot, strike, maturity, sigma, r, q).d1;
  return std::exp(-q * maturity) *
         (norm_cdf(d1) + norm_pdf(d1) / (sigma * std::sqrt(maturity)));
}

}  // namespace p4mfe::ch03::ex
