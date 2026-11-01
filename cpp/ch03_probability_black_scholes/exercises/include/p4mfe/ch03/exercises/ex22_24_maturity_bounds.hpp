#pragma once
// Exercises 22-24 (section 3.11): Gamma and vega of ATM calls across
// maturities, and no-arbitrage bounds for calls and puts.

#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>

#include <array>
#include <cmath>

namespace p4mfe::ch03::ex {

// Fifteen days, three months and one year. Fifteen days is taken as half a
// month, 1/24 years (15/365 = 0.0411 would raise Gamma and lower vega by
// about 0.7%).
inline constexpr std::array<double, 3> ex22_maturities{1.0 / 24.0, 0.25, 1.0};

// ----------------------------------------------------------- Exercise 22 ---
// Compute the Gamma of ATM calls with maturities of fifteen days, three
// months and one year on a non-dividend-paying asset with spot 50 and
// volatility 30%, with r = 5%. What can you infer about the hedging of ATM
// options with different maturities?
//
// Gamma = phi(d1) / (S sigma sqrt(T)), d1 = (r + sigma^2/2) sqrt(T)/sigma
// for K = S:
//   T = 1/24:  Gamma = 0.130022
//   T = 1/4:   Gamma = 0.052530
//   T = 1:     Gamma = 0.025296
// Since phi(d1) -> 1/sqrt(2 pi) as T -> 0, ATM Gamma grows like 1/sqrt(T):
// short-dated ATM options have the largest Gamma. Their Delta changes fastest
// with the spot, so a Delta hedge on a short-dated ATM option goes stale
// quickly and must be rebalanced much more often (and Delta-Gamma hedging
// matters most) than for long-dated options.
[[nodiscard]] inline double ex22_atm_gamma(double T, double S = 50.0, double sigma = 0.3, double r = 0.05) {
    return bs_greeks(OptionType::Call, S, S, T, sigma, r).gamma;
}

// ----------------------------------------------------------- Exercise 23 ---
// (i)   vega(C) = vega(P) = S e^(-q(T-t)) sqrt(T-t) phi(d1) > 0. Give a
//       financial explanation.
// (ii)  Compute the vega of ATM calls with maturities fifteen days, three
//       months and one year, S = 50, sigma = 30%, q = 0, r = 0.
// (iii) For r = q = 0 and K = S, vega = S sqrt(T-t) phi(d1) with
//       d1 = sigma sqrt(T-t)/2. Compute d(vega)/d(T-t) and explain (ii).
//
// (i) An option's payoff is convex in S_T (and the holder's loss is capped at
// the premium), so wider dispersion of S_T raises the expected payoff: more
// upside for a call (or downside protection for a put) is gained than is lost
// on the other side, where the payoff is floored at zero. Higher volatility
// means a more dispersed S_T, so a more valuable option (Jensen's inequality).
//
// (ii) With tau = T - t, d1 = 0.15 sqrt(tau):
//   tau = 1/24:  vega = 4.0698
//   tau = 1/4:   vega = 9.9455
//   tau = 1:     vega = 19.7240
// (per unit of sigma; per 1 vol point divide by 100: 0.041, 0.099, 0.197).
//
// (iii) d(vega)/d(tau) = S [phi(d1)/(2 sqrt(tau)) + sqrt(tau) phi'(d1) d(d1)/d(tau)]
//                      = S [phi(d1)/(2 sqrt(tau)) - sqrt(tau) d1 phi(d1) sigma/(4 sqrt(tau))]
//                      = S phi(d1) / (2 sqrt(tau)) * (1 - sigma^2 tau / 4).
// This is positive for tau < 4/sigma^2 (= 44.4 years for sigma = 0.3), so for
// all realistic maturities ATM vega increases with maturity, roughly like
// sqrt(tau): the 1-year vega is about sqrt(24) = 4.9 times the 15-day vega
// (4.85 exactly), as in (ii). Long-dated options are the most exposed to
// volatility.
[[nodiscard]] inline double ex23_atm_vega(double T, double S = 50.0, double sigma = 0.3, double r = 0.0) {
    return bs_greeks(OptionType::Call, S, S, T, sigma, r).vega;
}
[[nodiscard]] inline double ex23_vega_maturity_slope(double tau, double S = 50.0, double sigma = 0.3) {
    const double d1 = 0.5 * sigma * std::sqrt(tau);
    return S * norm_pdf(d1) / (2.0 * std::sqrt(tau)) * (1.0 - sigma * sigma * tau / 4.0);
}

// ----------------------------------------------------------- Exercise 24 ---
// With constant rate r, show that unless
//   S e^(-qT) - K e^(-rT) <= C <= S e^(-qT),
// arbitrage opportunities arise. Show that the put must satisfy
//   K e^(-rT) - S e^(-qT) <= P <= K e^(-rT).
// (The book says "non-dividend-paying", but the bounds carry the dividend
// yield q; with q = 0 they reduce to S - K e^(-rT) <= C <= S.)
//
// Call, upper bound: if C > S e^(-qT), sell the call and buy e^(-qT) shares
// (dividends reinvested), pocketing C - S e^(-qT) > 0. At T you hold one
// share, worth S_T >= max(S_T - K, 0), enough to cover the call.
// Call, lower bound: if C < S e^(-qT) - K e^(-rT), buy the call, short
// e^(-qT) shares and lend K e^(-rT), pocketing a positive amount. At T:
// max(S_T - K, 0) - S_T + K >= 0.
// Put, upper bound: if P > K e^(-rT), sell the put and lend K e^(-rT); at T
// you have K >= max(K - S_T, 0).
// Put, lower bound: if P < K e^(-rT) - S e^(-qT), buy the put and e^(-qT)
// shares, borrow K e^(-rT), pocketing a positive amount. At T:
// max(K - S_T, 0) + S_T - K >= 0.
// Equivalently, the put bounds follow from the call bounds and Put-Call parity
// P = C - S e^(-qT) + K e^(-rT): subtract S e^(-qT) - K e^(-rT) throughout,
// and C >= 0 gives the positive part of the lower bound.
// Black-Scholes prices always lie inside (ch01::call_bounds / put_bounds).
[[nodiscard]] inline ch01::PriceBounds ex24_put_bounds(double S, double K, double T, double r, double q = 0.0) {
    return {K * std::exp(-r * T) - S * std::exp(-q * T), K * std::exp(-r * T)};
}
[[nodiscard]] inline ch01::PriceBounds ex24_call_bounds(double S, double K, double T, double r, double q = 0.0) {
    return {S * std::exp(-q * T) - K * std::exp(-r * T), S * std::exp(-q * T)};
}

}  // namespace p4mfe::ch03::ex
