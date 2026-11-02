#pragma once
// Exercises 9-10 (section 4.9): real-world and risk-neutral probabilities
// that a put expires in the money, and when the Delta of an at-the-money
// call is above or below 0.5.

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch04::ex {

// ------------------------------------------------------------ Exercise 9 ---
// Put with strike 55 and maturity 4 months on a non-dividend-paying asset
// with spot 60, lognormal with drift mu = 0.1 and volatility sigma = 0.3;
// the risk-free rate is constant at 0.05.
// (i) Find the probability that the put expires in the money.
// (ii) Find the risk-neutral probability that the put expires in the money.
// (iii) Compute N(-d2).
//
// The put is in the money when S_T < K:
//   P(S_T < K) = P(ln(S_T/S_0) < ln(K/S_0)) = N(-(ln(S_0/K) + (mu - sigma^2/2)
//   T)/(sigma sqrt(T))).
// T = 4/12:
// (i)   mu = 0.1:  P = N(-0.608208) = 0.271525.
// (ii)  mu = r = 0.05: P = N(-0.511983) = 0.304331.
// (iii) d2 = (ln(60/55) + (0.05 - 0.045) / 3) / (0.3 sqrt(1/3)) = 0.511983,
//       N(-d2) = 0.304331: the same number as (ii). In the Black-Scholes
//       formula P = K e^(-rT) N(-d2) - S N(-d1), N(-d2) is exactly the
//       risk-neutral probability of exercise, not the real-world one; the
//       higher real drift makes the put less likely to finish in the money.
struct Ex09Market {
  double S = 60.0, K = 55.0, T = 4.0 / 12.0, mu = 0.1, sigma = 0.3, r = 0.05;
};
[[nodiscard]] inline double ex09_prob_itm(const Ex09Market& m = {}) {
  return prob_itm(OptionType::kPut, m.S, m.K, m.T, m.sigma, m.mu);
}
[[nodiscard]] inline double ex09_rn_prob_itm(const Ex09Market& m = {}) {
  return prob_itm(OptionType::kPut, m.S, m.K, m.T, m.sigma, m.r);
}
[[nodiscard]] inline double ex09_n_minus_d2(const Ex09Market& m = {}) {
  return norm_cdf(-ch03::BsD1D2(m.S, m.K, m.T, m.sigma, m.r).d2);
}

// ----------------------------------------------------------- Exercise 10 ---
// (i) Consider an at-the-money call on a non-dividend-paying asset in the
//     Black-Scholes framework. Show that its Delta is always greater than 0.5.
// (ii) If the asset pays dividends continuously at rate q, when is the Delta
//      of an at-the-money call less than 0.5?
//
// (i) At the money (K = S) with q = 0, ln(S/K) = 0 and
//   d1 = (r + sigma^2/2) T / (sigma sqrt(T)) = (r/sigma + sigma/2) sqrt(T) > 0
// for r >= 0 (indeed for any r > -sigma^2/2). N is increasing with N(0) = 1/2,
// so Delta(C) = N(d1) > 1/2.
//
// (ii) With dividends, Delta(C) = e^(-qT) N(d1), d1 = (r - q + sigma^2/2)
// sqrt(T)/sigma. Delta < 1/2  <=>  N(d1) < e^(qT)/2. Both factors decrease in
// q, from Delta = N((r/sigma + sigma/2) sqrt(T)) > 1/2 at q = 0 to 0 as q ->
// infinity, so there is a unique threshold q* and
//   Delta(ATM call) < 1/2  <=>  q > q*(r, sigma, T).
// A simple sufficient condition: q >= r + sigma^2/2 makes d1 <= 0, so
// N(d1) <= 1/2 and e^(-qT) < 1 give Delta < 1/2. The threshold is a bit below
// that; linearising e^(-qT) ~ 1 - qT and N(d1) ~ 1/2 + d1/sqrt(2 pi),
//   q* ~ (r + sigma^2/2) / (1 + sigma sqrt(pi T / 2)).
// E.g. r = 5%, sigma = 20%: q* = 6.21% (T = 3 months), 5.56% (T = 1 year),
// 4.28% (T = 5 years), so for long maturities the Delta can drop below 1/2
// even when q < r. As the Note says, for typical inputs Delta is close to 0.5.
[[nodiscard]] inline double ex10_atm_call_delta(double r, double q,
                                                double sigma, double T) {
  return std::exp(-q * T) *
         norm_cdf((r - q + 0.5 * sigma * sigma) * std::sqrt(T) / sigma);
}
// q* by bisection: Delta(ATM call) is decreasing in q.
[[nodiscard]] inline double ex10_threshold_q(double r, double sigma, double T) {
  if (!(r > -0.5 * sigma * sigma))
    throw std::domain_error("ex10_threshold_q: need r > -sigma^2/2");
  double lo = 0.0, hi = r + 0.5 * sigma * sigma;  // Delta(lo) > 1/2 > Delta(hi)
  for (int i = 0; i < 200 && hi - lo > 1e-15; ++i) {
    const double mid = 0.5 * (lo + hi);
    (ex10_atm_call_delta(r, mid, sigma, T) > 0.5 ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}
[[nodiscard]] inline double ex10_threshold_q_approx(double r, double sigma,
                                                    double T) {
  return (r + 0.5 * sigma * sigma) /
         (1.0 + sigma * std::sqrt(std::numbers::pi * T / 2.0));
}

}  // namespace p4mfe::ch04::ex
