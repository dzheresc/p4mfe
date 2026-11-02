#pragma once
// Chapter 4, sections 4.6-4.7: the probability that an option expires in the
// money, and the financial meaning of N(d1) and N(d2).
//
// 4.6. In the lognormal model with drift mu,
//   P(S_T > K) = N(d2(mu)),  d2(mu) = (ln(S_0/K) + (mu - q - sigma^2/2) T) /
//   (sigma sqrt(T)),
// and P(S_T < K) = N(-d2(mu)). This is the real-world probability; the
// risk-neutral one, which is what enters prices, takes mu = r and gives the
// N(d2) of the Black-Scholes formula.
//
// 4.7. The Black-Scholes call C = S e^(-qT) N(d1) - K e^(-rT) N(d2) splits
// into two digital (binary) options:
//   * cash-or-nothing: pays 1 if S_T > K; worth e^(-rT) N(d2), so N(d2) is the
//     risk-neutral probability of exercise;
//   * asset-or-nothing: pays S_T if S_T > K; worth S e^(-qT) N(d1).
// The call is long one asset-or-nothing and short K cash-or-nothing. N(d1) is
//   N(d1) = E_RN[S_T 1{S_T > K}] / E_RN[S_T],
// the probability of exercise when outcomes are weighted by the asset price
// (the probability under the measure with the asset as numeraire), and
// e^(-qT) N(d1) is the call's Delta: the number of shares to hold to
// replicate it. Likewise dC/dK = -e^(-rT) N(d2).

#include <cmath>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch04 {

using ch01::OptionType;

// ---------------------------------------------------------------- 4.6 ------
// P(option finishes in the money) under drift mu: P(S_T > K) for a call,
// P(S_T < K) for a put. Pass mu = r for the risk-neutral probability.
[[nodiscard]] inline double prob_itm(OptionType type, double S0, double K,
                                     double T, double sigma, double mu,
                                     double q = 0.0) {
  const double above = ch03::prob_price_above(K, S0, T, sigma, mu, q);
  return type == OptionType::kCall ? above : 1.0 - above;
}

// P(K1 < S_T < K2), K1 < K2, under drift mu.
[[nodiscard]] inline double prob_between(double K1, double K2, double S0,
                                         double T, double sigma, double mu,
                                         double q = 0.0) {
  return ch03::prob_price_above(K1, S0, T, sigma, mu, q) -
         ch03::prob_price_above(K2, S0, T, sigma, mu, q);
}

// ---------------------------------------------------------------- 4.7 ------
// Cash-or-nothing: pays 1 if the option finishes in the money.
//   call: e^(-rT) N(d2),  put: e^(-rT) N(-d2).
[[nodiscard]] inline double cash_or_nothing(OptionType type, double S, double K,
                                            double T, double sigma, double r,
                                            double q = 0.0) {
  const auto [d1, d2] = ch03::bs_d1_d2(S, K, T, sigma, r, q);
  return std::exp(-r * T) * norm_cdf(type == OptionType::kCall ? d2 : -d2);
}

// Asset-or-nothing: pays S_T if the option finishes in the money.
//   call: S e^(-qT) N(d1),  put: S e^(-qT) N(-d1).
[[nodiscard]] inline double asset_or_nothing(OptionType type, double S,
                                             double K, double T, double sigma,
                                             double r, double q = 0.0) {
  const auto [d1, d2] = ch03::bs_d1_d2(S, K, T, sigma, r, q);
  return S * std::exp(-q * T) * norm_cdf(type == OptionType::kCall ? d1 : -d1);
}

// The two probabilities of the call, side by side:
//   n_d2 = N(d2)  risk-neutral probability that S_T > K,
//   n_d1 = N(d1)  same event, outcomes weighted by S_T (share measure);
// always n_d1 > n_d2.
struct ExerciseProbabilities {
  double n_d1;
  double n_d2;
};
[[nodiscard]] inline ExerciseProbabilities exercise_probabilities(
    double S, double K, double T, double sigma, double r, double q = 0.0) {
  const auto [d1, d2] = ch03::bs_d1_d2(S, K, T, sigma, r, q);
  return {norm_cdf(d1), norm_cdf(d2)};
}

}  // namespace p4mfe::ch04
