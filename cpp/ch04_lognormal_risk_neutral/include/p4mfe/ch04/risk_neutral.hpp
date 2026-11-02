#pragma once
// Chapter 4, section 4.5: the risk-neutral derivation of the Black-Scholes
// formula.
//
// Under the risk-neutral measure the asset grows at the risk-free rate, so in
// the lognormal model of section 4.4 the drift mu is replaced by r:
//   S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z).
// The value at time 0 of a European derivative with payoff g(S_T) is its
// discounted risk-neutral expectation
//   V = e^(-rT) E_RN[g(S_T)] = e^(-rT) integral g(S_T(z)) phi(z) dz.
//
// For the call g = (S_T - K)^+ the integral is evaluated in closed form by
// keeping only z > -d2 (where S_T > K) and completing the square in the term
// with S_T:
//   E_RN[(S_T - K)^+] = S_0 e^((r - q) T) N(d1) - K N(d2),
// which, discounted, is the Black-Scholes formula of chapter 3.
//
// Three routes to the same number are provided: the closed form, numerical
// integration of the expectation, and Monte Carlo simulation of Z.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/normal.hpp>
#include <random>
#include <stdexcept>

namespace p4mfe::ch04 {

using ch01::OptionType;

// E_RN[g(S_T)] by integrating against the standard normal density (with
// ch03::ContinuousDistribution, so g is not evaluated where phi underflows).
template <std::invocable<double> Payoff>
[[nodiscard]] double risk_neutral_expectation(Payoff&& g, double S0, double T,
                                              double sigma, double r,
                                              double q = 0.0,
                                              double tol = 1e-9) {
  const ch03::ContinuousDistribution Z{[](double z) { return norm_pdf(z); },
                                       -ch02::infinity, ch02::infinity};
  return ch03::expected_value(
      Z, [&](double z) { return g(price_from_normal(z, S0, T, sigma, r, q)); },
      tol);
}

// V = e^(-rT) E_RN[g(S_T)].
template <std::invocable<double> Payoff>
[[nodiscard]] double risk_neutral_price(Payoff&& g, double S0, double T,
                                        double sigma, double r, double q = 0.0,
                                        double tol = 1e-9) {
  return std::exp(-r * T) *
         risk_neutral_expectation(g, S0, T, sigma, r, q, tol);
}

// The closed forms from completing the square:
//   E_RN[(S_T - K)^+] = S_0 e^((r - q) T) N(d1) - K N(d2)
//   E_RN[(K - S_T)^+] = K N(-d2) - S_0 e^((r - q) T) N(-d1)
[[nodiscard]] inline double rn_expected_payoff(OptionType type, double S0,
                                               double K, double T, double sigma,
                                               double r, double q = 0.0) {
  const auto [d1, d2] = ch03::bs_d1_d2(S0, K, T, sigma, r, q);
  const double F = S0 * std::exp((r - q) * T);  // forward price = E_RN[S_T]
  return type == OptionType::kCall ? F * norm_cdf(d1) - K * norm_cdf(d2)
                                   : K * norm_cdf(-d2) - F * norm_cdf(-d1);
}

// Monte Carlo estimate of e^(-rT) E_RN[g(S_T)] with n independent draws of Z,
// and its standard error (sample standard deviation / sqrt(n)). Antithetic
// pairs (z, -z) are used when `antithetic` is set, which keeps the estimate
// unbiased and lowers its variance for monotone payoffs.
struct MonteCarloResult {
  double price;
  double std_error;
};

template <std::invocable<double> Payoff>
[[nodiscard]] MonteCarloResult monte_carlo_price(
    Payoff&& g, double S0, double T, double sigma, double r, double q,
    std::size_t n, std::uint64_t seed = 42, bool antithetic = true) {
  if (n < 2) throw std::invalid_argument("monte_carlo_price: need n >= 2");
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double disc = std::exp(-r * T);
  // Welford's running mean and variance of the (discounted) samples.
  double mean = 0.0, m2 = 0.0;
  for (std::size_t i = 1; i <= n; ++i) {
    const double z = normal(rng);
    double x = g(price_from_normal(z, S0, T, sigma, r, q));
    if (antithetic)
      x = 0.5 * (x + g(price_from_normal(-z, S0, T, sigma, r, q)));
    x *= disc;
    const double delta = x - mean;
    mean += delta / static_cast<double>(i);
    m2 += delta * (x - mean);
  }
  const double var = m2 / static_cast<double>(n - 1);
  return {mean, std::sqrt(var / static_cast<double>(n))};
}

}  // namespace p4mfe::ch04
