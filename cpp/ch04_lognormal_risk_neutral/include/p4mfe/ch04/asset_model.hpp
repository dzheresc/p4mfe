#pragma once
// Chapter 4, section 4.4: the lognormal model for asset prices.
//
// The price S_t of an asset with drift mu, volatility sigma and continuous
// dividend yield q is assumed lognormal:
//   ln(S_T / S_0) ~ N((mu - q - sigma^2/2) T, sigma^2 T),
// i.e.  S_T = S_0 exp((mu - q - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1).
// The -sigma^2/2 correction is what makes E[S_T] = S_0 e^((mu - q) T): the
// expected growth rate is mu - q, while the median growth rate is lower.
//
// ch03::log_price_distribution already gives the law of ln(S_T); here are
// the consequences used in practice: moments, density, quantiles and the
// symmetric-probability price range the book computes.

#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/normal.hpp>

#include <cmath>
#include <stdexcept>

namespace p4mfe::ch04 {

using ch03::log_price_distribution;
using ch03::NormalParams;

// ln(S_T / S_0) ~ N((mu - q - sigma^2/2) T, sigma^2 T), the log return.
[[nodiscard]] inline NormalParams log_return_distribution(double T, double sigma, double mu, double q = 0.0) {
    return {(mu - q - 0.5 * sigma * sigma) * T, sigma * sigma * T};
}

// S_T as a function of the standard normal draw z.
[[nodiscard]] inline double price_from_normal(double z, double S0, double T, double sigma, double mu,
                                              double q = 0.0) noexcept {
    return S0 * std::exp((mu - q - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
}

// E[S_T] = S_0 e^((mu - q) T),  var(S_T) = S_0^2 e^(2 (mu - q) T) (e^(sigma^2 T) - 1).
[[nodiscard]] inline double price_mean(double S0, double T, double mu, double q = 0.0) noexcept {
    return S0 * std::exp((mu - q) * T);
}
[[nodiscard]] inline double price_variance(double S0, double T, double sigma, double mu, double q = 0.0) noexcept {
    const double m = price_mean(S0, T, mu, q);
    return m * m * std::expm1(sigma * sigma * T);
}
// Median S_0 e^((mu - q - sigma^2/2) T) < mean.
[[nodiscard]] inline double price_median(double S0, double T, double sigma, double mu, double q = 0.0) noexcept {
    return price_from_normal(0.0, S0, T, sigma, mu, q);
}

// Density of S_T at x > 0: the lognormal density with the parameters above.
[[nodiscard]] inline double price_pdf(double x, double S0, double T, double sigma, double mu, double q = 0.0) {
    const auto L = log_price_distribution(S0, T, sigma, mu, q);
    return ch03::lognormal_pdf(x, L.mean, L.stddev());
}

// P(S_T <= x).
[[nodiscard]] inline double price_cdf(double x, double S0, double T, double sigma, double mu, double q = 0.0) {
    const auto L = log_price_distribution(S0, T, sigma, mu, q);
    return ch03::lognormal_cdf(x, L.mean, L.stddev());
}

// The p-quantile of S_T: price_from_normal(N^(-1)(p)).
[[nodiscard]] inline double price_quantile(double p, double S0, double T, double sigma, double mu,
                                           double q = 0.0) {
    if (!(p > 0.0 && p < 1.0)) throw std::domain_error("price_quantile: need 0 < p < 1");
    return price_from_normal(norm_inv(p), S0, T, sigma, mu, q);
}

// The range [lo, hi] that S_T falls in with probability `confidence`, with
// equal probability (1 - confidence)/2 of falling below lo or above hi:
//   lo, hi = S_0 exp((mu - q - sigma^2/2) T -/+ z sigma sqrt(T)),
//   z = N^(-1)((1 + confidence) / 2)   (1.96 for 95%, 2.576 for 99%).
// The range is not symmetric around S_0: the lognormal is skewed right.
struct PriceRange {
    double lo;
    double hi;
};
[[nodiscard]] inline PriceRange price_confidence_interval(double confidence, double S0, double T, double sigma,
                                                          double mu, double q = 0.0) {
    if (!(confidence > 0.0 && confidence < 1.0))
        throw std::domain_error("price_confidence_interval: need 0 < confidence < 1");
    const double z = norm_inv(0.5 * (1.0 + confidence));
    return {price_from_normal(-z, S0, T, sigma, mu, q), price_from_normal(z, S0, T, sigma, mu, q)};
}

}  // namespace p4mfe::ch04
