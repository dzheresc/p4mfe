#pragma once
// Chapter 3, sections 3.3-3.4: the standard normal variable, normal and
// lognormal random variables, and the lognormal model for stock prices that
// underlies the Black-Scholes formula.
//
// The standard normal density and cdf themselves, phi and N, are the shared
// helpers p4mfe::norm_pdf and p4mfe::norm_cdf (cpp/common). Section 3.9's
// approximations of N are in cdf_approx.hpp.

#include <p4mfe/normal.hpp>

#include <cmath>
#include <stdexcept>

namespace p4mfe::ch03 {

using p4mfe::norm_cdf;
using p4mfe::norm_pdf;

// ---------------------------------------------------------------- 3.3 ------
// Z standard normal: E[Z] = 0, var(Z) = 1, and its moment generating function
// E[e^(tZ)] = e^(t^2/2), used to get the lognormal moments below.
[[nodiscard]] inline double std_normal_mgf(double t) noexcept { return std::exp(0.5 * t * t); }

// ---------------------------------------------------------------- 3.4 ------
// X = mu + sigma Z ~ N(mu, sigma^2).
struct NormalParams {
    double mean;
    double variance;
    [[nodiscard]] double stddev() const { return std::sqrt(variance); }
};

[[nodiscard]] inline double normal_pdf(double x, double mu, double sigma) {
    if (sigma <= 0.0) throw std::invalid_argument("normal_pdf: sigma must be positive");
    return norm_pdf((x - mu) / sigma) / sigma;
}
// P(X <= x) = N((x - mu) / sigma).
[[nodiscard]] inline double normal_cdf(double x, double mu, double sigma) {
    if (sigma <= 0.0) throw std::invalid_argument("normal_cdf: sigma must be positive");
    return norm_cdf((x - mu) / sigma);
}

// a X + b Y for jointly normal X, Y with correlation rho is normal with
//   mean a mu_X + b mu_Y,  variance a^2 s_X^2 + b^2 s_Y^2 + 2 a b rho s_X s_Y.
[[nodiscard]] inline NormalParams linear_combination(double a, NormalParams X, double b, NormalParams Y,
                                                     double rho = 0.0) {
    return {a * X.mean + b * Y.mean,
            a * a * X.variance + b * b * Y.variance + 2.0 * a * b * rho * X.stddev() * Y.stddev()};
}

// Y = e^X with X ~ N(mu, sigma^2) is lognormal:
//   E[Y] = e^(mu + sigma^2/2),  var(Y) = e^(2 mu + sigma^2) (e^(sigma^2) - 1),
//   density f(y) = phi((ln y - mu)/sigma) / (sigma y),  y > 0.
[[nodiscard]] inline double lognormal_mean(double mu, double sigma) noexcept {
    return std::exp(mu + 0.5 * sigma * sigma);
}
[[nodiscard]] inline double lognormal_variance(double mu, double sigma) noexcept {
    const double s2 = sigma * sigma;
    return std::exp(2.0 * mu + s2) * std::expm1(s2);
}
[[nodiscard]] inline double lognormal_pdf(double y, double mu, double sigma) {
    if (y <= 0.0) return 0.0;
    return normal_pdf(std::log(y), mu, sigma) / y;
}
[[nodiscard]] inline double lognormal_cdf(double y, double mu, double sigma) {
    if (y <= 0.0) return 0.0;
    return normal_cdf(std::log(y), mu, sigma);
}

// Lognormal model for the spot price: with drift mu, volatility sigma and
// dividend yield q,
//   ln(S_T) ~ N(ln S_0 + (mu - q - sigma^2/2) T, sigma^2 T),
// so E[S_T] = S_0 e^((mu - q) T). Under the risk-neutral measure mu = r.
[[nodiscard]] inline NormalParams log_price_distribution(double S0, double T, double sigma, double mu,
                                                         double q = 0.0) {
    return {std::log(S0) + (mu - q - 0.5 * sigma * sigma) * T, sigma * sigma * T};
}

// P(S_T > K) = N(d), d = (ln(S_0/K) + (mu - q - sigma^2/2) T) / (sigma sqrt(T));
// with mu = r this is N(d2), the risk-neutral probability that a call
// finishes in the money.
[[nodiscard]] inline double prob_price_above(double K, double S0, double T, double sigma, double mu,
                                             double q = 0.0) {
    const auto ln_ST = log_price_distribution(S0, T, sigma, mu, q);
    return 1.0 - normal_cdf(std::log(K), ln_ST.mean, ln_ST.stddev());
}

}  // namespace p4mfe::ch03
