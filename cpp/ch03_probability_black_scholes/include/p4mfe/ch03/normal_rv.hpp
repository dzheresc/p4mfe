#pragma once
// Chapter 3, sections 3.3-3.4: the standard normal variable, normal and
// lognormal random variables, and the lognormal model for stock prices that
// underlies the Black-Scholes formula.
//
// The standard normal density and cdf themselves, phi and N, are the shared
// helpers p4mfe::NormPdf and p4mfe::NormCdf (cpp/common). Section 3.9's
// approximations of N are in cdf_approx.hpp.

#include <cmath>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch03 {

using p4mfe::NormCdf;
using p4mfe::NormPdf;

// ---------------------------------------------------------------- 3.3 ------
// Z standard normal: E[Z] = 0, var(Z) = 1, and its moment generating function
// E[e^(tZ)] = e^(t^2/2), used to get the lognormal moments below.
[[nodiscard]] inline double StdNormalMgf(double t) noexcept {
  return std::exp(0.5 * t * t);
}

// ---------------------------------------------------------------- 3.4 ------
// X = mu + sigma Z ~ N(mu, sigma^2).
struct NormalParams {
  double mean;
  double variance;
  [[nodiscard]] double Stddev() const { return std::sqrt(variance); }
};

[[nodiscard]] inline double NormalPdf(double x, double mu, double sigma) {
  if (sigma <= 0.0)
    throw std::invalid_argument("normal_pdf: sigma must be positive");
  return NormPdf((x - mu) / sigma) / sigma;
}
// P(X <= x) = N((x - mu) / sigma).
[[nodiscard]] inline double NormalCdf(double x, double mu, double sigma) {
  if (sigma <= 0.0)
    throw std::invalid_argument("normal_cdf: sigma must be positive");
  return NormCdf((x - mu) / sigma);
}

// a X + b Y for jointly normal X, Y with correlation rho is normal with
//   mean a mu_X + b mu_Y,  variance a^2 s_X^2 + b^2 s_Y^2 + 2 a b rho s_X s_Y.
[[nodiscard]] inline NormalParams LinearCombination(double a, NormalParams x,
                                                    double b, NormalParams y,
                                                    double rho = 0.0) {
  return {a * x.mean + b * y.mean,
          a * a * x.variance + b * b * y.variance +
              2.0 * a * b * rho * x.Stddev() * y.Stddev()};
}

// Y = e^X with X ~ N(mu, sigma^2) is lognormal:
//   E[Y] = e^(mu + sigma^2/2),  var(Y) = e^(2 mu + sigma^2) (e^(sigma^2) - 1),
//   density f(y) = phi((ln y - mu)/sigma) / (sigma y),  y > 0.
[[nodiscard]] inline double LognormalMean(double mu, double sigma) noexcept {
  return std::exp(mu + 0.5 * sigma * sigma);
}
[[nodiscard]] inline double LognormalVariance(double mu,
                                              double sigma) noexcept {
  const double s2 = sigma * sigma;
  return std::exp(2.0 * mu + s2) * std::expm1(s2);
}
[[nodiscard]] inline double LognormalPdf(double y, double mu, double sigma) {
  if (y <= 0.0) return 0.0;
  return NormalPdf(std::log(y), mu, sigma) / y;
}
[[nodiscard]] inline double LognormalCdf(double y, double mu, double sigma) {
  if (y <= 0.0) return 0.0;
  return NormalCdf(std::log(y), mu, sigma);
}

// Lognormal model for the spot price: with drift mu, volatility sigma and
// dividend yield q,
//   ln(S_T) ~ N(ln S_0 + (mu - q - sigma^2/2) T, sigma^2 T),
// so E[S_T] = S_0 e^((mu - q) T). Under the risk-neutral measure mu = r.
[[nodiscard]] inline NormalParams LogPriceDistribution(double spot0,
                                                       double maturity,
                                                       double sigma, double mu,
                                                       double q = 0.0) {
  return {std::log(spot0) + (mu - q - 0.5 * sigma * sigma) * maturity,
          sigma * sigma * maturity};
}

// P(S_T > K) = N(d), d = (ln(S_0/K) + (mu - q - sigma^2/2) T) / (sigma
// sqrt(T)); with mu = r this is N(d2), the risk-neutral probability that a call
// finishes in the money.
[[nodiscard]] inline double ProbPriceAbove(double strike, double spot0,
                                           double maturity, double sigma,
                                           double mu, double q = 0.0) {
  const auto ln_spot_t = LogPriceDistribution(spot0, maturity, sigma, mu, q);
  return 1.0 - NormalCdf(std::log(strike), ln_spot_t.mean, ln_spot_t.Stddev());
}

}  // namespace p4mfe::ch03
