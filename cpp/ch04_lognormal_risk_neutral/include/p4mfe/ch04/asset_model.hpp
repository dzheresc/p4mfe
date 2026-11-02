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

#include <cmath>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch04 {

using ch03::LogPriceDistribution;
using ch03::NormalParams;

// ln(S_T / S_0) ~ N((mu - q - sigma^2/2) T, sigma^2 T), the log return.
[[nodiscard]] inline NormalParams LogReturnDistribution(double maturity,
                                                        double sigma, double mu,
                                                        double q = 0.0) {
  return {(mu - q - 0.5 * sigma * sigma) * maturity, sigma * sigma * maturity};
}

// S_T as a function of the standard normal draw z.
[[nodiscard]] inline double PriceFromNormal(double z, double spot0,
                                            double maturity, double sigma,
                                            double mu,
                                            double q = 0.0) noexcept {
  return spot0 * std::exp((mu - q - 0.5 * sigma * sigma) * maturity +
                          sigma * std::sqrt(maturity) * z);
}

// E[S_T] = S_0 e^((mu - q) T),  var(S_T) = S_0^2 e^(2 (mu - q) T) (e^(sigma^2
// T) - 1).
[[nodiscard]] inline double PriceMean(double spot0, double maturity, double mu,
                                      double q = 0.0) noexcept {
  return spot0 * std::exp((mu - q) * maturity);
}
[[nodiscard]] inline double PriceVariance(double spot0, double maturity,
                                          double sigma, double mu,
                                          double q = 0.0) noexcept {
  const double m = PriceMean(spot0, maturity, mu, q);
  return m * m * std::expm1(sigma * sigma * maturity);
}
// Median S_0 e^((mu - q - sigma^2/2) T) < mean.
[[nodiscard]] inline double PriceMedian(double spot0, double maturity,
                                        double sigma, double mu,
                                        double q = 0.0) noexcept {
  return PriceFromNormal(0.0, spot0, maturity, sigma, mu, q);
}

// Density of S_T at x > 0: the lognormal density with the parameters above.
[[nodiscard]] inline double PricePdf(double x, double spot0, double maturity,
                                     double sigma, double mu, double q = 0.0) {
  const auto log_price = LogPriceDistribution(spot0, maturity, sigma, mu, q);
  return ch03::LognormalPdf(x, log_price.mean, log_price.Stddev());
}

// P(S_T <= x).
[[nodiscard]] inline double PriceCdf(double x, double spot0, double maturity,
                                     double sigma, double mu, double q = 0.0) {
  const auto log_price = LogPriceDistribution(spot0, maturity, sigma, mu, q);
  return ch03::LognormalCdf(x, log_price.mean, log_price.Stddev());
}

// The p-quantile of S_T: price_from_normal(N^(-1)(p)).
[[nodiscard]] inline double PriceQuantile(double p, double spot0,
                                          double maturity, double sigma,
                                          double mu, double q = 0.0) {
  if (!(p > 0.0 && p < 1.0))
    throw std::domain_error("price_quantile: need 0 < p < 1");
  return PriceFromNormal(NormInv(p), spot0, maturity, sigma, mu, q);
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
[[nodiscard]] inline PriceRange PriceConfidenceInterval(double confidence,
                                                        double spot0,
                                                        double maturity,
                                                        double sigma, double mu,
                                                        double q = 0.0) {
  if (!(confidence > 0.0 && confidence < 1.0))
    throw std::domain_error(
        "price_confidence_interval: need 0 < confidence < 1");
  const double z = NormInv(0.5 * (1.0 + confidence));
  return {PriceFromNormal(-z, spot0, maturity, sigma, mu, q),
          PriceFromNormal(z, spot0, maturity, sigma, mu, q)};
}

}  // namespace p4mfe::ch04
