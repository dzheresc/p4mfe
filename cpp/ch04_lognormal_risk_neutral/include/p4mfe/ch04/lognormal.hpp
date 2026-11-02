#pragma once
// Chapter 4, section 4.2: lognormal random variables.
//
// Y is lognormal with parameters mu and sigma if ln(Y) ~ N(mu, sigma^2), i.e.
// Y = e^(mu + sigma Z) with Z standard normal. Its density
//   f_Y(y) = 1 / (sigma y sqrt(2 pi)) exp(-(ln y - mu)^2 / (2 sigma^2)),  y >
//   0,
// is the change of variables of section 4.1 applied to Y = e^X.
//
// The density, cdf, mean and variance already live in chapter 3
// (ch03::lognormal_pdf, lognormal_cdf, lognormal_mean, lognormal_variance)
// and are re-exported here. This header adds the rest of the picture: all
// moments from the normal moment generating function, the median, mode and
// quantiles, and the inverse problem of finding mu and sigma from a given
// mean and variance.

#include <cmath>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch04 {

using ch03::LognormalCdf;
using ch03::LognormalMean;
using ch03::LognormalPdf;
using ch03::LognormalVariance;
using ch03::NormalParams;

// E[Y^n] = E[e^(n X)] = e^(n mu + n^2 sigma^2 / 2), from the moment generating
// function of the normal X = ln Y. n need not be an integer. n = 1 and n = 2
// give the mean and, through E[Y^2] - E[Y]^2, the variance.
[[nodiscard]] inline double LognormalMoment(double n, double mu,
                                            double sigma) noexcept {
  return std::exp(n * mu + 0.5 * n * n * sigma * sigma);
}

// The p-quantile: P(Y <= y_p) = p  <=>  y_p = e^(mu + sigma N^(-1)(p)).
// Monotone transforms carry quantiles over, which is why this is easy while
// the mean is not simply e^(E[X]).
[[nodiscard]] inline double LognormalQuantile(double p, double mu,
                                              double sigma) {
  if (!(p > 0.0 && p < 1.0))
    throw std::domain_error("lognormal_quantile: need 0 < p < 1");
  return std::exp(mu + sigma * norm_inv(p));
}

// Median e^mu (the 0.5-quantile), below the mean e^(mu + sigma^2/2); the mode
// (argmax of the density) is lower still, e^(mu - sigma^2).
[[nodiscard]] inline double LognormalMedian(double mu) noexcept {
  return std::exp(mu);
}
[[nodiscard]] inline double LognormalMode(double mu, double sigma) noexcept {
  return std::exp(mu - sigma * sigma);
}

// Inverse problem: the parameters of ln(Y) for a lognormal Y with given mean m
// and variance v,
//   sigma^2 = ln(1 + v / m^2),  mu = ln(m) - sigma^2 / 2.
[[nodiscard]] inline NormalParams LognormalParamsFromMoments(double m,
                                                             double v) {
  if (!(m > 0.0 && v > 0.0))
    throw std::invalid_argument("lognormal_params_from_moments: need m, v > 0");
  const double s2 = std::log1p(v / (m * m));
  return {std::log(m) - 0.5 * s2, s2};
}

}  // namespace p4mfe::ch04
