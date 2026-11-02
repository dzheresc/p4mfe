#pragma once
// Chapter 6, sections 6.3-6.3.1: Taylor series expansions.
//
// If the Taylor remainder R_n(x) -> 0 as n -> infinity, f equals its Taylor
// series, f(x) = sum_{k>=0} f^(k)(a) (x-a)^k / k!, on the interval of
// convergence. Each function below returns the partial sum with terms
// k = 0..n of a classical expansion around 0 (6.3.1):
//
//   e^x        = sum x^k / k!                              all x
//   ln(1+x)    = sum_{k>=1} (-1)^(k+1) x^k / k             -1 < x <= 1
//   1/(1-x)    = sum x^k                                   |x| < 1
//   (1+x)^p    = sum C(p, k) x^k, C(p,k) = p(p-1)...(p-k+1)/k!   |x| < 1
//   sin x      = sum (-1)^k x^(2k+1) / (2k+1)!             all x
//   cos x      = sum (-1)^k x^(2k) / (2k)!                 all x
//
// and, integrating the series of e^(-t^2/2) term by term, the series of the
// standard normal cdf
//   N(x) = 1/2 + 1/sqrt(2 pi) sum (-1)^k x^(2k+1) / (2^k k! (2k+1)).
//
// For sin, cos and N the index k counts nonzero terms, so n = 2 gives
// x - x^3/6 + x^5/120 for sin x.

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace p4mfe::ch06 {

[[nodiscard]] inline double exp_series(double x, unsigned n) noexcept {
  double term = 1.0, sum = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    term *= x / k;
    sum += term;
  }
  return sum;
}

// Terms k = 1..n (there is no constant term).
[[nodiscard]] inline double log1p_series(double x, unsigned n) {
  if (!(x > -1.0 && x <= 1.0))
    throw std::domain_error("log1p_series: converges only for -1 < x <= 1");
  double power = 1.0, sum = 0.0;
  for (unsigned k = 1; k <= n; ++k) {
    power *= -x;
    sum -= power / k;
  }
  return sum;
}

[[nodiscard]] inline double geometric_series(double x, unsigned n) noexcept {
  double term = 1.0, sum = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    term *= x;
    sum += term;
  }
  return sum;
}

[[nodiscard]] inline double binomial_series(double x, double p,
                                            unsigned n) noexcept {
  double term = 1.0, sum = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    term *= (p - (k - 1)) / k * x;
    sum += term;
  }
  return sum;
}

[[nodiscard]] inline double sin_series(double x, unsigned n) noexcept {
  double term = x, sum = x;
  for (unsigned k = 1; k <= n; ++k) {
    term *= -x * x / ((2.0 * k) * (2.0 * k + 1.0));
    sum += term;
  }
  return sum;
}

[[nodiscard]] inline double cos_series(double x, unsigned n) noexcept {
  double term = 1.0, sum = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    term *= -x * x / ((2.0 * k - 1.0) * (2.0 * k));
    sum += term;
  }
  return sum;
}

// N(x) from the Taylor series of e^(-t^2/2), terms k = 0..n.
// t_k = (-1)^k x^(2k+1) / (2^k k!) satisfies t_k = t_{k-1} * (-x^2 / (2k)).
[[nodiscard]] inline double norm_cdf_series(double x, unsigned n) noexcept {
  double t = x, sum = x;
  for (unsigned k = 1; k <= n; ++k) {
    t *= -x * x / (2.0 * k);
    sum += t / (2.0 * k + 1.0);
  }
  return 0.5 + sum * std::numbers::inv_sqrtpi / std::numbers::sqrt2;
}

}  // namespace p4mfe::ch06
