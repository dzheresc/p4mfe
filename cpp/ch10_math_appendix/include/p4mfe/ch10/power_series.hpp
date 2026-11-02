#pragma once
// Chapter 10, section 10.6: power series.
//
// A power series sum_{k>=0} a_k (x - c)^k has a radius of convergence
//   R = 1 / limsup |a_k|^(1/k)   (root test),
//   R = lim |a_k / a_{k+1}|      (ratio test, when the limit exists):
// it converges absolutely for |x - c| < R and diverges for |x - c| > R.
// Inside (c - R, c + R) the sum is infinitely differentiable and may be
// differentiated and integrated term by term, with the same R:
//   (sum a_k (x-c)^k)'      = sum (k+1) a_{k+1} (x-c)^k,
//   int_c^x sum a_k (t-c)^k = sum a_k (x-c)^(k+1) / (k+1).
// Two series with the same center multiply by the Cauchy product
//   (a * b)_k = sum_{j=0}^{k} a_j b_{k-j}.
// Here a series is truncated to its first coefficients; chapter 6's
// exp_series, log1p_series, ... are particular partial sums.

#include <cmath>
#include <cstddef>
#include <limits>
#include <p4mfe/ch10/interpolation.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch10 {

struct PowerSeries {
  Vector a;  // a_0, a_1, ..., a_n
  double center = 0.0;

  // Partial sum sum_{k<=n} a_k (x - c)^k (Horner).
  [[nodiscard]] double operator()(double x) const noexcept {
    return polynomial_value(a, x - center);
  }
};

// Term-by-term derivative (one coefficient fewer).
[[nodiscard]] inline PowerSeries derivative(const PowerSeries& s) {
  PowerSeries d{{}, s.center};
  for (std::size_t k = 1; k < s.a.size(); ++k)
    d.a.push_back(static_cast<double>(k) * s.a[k]);
  if (d.a.empty()) d.a.push_back(0.0);
  return d;
}

// Term-by-term antiderivative vanishing at the center (one coefficient more).
[[nodiscard]] inline PowerSeries antiderivative(const PowerSeries& s) {
  PowerSeries d{{0.0}, s.center};
  for (std::size_t k = 0; k < s.a.size(); ++k)
    d.a.push_back(s.a[k] / static_cast<double>(k + 1));
  return d;
}

// Cauchy product, truncated to the shorter of the two lengths (the terms
// beyond it would be incomplete).
[[nodiscard]] inline PowerSeries cauchy_product(const PowerSeries& p,
                                                const PowerSeries& q) {
  if (p.center != q.center)
    throw std::invalid_argument("cauchy_product: series need the same center");
  const std::size_t n = std::min(p.a.size(), q.a.size());
  PowerSeries r{Vector(n, 0.0), p.center};
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t j = 0; j <= k; ++j) r.a[k] += p.a[j] * q.a[k - j];
  return r;
}

// Ratio-test estimate |a_{n-1} / a_n| from the last two coefficients
// (infinity if a_n = 0, i.e. the estimate does not apply).
[[nodiscard]] inline double radius_ratio_estimate(const Vector& a) {
  if (a.size() < 2)
    throw std::invalid_argument("radius_ratio_estimate: need two coefficients");
  const double last = a.back();
  if (last == 0.0) return std::numeric_limits<double>::infinity();
  return std::abs(a[a.size() - 2] / last);
}

// Root-test estimate 1 / |a_n|^(1/n) from the last nonzero coefficient, n >= 1.
[[nodiscard]] inline double radius_root_estimate(const Vector& a) {
  for (std::size_t n = a.size(); n-- > 1;)
    if (a[n] != 0.0)
      return std::pow(std::abs(a[n]), -1.0 / static_cast<double>(n));
  return std::numeric_limits<double>::infinity();
}

// Coefficients of classical series around 0, terms k = 0..n.
[[nodiscard]] inline PowerSeries exp_power_series(unsigned n) {
  PowerSeries s{Vector(n + 1), 0.0};
  double c = 1.0;
  for (unsigned k = 0; k <= n; ++k) {
    if (k > 0) c /= k;
    s.a[k] = c;
  }
  return s;
}
[[nodiscard]] inline PowerSeries geometric_power_series(unsigned n) {
  return {Vector(n + 1, 1.0), 0.0};
}
[[nodiscard]] inline PowerSeries log1p_power_series(unsigned n) {
  PowerSeries s{Vector(n + 1, 0.0), 0.0};
  for (unsigned k = 1; k <= n; ++k) s.a[k] = (k % 2 ? 1.0 : -1.0) / k;
  return s;
}

}  // namespace p4mfe::ch10
