#pragma once
// Chapter 10, section 10.3: useful sums with interesting proofs.
//
// 10.3.1  S_i(n) = sum_{k=1}^{n} k^i.
//   S_1 = n(n+1)/2,  S_2 = n(n+1)(2n+1)/6,  S_3 = (n(n+1)/2)^2.
//   In general, summing (k+1)^(i+1) - k^(i+1) = sum_{j=0}^{i} C(i+1, j) k^j
//   over k = 1..n telescopes to
//     (n+1)^(i+1) - 1 = sum_{j=0}^{i} C(i+1, j) S_j(n),
//   which gives S_i from S_0, ..., S_{i-1}. S_i is a polynomial of degree i+1
//   in n, so it is also the interpolating polynomial (10.2) through
//   n = 0, 1, ..., i+1.
//
// 10.3.2  sum_{k=1}^{n} k^j x^k. Starting from the geometric sum
//   sum_{k=1}^{n} x^k = x (1 - x^n) / (1 - x),
//   each application of x d/dx multiplies the k-th term by k:
//   sum k   x^k = x (1 - (n+1) x^n + n x^(n+1)) / (1-x)^2,
//   sum k^2 x^k = x (1 + x - (n+1)^2 x^n + (2n^2+2n-1) x^(n+1) - n^2 x^(n+2)) /
//   (1-x)^3. For |x| < 1 and n -> infinity, sum_{k>=1} k^j x^k = P_j(x) /
//   (1-x)^(j+1) with P_0 = x and P_{j+1}(x) = x (P_j'(x)(1-x) + (j+1) P_j(x)),
//   so sum k x^k = x/(1-x)^2 and sum k^2 x^k = x(1+x)/(1-x)^3. With x = e^(-y)
//   a discount factor these are the sums behind bond duration and convexity.

#include <cmath>
#include <cstdint>
#include <p4mfe/ch10/interpolation.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch10 {

// ------------------------------------------------------------- 10.3.1 ------
[[nodiscard]] constexpr std::uint64_t Ipow(std::uint64_t b,
                                           unsigned e) noexcept {
  std::uint64_t r = 1;
  while (e--) r *= b;
  return r;
}

[[nodiscard]] constexpr std::uint64_t Binomial(unsigned n,
                                               unsigned k) noexcept {
  if (k > n) return 0;
  std::uint64_t r = 1;
  for (unsigned j = 1; j <= k; ++j)
    r = r * (n - k + j) / j;  // exact at every step
  return r;
}

// sum_{k=1}^{n} k^i by direct summation (exact while it fits in 64 bits).
[[nodiscard]] constexpr std::uint64_t SumOfPowers(std::uint64_t n,
                                                  unsigned i) noexcept {
  std::uint64_t s = 0;
  for (std::uint64_t k = 1; k <= n; ++k) s += Ipow(k, i);
  return s;
}

[[nodiscard]] constexpr std::uint64_t SumK(std::uint64_t n) noexcept {
  return n * (n + 1) / 2;
}
[[nodiscard]] constexpr std::uint64_t SumK2(std::uint64_t n) noexcept {
  return n * (n + 1) * (2 * n + 1) / 6;
}
[[nodiscard]] constexpr std::uint64_t SumK3(std::uint64_t n) noexcept {
  return SumK(n) * SumK(n);
}

// S_0(n), ..., S_i(n) from the telescoping recursion, without summing over k.
// Exact while (n+1)^(i+1) fits in 64 bits.
[[nodiscard]] inline std::vector<std::uint64_t> SumsOfPowersRecursive(
    std::uint64_t n, unsigned i) {
  std::vector<std::uint64_t> sums(i + 1);
  sums[0] = n;
  for (unsigned p = 1; p <= i; ++p) {
    std::uint64_t rhs = Ipow(n + 1, p + 1) - 1;
    for (unsigned j = 0; j < p; ++j) rhs -= Binomial(p + 1, j) * sums[j];
    sums[p] = rhs / (p + 1);
  }
  return sums;
}

// Coefficients c_0..c_{i+1} of S_i(n) = sum_m c_m n^m, as the interpolating
// polynomial (10.2) through the points (m, S_i(m)), m = 0..i+1.
// E.g. i = 2 gives {0, 1/6, 1/2, 1/3}. Double precision: fine for i <~ 10.
[[nodiscard]] inline Vector SumOfPowersPolynomial(unsigned i) {
  Vector xs(i + 2), ys(i + 2);
  for (unsigned m = 0; m <= i + 1; ++m) {
    xs[m] = m;
    ys[m] = static_cast<double>(SumOfPowers(m, i));
  }
  return InterpolatingPolynomial(xs, ys);
}

// ------------------------------------------------------------- 10.3.2 ------
// sum_{k=1}^{n} k^j x^k by direct summation.
[[nodiscard]] inline double SumKjXk(unsigned n, unsigned j, double x) noexcept {
  double s = 0.0, xk = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    xk *= x;
    s += std::pow(static_cast<double>(k), j) * xk;
  }
  return s;
}

// Closed forms for j = 0, 1, 2 (x != 1; at x = 1 they reduce to S_j(n)).
[[nodiscard]] inline double GeometricSum(unsigned n, double x) noexcept {
  if (x == 1.0) return n;
  return x * (1.0 - std::pow(x, n)) / (1.0 - x);
}
[[nodiscard]] inline double SumKXk(unsigned n, double x) noexcept {
  if (x == 1.0) return static_cast<double>(SumK(n));
  const double nd = n, xn = std::pow(x, n);
  return x * (1.0 - (nd + 1.0) * xn + nd * xn * x) / ((1.0 - x) * (1.0 - x));
}
[[nodiscard]] inline double SumK2Xk(unsigned n, double x) noexcept {
  if (x == 1.0) return static_cast<double>(SumK2(n));
  const double nd = n, xn = std::pow(x, n), d = 1.0 - x;
  return x *
         (1.0 + x - (nd + 1.0) * (nd + 1.0) * xn +
          (2.0 * nd * nd + 2.0 * nd - 1.0) * xn * x - nd * nd * xn * x * x) /
         (d * d * d);
}

// Coefficients of P_j, the numerator of sum_{k>=1} k^j x^k = P_j(x) /
// (1-x)^(j+1). P_0 = x, P_1 = x, P_2 = x + x^2, P_3 = x + 4x^2 + x^3 (Eulerian
// numbers).
[[nodiscard]] inline Vector KjXkNumerator(unsigned j) {
  Vector poly{0.0, 1.0};
  for (unsigned m = 0; m < j; ++m) {
    // Q = x (P'(1-x) + (m+1) P), degree grows by one.
    Vector next_poly(poly.size() + 1, 0.0);
    for (std::size_t k = 0; k < poly.size(); ++k) {
      const double kd = static_cast<double>(k);
      // x * (m+1) P: coefficient k -> k+1.
      next_poly[k + 1] += (m + 1.0) * poly[k];
      if (k == 0) continue;
      // x * P' = sum k p_k x^k ; -x^2 P' = -sum k p_k x^(k+1).
      next_poly[k] += kd * poly[k];
      next_poly[k + 1] -= kd * poly[k];
    }
    while (next_poly.size() > 1 && next_poly.back() == 0.0)
      next_poly.pop_back();
    poly = std::move(next_poly);
  }
  return poly;
}

// sum_{k>=1} k^j x^k for |x| < 1.
[[nodiscard]] inline double SumKjXkInfinite(unsigned j, double x) {
  if (!(std::abs(x) < 1.0))
    throw std::domain_error("sum_kj_xk_infinite: need |x| < 1");
  return PolynomialValue(KjXkNumerator(j), x) / std::pow(1.0 - x, j + 1);
}

}  // namespace p4mfe::ch10
