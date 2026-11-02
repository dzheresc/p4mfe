#pragma once
// Exercise 10 (section 6.11): int_0^1 ln(1 - x) ln(x) dx = 2 - pi^2/6.

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 10 ---
// Compute I = int_0^1 ln(1 - x) ln(x) dx.                    (6.130)
//
// (i) lim_{x->0+} ln(1 - x) ln(x) = lim_{x->1-} ln(1 - x) ln(x) = 0, so I is a
// definite integral (of a function continuous on [0, 1] once extended by 0).
// Near 0, ln(1 - x) = -x + O(x^2), so ln(1 - x) ln x = -x ln x + O(x^2 ln x)
// -> 0 (x ln x -> 0 by l'Hopital: ln x / (1/x) has derivative ratio -x -> 0).
// The integrand is symmetric under x -> 1 - x, so the limit at 1 is the same.
//
// (ii) For 0 < x < 1, ln(1 - x) = -sum_{n>=1} x^n/n, so
//   I = -sum_{n>=1} (1/n) int_0^1 x^n ln(x) dx.
// The terms -x^n ln(x)/n are all >= 0 on (0, 1), so exchanging sum and
// integral is allowed (monotone convergence).
//
// (iii) Integrating by parts, for n >= 0,
//   int_0^1 x^n ln x dx = [x^(n+1) ln x / (n+1)]_0^1 - int_0^1 x^n/(n+1) dx =
//   -1/(n+1)^2,
// so
//   I = sum_{n>=1} 1 / (n (n+1)^2).
// (The book prints the summation index as k with n in the summand; the sum is
// over n.)
//
// (iv) Partial fractions: 1/(n(n+1)^2) = 1/n - 1/(n+1) - 1/(n+1)^2. The first
// two telescope to 1, and sum_{n>=1} 1/(n+1)^2 = pi^2/6 - 1, so
//   I = 1 - (pi^2/6 - 1) = 2 - pi^2/6 = 0.355065933151774.
[[nodiscard]] inline double Ex10Integrand(double x) {
  return std::log1p(-x) * std::log(x);
}
inline constexpr double kEx10Exact =
    2.0 - std::numbers::pi * std::numbers::pi / 6.0;

// The integral by chapter 2's midpoint-rule driver, which never evaluates the
// endpoints where the logs are infinite.
[[nodiscard]] inline double Ex10Numerical(double tol = 1e-10) {
  return ch02::ImproperIntegral(Ex10Integrand, 0.0, 1.0, tol).value;
}
// int_0^1 x^n ln x dx = -1/(n+1)^2, and the same numerically.
[[nodiscard]] constexpr double Ex10Moment(unsigned n) noexcept {
  const double m = static_cast<double>(n) + 1.0;
  return -1.0 / (m * m);
}
// Numerically after x = e^(-s): int_0^1 x^n ln x dx = -int_0^inf s e^(-(n+1) s)
// ds, which removes the ln singularity at 0 (for n = 0 the midpoint rule on [0,
// 1] would converge only like h).
[[nodiscard]] inline double Ex10MomentNumerical(unsigned n,
                                                double tol = 1e-11) {
  const double m = static_cast<double>(n) + 1.0;
  return ch02::ImproperIntegral([m](double s) { return -s * std::exp(-m * s); },
                                0.0, ch02::kInfinity, tol)
      .value;
}
// Partial sum sum_{n=1}^N 1/(n(n+1)^2); the tail is below 1/(2 N^2).
[[nodiscard]] inline double Ex10PartialSum(unsigned num_terms) noexcept {
  double s = 0.0;
  for (unsigned n = num_terms; n >= 1; --n) {  // smallest terms first
    const double m = static_cast<double>(n);
    s += 1.0 / (m * (m + 1.0) * (m + 1.0));
  }
  return s;
}

}  // namespace p4mfe::ch06::ex
