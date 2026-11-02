#pragma once
// Exercise 9 (section 6.11): the Taylor series of ln(1 + x) converges to
// ln(1 + x) for every |x| < 1.

#include <algorithm>
#include <cmath>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch06/taylor_series.hpp>
#include <stdexcept>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 9 ---
// Let T(x) = sum_{k>=1} (-1)^(k+1) x^k / k be the Taylor series of
// f(x) = ln(1 + x), and P_n(x) its Taylor polynomial of degree n. Section
// 6.3.1 showed T(x) = f(x) for |x| <= 1/2. Show it for all |x| < 1, i.e.
//   lim_{n->inf} |f(x) - P_n(x)| = 0  for all |x| < 1.       (6.128)
//
// (i) For any x > -1,
//   f(x) - P_n(x) = int_0^x (-1)^(n+2) (x - t)^n / (1 + t)^(n+1) dt.
// The integral form (6.4) of the remainder is
//   f(x) - P_n(x) = int_0^x (x - t)^n / n! f^(n+1)(t) dt,
// and (6.51) gives f^(m)(t) = (-1)^(m+1) (m-1)! / (1 + t)^m, so
// f^(n+1)(t) / n! = (-1)^(n+2) / (1 + t)^(n+1). Substitute.
//
// (ii) For 0 <= x < 1 and 0 <= t <= x, the integrand has absolute value
//   ((x - t)/(1 + t))^n / (1 + t),
// and (x - t)/(1 + t) <= x (equivalent to x - t <= x + xt, i.e. t(1 + x) >= 0).
// So
//   |f(x) - P_n(x)| <= int_0^x ((x - t)/(1 + t))^n dt/(1 + t)
//                   <= x^n int_0^x dt/(1 + t) = x^n ln(1 + x),      (6.129)
// which tends to 0 because 0 <= x < 1. This proves (6.128) on [0, 1).
//
// (iii) For -1 < x <= 0 put s = -x in [0, 1) and t = -z (dt = -dz):
//   f(x) - P_n(x) = int_0^{-s} (-1)^n (-s - t)^n / (1 + t)^(n+1) dt
//                 = -int_0^s (-1)^n (-1)^n (s - z)^n / (1 - z)^(n+1) dz,
// so
//   |f(x) - P_n(x)| = int_0^s (s - z)^n / (1 - z)^(n+1) dz.
// For 0 <= z <= s < 1, (s - z)/(1 - z) <= s (equivalent to s - z <= s - sz,
// i.e. z(1 - s) >= 0), hence
//   |f(x) - P_n(x)| <= s^n int_0^s dz/(1 - z) = s^n |ln(1 - s)| = (-x)^n |ln(1
//   + x)| -> 0.
// Together with (ii), (6.128) holds for every |x| < 1, so ln(1 + x) = T(x)
// there. (The series also converges at x = 1, to ln 2, by the alternating
// series test; it diverges at x = -1.)

// f(x) - P_n(x), reusing chapter 6's partial sums of ln(1 + x).
[[nodiscard]] inline double Ex09Error(double x, unsigned n) {
  return std::log1p(x) - Log1pSeries(x, n);
}

// (i): the integral form, computed with chapter 2's Simpson's rule.
[[nodiscard]] inline double Ex09ErrorIntegral(double x, unsigned n,
                                              double tol = 1e-13) {
  if (!(x > -1.0)) throw std::domain_error("ex09_error_integral: need x > -1");
  if (x == 0.0) return 0.0;
  const double sign = (n % 2 == 0) ? 1.0 : -1.0;  // (-1)^(n+2) = (-1)^n
  const auto g = [&](double t) {
    return sign * std::pow(x - t, n) / std::pow(1.0 + t, n + 1);
  };
  const double v =
      ch02::IntegrateToTolerance(g, std::min(0.0, x), std::max(0.0, x), tol)
          .value;
  return x > 0.0 ? v : -v;
}

// (ii): the middle term of (6.129), 0 <= x < 1.
[[nodiscard]] inline double Ex09MiddleBound(double x, unsigned n,
                                            double tol = 1e-13) {
  if (!(x >= 0.0 && x < 1.0))
    throw std::domain_error("ex09_middle_bound: need 0 <= x < 1");
  if (x == 0.0) return 0.0;
  const auto g = [&](double t) {
    return std::pow((x - t) / (1.0 + t), n) / (1.0 + t);
  };
  return ch02::IntegrateToTolerance(g, 0.0, x, tol).value;
}

// (ii)-(iii): the final bound, |x|^n |ln(1 + x)|, valid for every |x| < 1.
[[nodiscard]] inline double Ex09Bound(double x, unsigned n) {
  if (!(std::abs(x) < 1.0)) throw std::domain_error("ex09_bound: need |x| < 1");
  return std::pow(std::abs(x), n) * std::abs(std::log1p(x));
}

// (iii): int_0^s (s - z)^n / (1 - z)^(n+1) dz, s = -x in [0, 1).
[[nodiscard]] inline double Ex09ErrorNegative(double x, unsigned n,
                                              double tol = 1e-13) {
  if (!(x > -1.0 && x <= 0.0))
    throw std::domain_error("ex09_error_negative: need -1 < x <= 0");
  const double s = -x;
  if (s == 0.0) return 0.0;
  const auto g = [&](double z) {
    return std::pow(s - z, n) / std::pow(1.0 - z, n + 1);
  };
  return ch02::IntegrateToTolerance(g, 0.0, s, tol).value;
}

}  // namespace p4mfe::ch06::ex
