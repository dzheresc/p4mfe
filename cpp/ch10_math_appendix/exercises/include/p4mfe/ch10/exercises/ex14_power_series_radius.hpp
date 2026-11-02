#pragma once
// Exercise 14 (section 10.10): radius of convergence of sum x^k / (k ln k)
// and behavior at x = +-1.

#include <cmath>
#include <p4mfe/ch10/power_series.hpp>

namespace p4mfe::ch10::ex {

// ----------------------------------------------------------- Exercise 14 ---
// Find the radius of convergence R of sum_{k>=2} x^k / (k ln k) and
// investigate the points |x| = R.
//
// a_k = 1/(k ln k). Ratio test:
//   |a_k / a_{k+1}| = (k+1) ln(k+1) / (k ln k) = (1 + 1/k) (1 + ln(1 + 1/k)/ln
//   k) -> 1,
// so R = 1. (Root test agrees: |a_k|^(1/k) = exp(-(ln k + ln ln k)/k) -> 1.)
//
// x = 1: sum 1/(k ln k) diverges (exercise 12; its partial sums grow like
// ln ln n).
// x = -1: sum (-1)^k / (k ln k) is alternating with |terms| = 1/(k ln k)
// decreasing to 0, so it converges by the alternating series (Leibniz) test.
// The convergence is conditional, since the series of absolute values is the
// divergent one at x = 1. The partial sums S_n bracket the limit, and
// |S - S_n| <= 1/((n+1) ln(n+1)).
// So the series converges exactly on [-1, 1).

// Coefficients a_0..a_n of the series (a_0 = a_1 = 0).
[[nodiscard]] inline Vector Ex14Coefficients(unsigned n) {
  Vector a(n + 1, 0.0);
  for (unsigned k = 2; k <= n; ++k)
    a[k] = 1.0 / (k * std::log(static_cast<double>(k)));
  return a;
}

// sum_{k=2}^{n} x^k / (k ln k).
[[nodiscard]] inline double Ex14PartialSum(double x, unsigned n) {
  double s = 0.0, xk = x;
  for (unsigned k = 2; k <= n; ++k) {
    xk *= x;
    s += xk / (k * std::log(static_cast<double>(k)));
  }
  return s;
}

// Ratio |a_k / a_{k+1}| for one k, the ratio-test estimate of R.
[[nodiscard]] inline double Ex14Ratio(unsigned k) {
  const double kd = k;
  return (kd + 1.0) * std::log(kd + 1.0) / (kd * std::log(kd));
}

}  // namespace p4mfe::ch10::ex
