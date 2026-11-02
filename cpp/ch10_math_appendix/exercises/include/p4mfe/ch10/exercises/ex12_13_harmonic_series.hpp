#pragma once
// Exercises 12-13 (section 10.10): sum 1/k^2 converges, sum 1/k and
// sum 1/(k ln k) diverge; H_n - ln n converges to Euler's constant.

#include <cmath>
#include <numbers>
#include <utility>

namespace p4mfe::ch10::ex {

// ----------------------------------------------------------- Exercise 12 ---
// Show that sum_{k>=1} 1/k^2 converges, while sum_{k>=1} 1/k and
// sum_{k>=2} 1/(k ln k) diverge.
//
// sum 1/k^2. The partial sums increase. For k >= 2, 1/k^2 < 1/(k(k-1)) =
// 1/(k-1) - 1/k, which telescopes:
//   sum_{k=1}^{n} 1/k^2 < 1 + sum_{k=2}^{n} (1/(k-1) - 1/k) = 2 - 1/n < 2.
// An increasing sequence bounded above converges. (Its limit is pi^2/6; the
// same comparison on the tail gives 1/(n+1) < pi^2/6 - sum_{k<=n} 1/k^2 < 1/n.)
//
// sum 1/k (hint). For k < x < k + 1, 1/(k+1) < 1/x < 1/k; integrating over
// [k, k+1] and summing k = 1..n-1,
//   sum_{k=1}^{n-1} 1/(k+1) < ln n = int_1^n dx/x < sum_{k=1}^{n-1} 1/k.
// With H_n = sum_{k=1}^{n} 1/k the left inequality is H_n - 1 < ln n and the
// right one is ln n < H_n - 1/n, so
//   ln n + 1/n < H_n < ln n + 1   for n >= 2
// (for n = 1 all three are equal to 1, see the notes). Since ln n -> oo,
// H_n -> oo: the harmonic series diverges.
//
// sum 1/(k ln k). The function 1/(x ln x) is decreasing on [2, oo) and has
// the antiderivative ln(ln x). For k < x < k + 1,
// 1/((k+1) ln(k+1)) < 1/(x ln x) < 1/(k ln k), so with
// I_n = int_2^n dx/(x ln x) = ln(ln n) - ln(ln 2),
//   sum_{k=3}^{n} 1/(k ln k) < I_n < sum_{k=2}^{n-1} 1/(k ln k).
// Hence, for n >= 3,
//   I_n + 1/(n ln n) < sum_{k=2}^{n} 1/(k ln k) < I_n + 1/(2 ln 2).
// The lower bound tends to oo (very slowly: ln ln 10^6 = 2.63), so the series
// diverges.

// sum_{k=1}^{n} 1/k^2, summed from the smallest term up for accuracy.
[[nodiscard]] inline double Ex12SumInvSquares(unsigned n) {
  double s = 0.0;
  for (unsigned k = n; k >= 1; --k) s += 1.0 / (static_cast<double>(k) * k);
  return s;
}
// H_n = sum_{k=1}^{n} 1/k.
[[nodiscard]] inline double Ex12Harmonic(unsigned n) {
  double s = 0.0;
  for (unsigned k = n; k >= 1; --k) s += 1.0 / k;
  return s;
}
// sum_{k=2}^{n} 1/(k ln k).
[[nodiscard]] inline double Ex12SumInvKLogK(unsigned n) {
  double s = 0.0;
  for (unsigned k = n; k >= 2; --k)
    s += 1.0 / (k * std::log(static_cast<double>(k)));
  return s;
}
// The bounds (lower, upper) derived above.
[[nodiscard]] inline std::pair<double, double> Ex12HarmonicBounds(unsigned n) {
  const double l = std::log(static_cast<double>(n));
  return {l + 1.0 / n, l + 1.0};
}
[[nodiscard]] inline std::pair<double, double> Ex12KLogKBounds(unsigned n) {
  const double nd = n,
               integral = std::log(std::log(nd)) - std::log(std::numbers::ln2);
  return {integral + 1.0 / (nd * std::log(nd)),
          integral + 1.0 / (2.0 * std::numbers::ln2)};
}

// ----------------------------------------------------------- Exercise 13 ---
// Show that x_n = H_n - ln n converges to a limit between 0 and 1.
//
// Decreasing: x_{n+1} - x_n = 1/(n+1) - ln((n+1)/n) = 1/(n+1) - int_n^{n+1}
// dx/x < 0, since 1/x > 1/(n+1) on (n, n+1). Bounded below: by exercise 12, x_n
// > 1/n > 0. A decreasing sequence bounded below converges; call the limit
// gamma. Then gamma <= x_n for every n. Upper bound: gamma <= x_2 = 3/2 - ln 2
// = 0.8069 < 1. Lower bound: y_n = H_n - ln(n+1) is increasing,
//   y_{n+1} - y_n = 1/(n+1) - int_{n+1}^{n+2} dx/x > 0,
// and x_n - y_n = ln(1 + 1/n) -> 0, so gamma = lim y_n >= y_1 = 1 - ln 2 =
// 0.3069 > 0. So 0.3069 <= gamma <= 0.8069; in fact gamma = 0.5772156649...
// Numerically, x_n - gamma = 1/(2n) - 1/(12 n^2) + O(1/n^4).

inline constexpr double kEx13EulerGamma = std::numbers::egamma;

[[nodiscard]] inline double Ex13X(unsigned n) {
  return Ex12Harmonic(n) - std::log(static_cast<double>(n));
}
[[nodiscard]] inline double Ex13Y(unsigned n) {
  return Ex12Harmonic(n) - std::log(static_cast<double>(n) + 1.0);
}

}  // namespace p4mfe::ch10::ex
