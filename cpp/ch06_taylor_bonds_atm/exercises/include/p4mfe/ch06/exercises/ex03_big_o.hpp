#pragma once
// Exercise 3 (section 6.11): e^(-x) - 1/(1 + x) = O(x^2) as x -> 0.

#include <cmath>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 3 ---
// Show that e^(-x) - 1/(1 + x) = O(x^2) as x -> 0.
//
// For |x| < 1 both functions are their Taylor series:
//   e^(-x)      = sum_k (-1)^k x^k / k!  = 1 - x + x^2/2 - x^3/6 + ...
//   1/(1 + x)   = sum_k (-1)^k x^k       = 1 - x + x^2   - x^3   + ...
// so
//   e^(-x) - 1/(1 + x) = sum_{k>=2} (-1)^k (1/k! - 1) x^k = -x^2/2 + (5/6) x^3
//   - ...
// The constant and linear terms cancel, hence for |x| <= 1/2, say,
//   |e^(-x) - 1/(1 + x)| <= C x^2
// with C = sum_{k>=2} (1 - 1/k!) (1/2)^(k-2) < sum_{k>=2} (1/2)^(k-2) = 2,
// which is the definition of O(x^2). The quotient tends to -1/2.
//
// Evaluated as expm1(-x) + x/(1 + x), which avoids the cancellation of the
// two leading 1's for small x.
[[nodiscard]] inline double ex03_difference(double x) {
  return std::expm1(-x) + x / (1.0 + x);
}
[[nodiscard]] constexpr double ex03_leading(double x) noexcept {
  return -0.5 * x * x;
}
[[nodiscard]] constexpr double ex03_two_terms(double x) noexcept {
  return x * x * (-0.5 + 5.0 / 6.0 * x);
}
inline constexpr double ex03_big_o_constant = 2.0;  // valid for |x| <= 1/2

}  // namespace p4mfe::ch06::ex
