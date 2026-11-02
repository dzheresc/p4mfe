#pragma once
// Exercises 6-7 (section 8.12): e^pi vs. pi^e, and the convexity of e^x.

#include <cmath>
#include <numbers>

namespace p4mfe::ch08::ex {

// ------------------------------------------------------------ Exercise 6 ---
// Which number is larger, e^pi or pi^e?
//
// Both are positive, so compare logarithms: pi vs. e ln(pi), i.e.
//   ln(e)/e = 1/e   vs.   ln(pi)/pi.
// Let f(x) = ln(x)/x on (0, inf). f'(x) = (1 - ln x)/x^2 is > 0 for x < e and
// < 0 for x > e, so e is the strict global maximum of f. Since pi != e,
//   ln(pi)/pi < 1/e   =>   e ln(pi) < pi   =>   pi^e < e^pi.
// Numerically e^pi = 23.140693 and pi^e = 22.459158.
// (More generally, for e <= a < b: a^b > b^a, since f is decreasing on [e,
// inf).)
[[nodiscard]] inline double ex06_f(double x) { return std::log(x) / x; }
[[nodiscard]] inline double ex06_e_pow_pi() {
  return std::exp(std::numbers::pi);
}
[[nodiscard]] inline double ex06_pi_pow_e() {
  return std::pow(std::numbers::pi, std::numbers::e);
}

// ------------------------------------------------------------ Exercise 7 ---
// Show that (e^x + e^y)/2 >= e^((x+y)/2) for all real x, y.
//
// Proof 1 (AM-GM). With A = e^(x/2), B = e^(y/2):
//   (e^x + e^y)/2 - e^((x+y)/2) = (A^2 + B^2)/2 - AB = (A - B)^2/2 >= 0,
// with equality iff A = B iff x = y.
// Proof 2 (convexity). e^x is convex ((e^x)'' = e^x > 0), so its value at the
// midpoint is at most the average of its values at x and y.
// The gap is (e^(x/2) - e^(y/2))^2 / 2; ex07_gap computes it in this form so it
// is accurate even when x and y are close.
[[nodiscard]] inline double ex07_lhs(double x, double y) {
  return 0.5 * (std::exp(x) + std::exp(y));
}
[[nodiscard]] inline double ex07_rhs(double x, double y) {
  return std::exp(0.5 * (x + y));
}
[[nodiscard]] inline double ex07_gap(double x, double y) {
  const double d = std::exp(0.5 * x) - std::exp(0.5 * y);
  return 0.5 * d * d;
}

}  // namespace p4mfe::ch08::ex
