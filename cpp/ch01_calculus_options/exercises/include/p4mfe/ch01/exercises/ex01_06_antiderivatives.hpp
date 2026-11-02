#pragma once
// Exercises 1-6 (section 1.12): antiderivatives by parts and by substitution.
//
// Each function returns one antiderivative F (constant C = 0). The tests
// check F' = f by central differences and F(b) - F(a) against a midpoint
// Riemann sum of f.

#include <cmath>
#include <stdexcept>

namespace p4mfe::ch01::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Use integration by parts to compute  integral ln(x) dx.
//
// u = ln x, dv = dx:  integral ln x dx = x ln x - integral x (1/x) dx
//                                      = x ln x - x + C,     x > 0.
[[nodiscard]] inline double Ex01LnAntiderivative(double x) {
  return x * std::log(x) - x;
}

// ------------------------------------------------------------ Exercise 2 ---
// Compute  integral 1 / (x ln(x)) dx.   Hint: u = ln(x).
//
// du = dx / x, so the integral is  integral du / u = ln|u| = ln|ln x| + C,
// on (0, 1) and on (1, infinity) separately.
[[nodiscard]] inline double Ex02InvXLnXAntiderivative(double x) {
  return std::log(std::abs(std::log(x)));
}

// ------------------------------------------------------------ Exercise 3 ---
// Show that (tan x)' = 1/(cos x)^2 and conclude that
//   integral 1/(1 + x^2) dx = arctan(x) + C.
//
// Quotient rule: (sin/cos)' = (cos^2 + sin^2)/cos^2 = 1/cos^2 = 1 + tan^2.
// Let y = arctan x, so tan y = x. Differentiating, y' (1 + tan^2 y) = 1, hence
// (arctan x)' = 1/(1 + tan^2 y) = 1/(1 + x^2), which is the claim.
[[nodiscard]] inline double Ex03TanDerivative(double x) {
  const double c = std::cos(x);
  return 1.0 / (c * c);
}
[[nodiscard]] inline double Ex03ArctanDerivative(double x) {
  return 1.0 / (1.0 + x * x);
}
[[nodiscard]] inline double Ex03Inv1PlusX2Antiderivative(double x) {
  return std::atan(x);
}

// ------------------------------------------------------------ Exercise 4 ---
// Compute  integral x^n ln(x) dx.
//
// n != -1: u = ln x, dv = x^n dx, v = x^(n+1)/(n+1):
//   integral x^n ln x dx = x^(n+1) ln(x) / (n+1) - x^(n+1) / (n+1)^2 + C.
// n == -1: substitution u = ln x gives (ln x)^2 / 2 + C.
// Valid for any real n, x > 0.
[[nodiscard]] inline double Ex04XPowNLnAntiderivative(double n, double x) {
  const double lx = std::log(x);
  if (n == -1.0) return 0.5 * lx * lx;
  const double m = n + 1.0;
  const double xm = std::pow(x, m);
  return xm * lx / m - xm / (m * m);
}

// ------------------------------------------------------------ Exercise 5 ---
// Compute  integral x^n e^x dx   (n a nonnegative integer).
//
// Parts with u = x^n, dv = e^x dx gives the recursion
//   I_n = x^n e^x - n I_(n-1),   I_0 = e^x,
// which unrolls to
//   I_n = e^x sum_{k=0}^{n} (-1)^(n-k) n!/k! x^k + C.
// E.g. I_1 = (x - 1) e^x, I_2 = (x^2 - 2x + 2) e^x.
[[nodiscard]] inline double Ex05XPowNExpAntiderivative(unsigned n, double x) {
  // Horner-like evaluation of the recursion: I_n / e^x = x^n - n (I_(n-1) /
  // e^x).
  double p = 1.0;  // I_0 / e^x
  double xk = 1.0;
  for (unsigned k = 1; k <= n; ++k) {
    xk *= x;
    p = xk - static_cast<double>(k) * p;
  }
  return p * std::exp(x);
}

// ------------------------------------------------------------ Exercise 6 ---
// Compute  integral (ln(x))^n dx   (n a nonnegative integer).
//
// Parts with u = (ln x)^n, dv = dx gives the recursion
//   J_n = x (ln x)^n - n J_(n-1),   J_0 = x,
// i.e.  J_n = x sum_{k=0}^{n} (-1)^(n-k) n!/k! (ln x)^k + C.
// (Substituting x = e^t turns J_n into exercise 5: integral t^n e^t dt.)
[[nodiscard]] inline double Ex06LnPowNAntiderivative(unsigned n, double x) {
  if (x <= 0.0) throw std::domain_error("ex06: x must be positive");
  return Ex05XPowNExpAntiderivative(n, std::log(x));
}

}  // namespace p4mfe::ch01::ex
