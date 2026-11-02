#pragma once
// Exercises 3-4 (section 2.12): the midpoint, trapezoidal and Simpson rules
// run to a tolerance, and Simpson's rule on an integrand whose fourth
// derivative is unbounded.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <vector>

namespace p4mfe::ch02::ex {

// ------------------------------------------------------------ Exercise 3 ---
// Compute an approximate value of integral_1^3 sqrt(x) e^(-x) dx using the
// Midpoint, Trapezoidal and Simpson's rules. Start with n = 4 intervals and
// double n until two consecutive approximations are within 1e-6.
//
// This is integrate_to_tolerance (section 2.4.1) with tol = 1e-6:
//   Midpoint     0.408369962   n = 256
//   Trapezoidal  0.408370390   n = 512
//   Simpson      0.408370194   n = 16
// Exact value: with x = u^2 the integral is integral_1^sqrt(3) 2 u^2 e^(-u^2)
// du, a smooth integrand, so Simpson gets it to full precision:
//   0.4083702471769958...
// Midpoint and trapezoidal errors are O(h^2), with M_n below and T_n above the
// exact value and |T_n - I| ~ 2 |M_n - I| (the leading error terms are
// -h^2/24 and +h^2/12 times integral f''). Simpson's error is O(h^4), so it
// needs 16 intervals where the others need hundreds.
[[nodiscard]] inline double ex03_integrand(double x) {
  return std::sqrt(x) * std::exp(-x);
}

inline constexpr double ex03_a = 1.0;
inline constexpr double ex03_b = 3.0;
inline constexpr double ex03_tol = 1e-6;

[[nodiscard]] inline IntegrationResult ex03_approx(Rule rule,
                                                   double tol = ex03_tol) {
  return integrate_to_tolerance(ex03_integrand, ex03_a, ex03_b, tol, rule, 4);
}

// Reference value via x = u^2.
[[nodiscard]] inline double ex03_exact() {
  return integrate_to_tolerance(
             [](double u) { return 2.0 * u * u * std::exp(-u * u); }, 1.0,
             std::sqrt(3.0), 1e-15)
      .value;
}

// ------------------------------------------------------------ Exercise 4 ---
// Let f(x) = x^(5/2) / (1 + x^2).
// (i)   Use the Midpoint rule with tol = 1e-6 to approximate
//         I = integral_0^1 x^(5/2) / (1 + x^2) dx.                     (2.89)
// (ii)  Show that f''''(x) is not bounded on (0, 1).
// (iii) Apply Simpson's rule with n = 2^k, k = 2:8, intervals to compute I.
//       Comment on the speed of convergence of Simpson's rule.
//
// Exact value. With x = u^2, dx = 2u du:
//   I = integral_0^1 2 u^6 / (1 + u^4) du = integral_0^1 (2 u^2 - 2 u^2/(1 +
//   u^4)) du
//     = 2/3 - 2 J,   J = integral_0^1 u^2/(1 + u^4) du = (pi - 2 ln(1 + sqrt
//     2)) / (4 sqrt 2),
// (partial fractions over u^4 + 1 = (u^2 + sqrt2 u + 1)(u^2 - sqrt2 u + 1)), so
//   I = 2/3 - (pi - 2 ln(1 + sqrt 2)) / (2 sqrt 2) = 0.1791711722673056...
//
// (i) Midpoint with n = 4, 8, ... stops at n = 512 with 0.179171053.
//
// (ii) Near 0, 1/(1 + x^2) = 1 - x^2 + x^4 - ..., so
//   f(x) = x^(5/2) - x^(9/2) + x^(13/2) - ...,
// and differentiating four times,
//   f''''(x) = (5/2)(3/2)(1/2)(-1/2) x^(-3/2) + (9/2)(7/2)(5/2)(3/2) x^(1/2) -
//   ...
//            = -(15/16) x^(-3/2) + O(x^(1/2)),
// which tends to -infinity as x -> 0+. (Every other term is bounded on (0,1);
// the expansion converges for |x| < 1.) So f'''' is unbounded on (0, 1) and
// the O(h^4) error bound of Simpson's rule, max|f''''| (b-a) h^4 / 2880, does
// not apply.
//
// (iii) Simpson's rule, error |S_n - I|:
//    n      S_n                  error      log2(err_n / err_2n)
//    4   0.1791550997   1.61e-05
//    8   0.1791698156   1.36e-06   3.57
//   16   0.1791710551   1.17e-07   3.53
//   32   0.1791711621   1.02e-08   3.52
//   64   0.1791711714   8.95e-10   3.51
//  128   0.1791711722   7.86e-11   3.51
//  256   0.1791711723   6.91e-12   3.51
// The error still goes to zero quickly, but the observed order is 3.5, not 4:
// each halving of h divides the error by 2^3.5 ~ 11.3 instead of 16. The loss
// comes from the first interval [0, h], where f behaves like x^(5/2); the
// error of any rule exact for cubics on that interval is of size h^(7/2).
// The rest of [0, 1], where f is smooth, still contributes O(h^4).
[[nodiscard]] inline double ex04_f(double x) {
  return std::pow(x, 2.5) / (1.0 + x * x);
}

[[nodiscard]] inline double ex04_exact() {
  const double s2 = std::numbers::sqrt2;
  return 2.0 / 3.0 - (std::numbers::pi - 2.0 * std::log(1.0 + s2)) / (2.0 * s2);
}

// (i)
[[nodiscard]] inline IntegrationResult ex04_midpoint(double tol = 1e-6) {
  return integrate_to_tolerance(ex04_f, 0.0, 1.0, tol, Rule::Midpoint, 4);
}

// (ii) f'''' in closed form. With f = x^(5/2) g(x), g = 1/(1+x^2), Leibniz:
//   f'''' = sum_{j=0}^4 C(4,j) (x^(5/2))^(4-j) g^(j).
// The leading term (j = 0) is -(15/16) x^(-3/2) / (1 + x^2).
[[nodiscard]] inline double ex04_f4(double x) {
  const double q = 1.0 + x * x;
  // derivatives of x^(5/2): orders 0..4
  const double p[5] = {std::pow(x, 2.5), 2.5 * std::pow(x, 1.5),
                       3.75 * std::pow(x, 0.5), 1.875 * std::pow(x, -0.5),
                       -0.9375 * std::pow(x, -1.5)};
  // derivatives of g = 1/(1+x^2): orders 0..4
  const double g[5] = {
      1.0 / q, -2.0 * x / (q * q), (6.0 * x * x - 2.0) / (q * q * q),
      24.0 * x * (1.0 - x * x) / (q * q * q * q),
      24.0 * (5.0 * x * x * x * x - 10.0 * x * x + 1.0) / (q * q * q * q * q)};
  constexpr double binom[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
  double s = 0.0;
  for (int j = 0; j <= 4; ++j) s += binom[j] * p[4 - j] * g[j];
  return s;
}

// (iii)
struct Ex04SimpsonRow {
  std::size_t n;
  double value;
  double error;  // |S_n - I|
};
[[nodiscard]] inline std::vector<Ex04SimpsonRow> ex04_simpson_table(
    int k_min = 2, int k_max = 8) {
  const double exact = ex04_exact();
  std::vector<Ex04SimpsonRow> rows;
  for (int k = k_min; k <= k_max; ++k) {
    const std::size_t n = std::size_t{1} << k;
    const double v = simpson_rule(ex04_f, 0.0, 1.0, n);
    rows.push_back({n, v, std::abs(v - exact)});
  }
  return rows;
}

// The asymptotic order of Simpson's rule for this integrand.
inline constexpr double ex04_simpson_order = 3.5;

}  // namespace p4mfe::ch02::ex
