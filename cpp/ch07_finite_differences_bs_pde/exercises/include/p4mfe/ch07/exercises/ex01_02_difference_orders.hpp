#pragma once
// Exercises 1-2 (section 7.6): a central and a forward difference whose order
// is one higher than usual, because a Taylor coefficient happens to vanish.

#include <p4mfe/ch07/finite_differences.hpp>
#include <p4mfe/normal.hpp>

#include <cmath>
#include <numbers>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Let f(x) = x^3 e^x - 6 e^x. Show that the central finite difference
// approximation for f'(0) is a fourth order approximation.
//
// Taylor's formula around 0 (the even terms cancel):
//   (f(h) - f(-h)) / (2h) = f'(0) + f'''(0) h^2/6 + f^(5)(0) h^4/120 + O(h^6).
// Write f = p e^x with p(x) = x^3 - 6. By Leibniz's rule
//   f^(k)(x) = e^x sum_j C(k, j) p^(j)(x),
// and at 0: p = -6, p' = p'' = 0, p''' = 6, p^(j) = 0 for j >= 4. Hence
//   f'(0)     = p + p'                      = -6,
//   f'''(0)   = p + 3p' + 3p'' + p'''       = -6 + 6 = 0,
//   f^(5)(0)  = p + 5p' + 10p'' + 10p'''    = -6 + 60 = 54.
// The h^2 term vanishes, so
//   (f(h) - f(-h)) / (2h) - f'(0) = 54 h^4 / 120 + O(h^6) = 0.45 h^4 + O(h^6):
// a fourth order approximation (for a generic f it is only second order).
[[nodiscard]] inline double ex01_f(double x) { return (x * x * x - 6.0) * std::exp(x); }
inline constexpr double ex01_exact_derivative = -6.0;   // f'(0)
inline constexpr double ex01_leading_coefficient = 0.45;  // f^(5)(0)/120

[[nodiscard]] inline double ex01_central(double h) { return central_difference(ex01_f, 0.0, h); }
[[nodiscard]] inline double ex01_error(double h) { return ex01_central(h) - ex01_exact_derivative; }

// ------------------------------------------------------------ Exercise 2 ---
// Let N(x) be the cumulative density of the standard normal variable. Show
// that the forward finite difference approximation of N'(0) is a second order
// approximation.
//
// Taylor's formula:
//   (N(h) - N(0)) / h = N'(0) + N''(0) h/2 + N'''(0) h^2/6 + O(h^3).
// N'(x) = phi(x) = e^(-x^2/2)/sqrt(2 pi), N''(x) = -x phi(x), and
// N'''(x) = (x^2 - 1) phi(x). So N''(0) = 0 and N'''(0) = -phi(0):
//   (N(h) - N(0)) / h - N'(0) = -phi(0) h^2 / 6 + O(h^3)
//                             = -h^2 / (6 sqrt(2 pi)) + O(h^3),
// second order instead of the usual first order. (The O(h^3) term is in fact
// zero too: N^(4)(0) = 0 since N - 1/2 is odd, so the next term is O(h^4).)
inline constexpr double ex02_exact_derivative = 0.3989422804014327;  // phi(0) = 1/sqrt(2 pi)
inline const double ex02_leading_coefficient = -ex02_exact_derivative / 6.0;

[[nodiscard]] inline double ex02_forward(double h) {
    return forward_difference([](double x) { return norm_cdf(x); }, 0.0, h);
}
[[nodiscard]] inline double ex02_error(double h) { return ex02_forward(h) - ex02_exact_derivative; }

// Observed order log2(|e(h)| / |e(h/2)|) of an error function e(h).
template <std::invocable<double> E>
[[nodiscard]] double ex_observed_order(E&& error, double h) {
    return std::log2(std::abs(error(h)) / std::abs(error(0.5 * h)));
}

}  // namespace p4mfe::ch07::ex
