#pragma once
// Exercise 11 (section 1.12): differentiating an integral with a variable
// upper limit.
//
// Let K, T, sigma and r be positive constants, and define g : R -> R by
//   g(x) = 1/sqrt(2 pi) integral_0^(b(x)) e^(-y^2/2) dy,
// where b(x) = (ln(x/K) + (r + sigma^2/2) T) / (sigma sqrt(T)).
// Compute g'(x).
// (Note: g is related to the Delta of a plain vanilla call; see section 3.5.)
//
// Solution. By the fundamental theorem of calculus and the chain rule
// (section 1.3 with a constant lower limit and an integrand free of x),
//   g'(x) = 1/sqrt(2 pi) e^(-b(x)^2/2) b'(x),    b'(x) = 1 / (x sigma sqrt(T)),
// so
//   g'(x) = e^(-b(x)^2/2) / (x sigma sqrt(2 pi T)).
// Indeed g(x) = N(b(x)) - 1/2 = N(d1) - 1/2 = Delta_call - 1/2 for x = S, q =
// 0, and g'(x) is the call's Gamma.

#include <cmath>
#include <numbers>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch01::ex {

struct Ex11Params {
  double K;
  double T;
  double sigma;
  double r;
};

[[nodiscard]] inline double ex11_b(const Ex11Params& p, double x) {
  return (std::log(x / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T) /
         (p.sigma * std::sqrt(p.T));
}

// g(x) in closed form, N(b(x)) - 1/2.
[[nodiscard]] inline double ex11_g(const Ex11Params& p, double x) {
  return norm_cdf(ex11_b(p, x)) - 0.5;
}

// g'(x), the answer.
[[nodiscard]] inline double ex11_g_prime(const Ex11Params& p, double x) {
  const double b = ex11_b(p, x);
  return std::exp(-0.5 * b * b) /
         (x * p.sigma * std::sqrt(2.0 * std::numbers::pi * p.T));
}

}  // namespace p4mfe::ch01::ex
