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
  double strike;
  double maturity;
  double sigma;
  double r;
};

[[nodiscard]] inline double Ex11B(const Ex11Params& p, double x) {
  return (std::log(x / p.strike) +
          (p.r + 0.5 * p.sigma * p.sigma) * p.maturity) /
         (p.sigma * std::sqrt(p.maturity));
}

// g(x) in closed form, N(b(x)) - 1/2.
[[nodiscard]] inline double Ex11G(const Ex11Params& p, double x) {
  return NormCdf(Ex11B(p, x)) - 0.5;
}

// g'(x), the answer.
[[nodiscard]] inline double Ex11GPrime(const Ex11Params& p, double x) {
  const double b = Ex11B(p, x);
  return std::exp(-0.5 * b * b) /
         (x * p.sigma * std::sqrt(2.0 * std::numbers::pi * p.maturity));
}

}  // namespace p4mfe::ch01::ex
