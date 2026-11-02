#pragma once
// Chapter 10, section 10.8: convergence results for Taylor expansions.
//
// The Taylor series of f at a converges to f(x) exactly when the remainder
// R_n(x) -> 0 (chapter 6). A sufficient condition: if on the interval
// between a and x the derivatives grow at most geometrically,
//   |f^(k)(t)| <= C M^k   for all k,
// then by the Lagrange form of the remainder
//   |R_n(x)| <= C (M |x - a|)^(n+1) / (n+1)!  -> 0,
// since factorials beat powers. This covers e^x (C = e^max(a,x), M = 1),
// sin, cos (C = M = 1) and e^(Mx).
//
// Smoothness alone is not enough: f(x) = e^(-1/x^2), f(0) = 0, has
// f^(k)(0) = 0 for every k, so its Taylor series at 0 is identically zero and
// equals f only at x = 0.
//
// Conversely, a power series is the Taylor series of its sum inside its
// interval of convergence: a_k = f^(k)(c) / k!.

#include <cmath>
#include <p4mfe/ch06/taylor.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch10 {

// C (M |x - a|)^(n+1) / (n+1)!, via chapter 6's Lagrange remainder bound.
[[nodiscard]] inline double TaylorBoundGeometric(double c, double m, double a,
                                                 double x, unsigned n) {
  return ch06::TaylorRemainderBound(c * std::pow(m, n + 1), a, x, n);
}

// Smallest n with C (M |x-a|)^(n+1) / (n+1)! < tol.
[[nodiscard]] inline unsigned TaylorTermsNeeded(double c, double m, double a,
                                                double x, double tol,
                                                unsigned max_n = 1000) {
  if (!(tol > 0.0))
    throw std::invalid_argument("taylor_terms_needed: tol must be positive");
  const double q = m * std::abs(x - a);
  double bound = c * q;  // n = 0
  for (unsigned n = 0; n <= max_n; ++n) {
    if (bound < tol) return n;
    bound *= q / (n + 2.0);
  }
  throw std::runtime_error("taylor_terms_needed: tolerance not reached");
}

// The smooth, non-analytic function e^(-1/x^2) (0 at x = 0).
[[nodiscard]] inline double FlatFunction(double x) noexcept {
  return x == 0.0 ? 0.0 : std::exp(-1.0 / (x * x));
}

// Taylor series at a from the derivatives f(a), f'(a), ..., f^(n)(a): a_k =
// f^(k)(a) / k!.
[[nodiscard]] inline PowerSeries TaylorSeries(std::span<const double> derivs,
                                              double a) {
  PowerSeries s{Vector(derivs.begin(), derivs.end()), a};
  double fact = 1.0;
  for (std::size_t k = 1; k < s.a.size(); ++k) {
    fact *= static_cast<double>(k);
    s.a[k] /= fact;
  }
  return s;
}

// f(c), f'(c), ..., f^(n)(c) of a power series: k! a_k.
[[nodiscard]] inline Vector DerivativesAtCenter(const PowerSeries& s) {
  Vector d(s.a);
  double fact = 1.0;
  for (std::size_t k = 1; k < d.size(); ++k) {
    fact *= static_cast<double>(k);
    d[k] *= fact;
  }
  return d;
}

}  // namespace p4mfe::ch10
