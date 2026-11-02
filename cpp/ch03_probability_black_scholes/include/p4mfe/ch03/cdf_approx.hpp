#pragma once
// Chapter 3, section 3.9: computing the cumulative normal N(x) accurately
// enough to implement the Black-Scholes formula without a library erf.
//
// Two methods, as in the book:
//  * a polynomial-in-z approximation (Abramowitz & Stegun 26.2.17),
//      N(x) ~ 1 - phi(x) (a1 z + a2 z^2 + a3 z^3 + a4 z^4 + a5 z^5),
//      z = 1 / (1 + 0.2316419 x),  x >= 0,
//    and N(x) = 1 - N(-x) for x < 0; absolute error below 7.5e-8;
//  * Simpson's rule on
//      N(x) = 1/2 + 1/sqrt(2 pi) integral_0^x e^(-t^2/2) dt,
//    with n doubled until the tolerance is met (section 2.4.1).
// The reference p4mfe::norm_cdf uses std::erfc and is exact to rounding.

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch03 {

[[nodiscard]] inline double NormCdfPoly(double x) noexcept {
  if (x < 0.0) return 1.0 - NormCdfPoly(-x);
  constexpr double kA1 = 0.319381530, kA2 = -0.356563782, kA3 = 1.781477937,
                   kA4 = -1.821255978, kA5 = 1.330274429;
  const double z = 1.0 / (1.0 + 0.2316419 * x);
  const double poly =
      z * (kA1 + z * (kA2 + z * (kA3 + z * (kA4 + z * kA5))));  // Horner
  return 1.0 - p4mfe::norm_pdf(x) * poly;
}

[[nodiscard]] inline double NormCdfSimpson(double x, double tol = 1e-12) {
  if (x == 0.0) return 0.5;
  const auto g = [](double t) { return std::exp(-0.5 * t * t); };
  // Integrate over [0, |x|] so the interval is never reversed.
  const double integral =
      ch02::IntegrateToTolerance(g, 0.0, std::abs(x), tol).value;
  const double half_mass =
      integral * std::numbers::inv_sqrtpi / std::numbers::sqrt2;
  return x > 0.0 ? 0.5 + half_mass : 0.5 - half_mass;
}

}  // namespace p4mfe::ch03
