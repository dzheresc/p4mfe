#pragma once
// Exercises 1-2 (section 2.12): the cubic term of N(x) at 0, and the Gamma
// function as an improper integral.

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch02::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Compute
//   lim_{x->0} (N(x) - 1/2 - x/sqrt(2 pi)) / x^3,
// where N(x) = (1/sqrt(2 pi)) integral_{-inf}^x e^(-y^2/2) dy is the
// standard normal cdf.
//
// Solution. N(0) = 1/2, N'(x) = phi(x) = e^(-x^2/2)/sqrt(2 pi), so the
// quotient is 0/0. By l'Hopital, three times:
//   (phi(x) - 1/sqrt(2 pi)) / (3 x^2)       0/0
//   phi'(x) / (6 x) = -x phi(x) / (6 x)  =  -phi(x) / 6  ->  -1 / (6 sqrt(2
//   pi)).
// Equivalently, integrate the series e^(-y^2/2) = sum_k (-1)^k y^(2k) / (2^k
// k!):
//   N(x) - 1/2 = (1/sqrt(2 pi)) sum_{k>=0} (-1)^k x^(2k+1) / (2^k k! (2k+1))
//              = (1/sqrt(2 pi)) (x - x^3/6 + x^5/40 - ...),
// so the limit is -1/(6 sqrt(2 pi)) = -0.0664903801...
inline const double kEx01Limit =
    -1.0 / (6.0 * std::sqrt(2.0 * std::numbers::pi));

// The quotient itself. N(x) - 1/2 - x/sqrt(2 pi) loses every digit to
// cancellation for small x, so for |x| < 1 sum the series above (after
// dividing by x^3); otherwise use norm_cdf directly.
[[nodiscard]] inline double Ex01Quotient(double x) {
  const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * std::numbers::pi);
  if (std::abs(x) < 1.0) {
    // sum_{k>=1} (-1)^k x^(2k-2) / (2^k k! (2k+1))
    double term = -0.5;  // (-1)^k x^(2k-2) / (2^k k!) at k = 1
    double sum = 0.0;
    for (int k = 1; k < 30; ++k) {
      sum += term / (2.0 * k + 1.0);
      term *= -x * x / (2.0 * (k + 1.0));
    }
    return inv_sqrt_2pi * sum;
  }
  return (NormCdf(x) - 0.5 - x * inv_sqrt_2pi) / (x * x * x);
}

// ------------------------------------------------------------ Exercise 2 ---
// Let f : (0, inf) -> R be the Gamma function, f(a) = integral_0^inf x^(a-1)
// e^(-x) dx. (i)  Show f(a) is well defined for any a > 0: both
//        integral_0^1 x^(a-1) e^(-x) dx = lim_{t->0+} integral_t^1 ...
//        integral_1^inf x^(a-1) e^(-x) dx = lim_{t->inf} integral_1^t ...
//      exist and are finite.
// (ii) Prove by parts that f(a) = (a-1) f(a-1) for a > 1. Show f(1) = 1 and
//      conclude f(n) = (n-1)! for every positive integer n.
//
// (i) The integrand is positive, so t -> integral_t^1 (t decreasing) and
// t -> integral_1^t (t increasing) are increasing; each limit exists and is
// finite iff the integrals stay bounded.
//   On (0, 1]:  x^(a-1) e^(-x) <= x^(a-1), and integral_t^1 x^(a-1) dx
//               = (1 - t^a)/a < 1/a, since a > 0.
//   On [1, inf): x^(a-1) e^(-x/2) -> 0 as x -> inf (exponentials beat
//               powers), so it is bounded by some M_a on [1, inf), and
//               x^(a-1) e^(-x) <= M_a e^(-x/2);  integral_1^t M_a e^(-x/2) dx
//               < 2 M_a e^(-1/2).
// (ii) Parts with u = x^(a-1), dv = e^(-x) dx:
//   integral_t^s x^(a-1) e^(-x) dx = [-x^(a-1) e^(-x)]_t^s + (a-1) integral_t^s
//   x^(a-2) e^(-x) dx.
// As t -> 0+ the boundary term t^(a-1) e^(-t) -> 0 because a > 1, and as
// s -> inf s^(a-1) e^(-s) -> 0; the last integral converges by (i) since
// a - 1 > 0. So f(a) = (a-1) f(a-1).
//   f(1) = integral_0^inf e^(-x) dx = lim (1 - e^(-s)) = 1, and by induction
//   f(n) = (n-1) f(n-1) = (n-1)(n-2)...1 * f(1) = (n-1)!.

// Bound from (i) on the [0, 1] piece: integral_0^1 x^(a-1) e^(-x) dx < 1/a.
[[nodiscard]] inline double Ex02LeftBound(double a) {
  if (a <= 0.0) throw std::domain_error("ex02_left_bound: a must be positive");
  return 1.0 / a;
}

// The two pieces of (i), computed numerically. For 0 < a < 1 the integrand
// x^(a-1) e^(-x) is unbounded at 0, so the [0, 1] piece uses x = u^(1/a),
// dx = (1/a) u^(1/a - 1) du, which removes the singularity:
//   integral_0^1 x^(a-1) e^(-x) dx = (1/a) integral_0^1 e^(-u^(1/a)) du.
// For a >= 1 the integrand is bounded and Simpson's rule is used directly
// (it converges like h^a for 1 < a < 2, where x^(a-1) is not smooth at 0).
[[nodiscard]] inline double Ex02LeftPiece(double a, double tol = 1e-10) {
  if (a <= 0.0) throw std::domain_error("ex02_left_piece: a must be positive");
  if (a < 1.0)
    return IntegrateToTolerance(
               [a](double u) { return std::exp(-std::pow(u, 1.0 / a)); }, 0.0,
               1.0, tol)
               .value /
           a;
  return IntegrateToTolerance(
             [a](double x) { return std::pow(x, a - 1.0) * std::exp(-x); }, 0.0,
             1.0, tol)
      .value;
}
[[nodiscard]] inline double Ex02RightPiece(double a, double tol = 1e-10) {
  if (a <= 0.0) throw std::domain_error("ex02_right_piece: a must be positive");
  return ImproperIntegral(
             [a](double x) { return std::pow(x, a - 1.0) * std::exp(-x); }, 1.0,
             kInfinity, tol)
      .value;
}

// f(a) = left piece + right piece, accurate for every a > 0.
[[nodiscard]] inline double Ex02Gamma(double a, double tol = 1e-10) {
  return Ex02LeftPiece(a, tol) + Ex02RightPiece(a, tol);
}

}  // namespace p4mfe::ch02::ex
