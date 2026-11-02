#pragma once
// Exercise 1 (section 9.6): extrema of a linear function on the intersection
// of two quadrics.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/ch09/lagrange.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch09::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Find the maximum and minimum of
//   f(x1, x2, x3, x4) = x1 - 2 x2 + 3 x3 - 4 x4
// subject to the constraints
//   x1^2 + x2^2 = x4^2 + 1   and   x1^2 + 2 x2^2 + 3 x3^2 = 6.
//
// Solution.
// Write the constraints as g(x) = 0 with
//   g1(x) = x1^2 + x2^2 - x4^2 - 1,   g2(x) = x1^2 + 2 x2^2 + 3 x3^2 - 6.
//
// Existence. g2 = 0 is an ellipsoid in (x1, x2, x3), so |x1| <= sqrt 6,
// |x2| <= sqrt 3, |x3| <= sqrt 2, and then x4^2 = x1^2 + x2^2 - 1 <= 8. The
// feasible set is closed and bounded, hence compact, and the continuous f
// attains a global maximum and minimum on it.
//
// Regularity. grad g1 = (2x1, 2x2, 0, -2x4), grad g2 = (2x1, 4x2, 6x3, 0).
// Suppose a grad g1 + b grad g2 = 0 with (a, b) != 0. The third and fourth
// entries give b x3 = 0 and a x4 = 0.
//   b = 0: then a != 0 and x1 = x2 = x4 = 0, against g1 = 0 (0 = 0 + 1).
//   a = 0: then b != 0 and x1 = x2 = x3 = 0, against g2 = 0 (0 = 6).
//   a, b != 0: then x3 = x4 = 0, so x1^2 + x2^2 = 1 and x1^2 + 2 x2^2 = 6,
//              i.e. x2^2 = 5 and x1^2 = -4: impossible.
// So the constraint gradients are independent at every feasible point, and
// both global extrema are among the solutions of the Lagrange system
// (L = f + lambda1 g1 + lambda2 g2):
//    1 + 2 (lambda1 + lambda2) x1 = 0
//   -2 + 2 (lambda1 + 2 lambda2) x2 = 0
//    3 + 6 lambda2 x3 = 0
//   -4 - 2 lambda1 x4 = 0,         g1(x) = 0,  g2(x) = 0.
// The first four equations force lambda1, lambda2, lambda1 + lambda2,
// lambda1 + 2 lambda2 to be nonzero and give x in terms of the multipliers:
//   x1 = -1 / (2 (lambda1 + lambda2)),   x2 = 1 / (lambda1 + 2 lambda2),
//   x3 = -1 / (2 lambda2),               x4 = -2 / lambda1.
// Substituting into the constraints leaves two equations in (lambda1, lambda2)
//   1/(4 (l1 + l2)^2) + 1/(l1 + 2 l2)^2 - 4/l1^2 = 1,
//   1/(4 (l1 + l2)^2) + 2/(l1 + 2 l2)^2 + 3/(4 l2^2) = 6,
// which have no closed-form solution; they are solved numerically, here by
// Newton's method on the full 6 x 6 Lagrange system (lagrange_stationary_point)
// started from every point of the grid {-2, -1, 1, 2}^4. That finds four
// stationary points, two of them symmetric pairs:
//   x* = ( 2.137669,  -0.683412,   0.406720,  -2.009150),
//        lambda = (0.995446, -1.229346),  f(x*) =  12.761255  (maximum)
//   x  = (-0.375584,   1.385891,   0.820071,  -1.030416),  f = 3.434511
//   (saddle) and their negatives, with f = -12.761255 (minimum) and -3.434511
//   (saddle).
// The symmetry is exact: f is odd and g is even, so x -> -x maps feasible
// points to feasible points and stationary points to stationary points
// (with lambda -> -lambda) and f -> -f. Hence min f = -max f.
//
// Answer: max f = 12.761255 at x*, min f = -12.761255 at -x*.
//
// Checks (in the tests): the second-order test on the tangent space of the
// constraints classifies x* as a maximum, and a dense sweep of the whole
// feasible set (x1, x2, x3 on the ellipsoid, x4 = -/+ sqrt(x1^2 + x2^2 - 1)
// with the sign that increases/decreases f) never beats the Lagrange values.

[[nodiscard]] inline double ex01_f(const Vector& x) {
  return x[0] - 2.0 * x[1] + 3.0 * x[2] - 4.0 * x[3];
}

[[nodiscard]] inline Vector ex01_g(const Vector& x) {
  return {x[0] * x[0] + x[1] * x[1] - x[3] * x[3] - 1.0,
          x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2] - 6.0};
}

// x from the multipliers (the first four Lagrange equations).
[[nodiscard]] inline Vector ex01_x_from_lambda(const Vector& lambda) {
  const double l1 = lambda[0], l2 = lambda[1];
  return {-1.0 / (2.0 * (l1 + l2)), 1.0 / (l1 + 2.0 * l2), -1.0 / (2.0 * l2),
          -2.0 / l1};
}

struct Ex01Point {
  Vector x;
  Vector lambda;
  double value;
  CriticalPointType type;
};

// All stationary points found by Newton's method from the grid {-2,-1,1,2}^4,
// without duplicates, sorted by decreasing f.
[[nodiscard]] inline std::vector<Ex01Point> ex01_stationary_points() {
  const double grid[] = {-2.0, -1.0, 1.0, 2.0};
  std::vector<Ex01Point> pts;
  for (double a : grid)
    for (double b : grid)
      for (double c : grid)
        for (double d : grid) {
          LagrangeResult r;
          try {
            r = lagrange_stationary_point(ex01_f, ex01_g, {a, b, c, d});
          } catch (const std::exception&) {
            continue;  // Newton did not converge from this start
          }
          const Vector gx = ex01_g(r.x);
          if (std::abs(gx[0]) > 1e-8 || std::abs(gx[1]) > 1e-8) continue;
          const bool seen =
              std::any_of(pts.begin(), pts.end(), [&](const Ex01Point& p) {
                double d2 = 0.0;
                for (std::size_t i = 0; i < 4; ++i)
                  d2 += (p.x[i] - r.x[i]) * (p.x[i] - r.x[i]);
                return d2 < 1e-10;
              });
          if (seen) continue;
          const auto type = classify_constrained(ex01_f, ex01_g, r.x, r.lambda);
          pts.push_back({r.x, r.lambda, ex01_f(r.x), type});
        }
  std::sort(pts.begin(), pts.end(), [](const Ex01Point& p, const Ex01Point& q) {
    return p.value > q.value;
  });
  return pts;
}

// Brute-force check: sweep (x1, x2, x3) = (sqrt6 sin t cos p, sqrt3 sin t sin
// p, sqrt2 cos t) over an n_t x n_p grid; where x1^2 + x2^2 >= 1 take x4 = -/+
// sqrt(x1^2 + x2^2 - 1). Returns {largest f, smallest f}.
struct Ex01Sweep {
  double max, min;
};
[[nodiscard]] inline Ex01Sweep ex01_sweep(std::size_t n_t, std::size_t n_p) {
  const double pi = std::numbers::pi;
  Ex01Sweep s{-1e300, 1e300};
  for (std::size_t i = 0; i <= n_t; ++i) {
    const double t = pi * static_cast<double>(i) / static_cast<double>(n_t);
    for (std::size_t j = 0; j < n_p; ++j) {
      const double p =
          2.0 * pi * static_cast<double>(j) / static_cast<double>(n_p);
      const double x1 = std::sqrt(6.0) * std::sin(t) * std::cos(p);
      const double x2 = std::sqrt(3.0) * std::sin(t) * std::sin(p);
      const double x3 = std::sqrt(2.0) * std::cos(t);
      const double r = x1 * x1 + x2 * x2 - 1.0;
      if (r < 0.0) continue;
      const double lin = x1 - 2.0 * x2 + 3.0 * x3, q = 4.0 * std::sqrt(r);
      s.max = std::max(s.max, lin + q);
      s.min = std::min(s.min, lin - q);
    }
  }
  return s;
}

}  // namespace p4mfe::ch09::ex
