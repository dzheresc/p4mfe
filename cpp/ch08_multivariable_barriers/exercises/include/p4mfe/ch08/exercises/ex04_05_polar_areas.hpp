#pragma once
// Exercises 4-5 (section 8.12): the areas of a disk and of an ellipse by
// polar coordinates.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/ch08/double_integrals.hpp>

namespace p4mfe::ch08::ex {

// ------------------------------------------------------------ Exercise 4 ---
// Use polar coordinates to show that the area of a circle of radius R is
// pi R^2, i.e. that the integral of 1 over D(0, R) is pi R^2.
//
// The map (r, theta) -> (r cos theta, r sin theta) takes
// [0, R] x [0, 2 pi) one-to-one (up to the null set r = 0) onto D(0, R), and
//   det J = det [[cos theta, -r sin theta], [sin theta, r cos theta]] = r
//   (cos^2 + sin^2) = r.
// By the change of variables formula (8.3),
//   integral_{D(0,R)} 1 dx dy = integral_0^{2 pi} integral_0^R r dr dtheta
//                             = integral_0^{2 pi} R^2/2 dtheta = pi R^2.
[[nodiscard]] inline double ex04_disk_area_polar(double R, std::size_t n = 50) {
  return polar_integral([](double, double) { return 1.0; }, 0.0, R, 0.0,
                        2.0 * std::numbers::pi, n, n);
}

// The same area as an x-y integral: integral_{-R}^R 2 sqrt(R^2 - x^2) dx.
// The square-root endpoints make Simpson's rule converge only like n^(-3/2),
// which is why polar coordinates are the better choice here.
[[nodiscard]] inline double ex04_disk_area_cartesian(double R,
                                                     std::size_t n = 2000) {
  return double_integral_region(
      [](double, double) { return 1.0; }, -R, R,
      [R](double x) { return -std::sqrt(std::max(R * R - x * x, 0.0)); },
      [R](double x) { return std::sqrt(std::max(R * R - x * x, 0.0)); }, n, 2);
}

// ------------------------------------------------------------ Exercise 5 ---
// Compute the area of the ellipse D = {x^2/a^2 + y^2/b^2 <= 1}, a, b > 0,
// using the change of variables x = a r cos theta, y = b r sin theta.
//
// The map takes [0, 1] x [0, 2 pi) onto D (x^2/a^2 + y^2/b^2 = r^2 <= 1), and
//   det J = det [[a cos theta, -a r sin theta], [b sin theta, b r cos theta]] =
//   a b r.
// Hence
//   area(D) = integral_0^{2 pi} integral_0^1 a b r dr dtheta = 2 pi a b / 2 =
//   pi a b.
// (For a = b = R this is exercise 4.)
[[nodiscard]] inline Point2 ex05_map(double a, double b, double r,
                                     double theta) {
  return {a * r * std::cos(theta), b * r * std::sin(theta)};
}

// Uses the numerical Jacobian of the chapter's change_of_variables_integral.
[[nodiscard]] inline double ex05_ellipse_area(double a, double b,
                                              std::size_t n = 50) {
  return change_of_variables_integral(
      [](double, double) { return 1.0; },
      [a, b](double r, double th) { return ex05_map(a, b, r, th); }, 0.0, 1.0,
      0.0, 2.0 * std::numbers::pi, n, n);
}

[[nodiscard]] inline double ex05_exact(double a, double b) {
  return std::numbers::pi * a * b;
}

}  // namespace p4mfe::ch08::ex
