#pragma once
// Chapter 8, sections 8.2, 8.3 and 8.6: double integrals, change of
// variables, and the integral of the standard normal density.
//
// 8.2   Over a rectangle (Fubini) or a region a <= x <= b,
//       y_lo(x) <= y <= y_hi(x), a double integral is an iterated one; both
//       one-dimensional integrals use chapter 2's rules (Simpson by default).
// 8.3   For a one-to-one map (x, y) = T(u, v),
//         integral over T(D) of f(x, y) dx dy = integral over D of f(T(u, v))
//         |det J_T(u, v)| du dv.
// 8.3.1 Polar coordinates x = r cos(theta), y = r sin(theta): |det J| = r.
// 8.6   I = integral of e^(-x^2/2) over R. Then
//         I^2 = double integral of e^(-(x^2+y^2)/2) = integral_0^(2 pi)
//         integral_0^inf e^(-r^2/2) r dr dtheta = 2 pi,
//       so I = sqrt(2 pi) and the standard normal density integrates to 1.
//       Over the disk of radius R the polar integral is 2 pi (1 - e^(-R^2/2));
//       the square [-R, R]^2 lies between the disks of radius R and R sqrt(2).

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>

namespace p4mfe::ch08 {

using Point2 = std::array<double, 2>;

// ---------------------------------------------------------------- 8.2 ------
// integral_a^b integral_c^d f(x, y) dy dx.
template <std::invocable<double, double> F>
[[nodiscard]] double double_integral(F&& f, double a, double b, double c,
                                     double d, std::size_t nx = 200,
                                     std::size_t ny = 200,
                                     ch02::Rule rule = ch02::Rule::Simpson) {
  return ch02::integrate(
      [&](double x) {
        return ch02::integrate([&](double y) { return f(x, y); }, c, d, ny,
                               rule);
      },
      a, b, nx, rule);
}

// integral_a^b integral_{y_lo(x)}^{y_hi(x)} f(x, y) dy dx.
template <std::invocable<double, double> F, std::invocable<double> Lo,
          std::invocable<double> Hi>
[[nodiscard]] double double_integral_region(
    F&& f, double a, double b, Lo&& y_lo, Hi&& y_hi, std::size_t nx = 200,
    std::size_t ny = 200, ch02::Rule rule = ch02::Rule::Simpson) {
  return ch02::integrate(
      [&](double x) {
        return ch02::integrate([&](double y) { return f(x, y); }, y_lo(x),
                               y_hi(x), ny, rule);
      },
      a, b, nx, rule);
}

// ---------------------------------------------------------------- 8.3 ------
// |det J_T(u, v)| by central differences.
template <std::invocable<double, double> T>
[[nodiscard]] double jacobian_determinant(T&& map, double u, double v) {
  const double hu = 1e-6 * std::max(1.0, std::abs(u));
  const double hv = 1e-6 * std::max(1.0, std::abs(v));
  const Point2 pu = map(u + hu, v), mu = map(u - hu, v);
  const Point2 pv = map(u, v + hv), mv = map(u, v - hv);
  const double xu = (pu[0] - mu[0]) / (2 * hu), yu = (pu[1] - mu[1]) / (2 * hu);
  const double xv = (pv[0] - mv[0]) / (2 * hv), yv = (pv[1] - mv[1]) / (2 * hv);
  return std::abs(xu * yv - xv * yu);
}

// integral over [u0, u1] x [v0, v1] of f(T(u, v)) |det J_T(u, v)| du dv.
template <std::invocable<double, double> F, std::invocable<double, double> T>
[[nodiscard]] double change_of_variables_integral(F&& f, T&& map, double u0,
                                                  double u1, double v0,
                                                  double v1,
                                                  std::size_t nu = 200,
                                                  std::size_t nv = 200) {
  return double_integral(
      [&](double u, double v) {
        const Point2 p = map(u, v);
        return f(p[0], p[1]) * jacobian_determinant(map, u, v);
      },
      u0, u1, v0, v1, nu, nv);
}

// -------------------------------------------------------------- 8.3.1 ------
[[nodiscard]] inline Point2 from_polar(double r, double theta) noexcept {
  return {r * std::cos(theta), r * std::sin(theta)};
}

// integral_{theta0}^{theta1} integral_{r0}^{r1} f(r cos theta, r sin theta) r
// dr dtheta.
template <std::invocable<double, double> F>
[[nodiscard]] double polar_integral(F&& f, double r0, double r1,
                                    double theta0 = 0.0,
                                    double theta1 = 2.0 * std::numbers::pi,
                                    std::size_t nr = 200,
                                    std::size_t ntheta = 200) {
  return double_integral(
      [&](double r, double theta) {
        const Point2 p = from_polar(r, theta);
        return f(p[0], p[1]) * r;
      },
      r0, r1, theta0, theta1, nr, ntheta);
}

// ---------------------------------------------------------------- 8.6 ------
// Closed form of the polar integral of e^(-(x^2+y^2)/2) over the disk of radius
// R.
[[nodiscard]] inline double gaussian_disk_integral(double R) noexcept {
  return 2.0 * std::numbers::pi * -std::expm1(-0.5 * R * R);
}

// integral over [-R, R]^2 of e^(-(x^2+y^2)/2) = (integral_{-R}^R e^(-x^2/2)
// dx)^2.
[[nodiscard]] inline double gaussian_square_integral(double R,
                                                     std::size_t n = 400) {
  const double I = ch02::simpson_rule(
      [](double x) { return std::exp(-0.5 * x * x); }, -R, R, n);
  return I * I;
}

}  // namespace p4mfe::ch08
