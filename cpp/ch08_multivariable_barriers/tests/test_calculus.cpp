#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch08/chain_rule.hpp>
#include <p4mfe/ch08/double_integrals.hpp>
#include <p4mfe/ch08/extrema.hpp>

using namespace p4mfe::ch08;
using std::numbers::pi;

// ---------------------------------------------------------------- 8.1 -----
TEST(Ch08ChainRule, Curve) {
  // f(x, y) = x^2 y + e^y on g(t) = (cos t, t^2):
  //   d/dt = 2 x y x' + (x^2 + e^y) y' = -2 cos t sin t t^2 + (cos^2 t +
  //   e^(t^2)) 2t.
  const auto f = [](const Vector& v) {
    return v[0] * v[0] * v[1] + std::exp(v[1]);
  };
  const auto g = [](double t) { return Vector{std::cos(t), t * t}; };
  for (double t : {-0.5, 0.3, 1.1}) {
    const double exact = -2 * std::cos(t) * std::sin(t) * t * t +
                         (std::cos(t) * std::cos(t) + std::exp(t * t)) * 2 * t;
    EXPECT_NEAR(chain_rule_derivative(f, g, t), exact, 1e-7);
    EXPECT_NEAR(chain_rule_derivative(f, g, t),
                p4mfe::ch01::derivative([&](double s) { return f(g(s)); }, t),
                1e-7);
  }
}

TEST(Ch08ChainRule, GradientAndJacobian) {
  // f(x, y) = x y with polar coordinates G(r, theta): f(G) = r^2 cos sin = r^2
  // sin(2 theta) / 2.
  const auto f = [](const Vector& v) { return v[0] * v[1]; };
  const auto G = [](const Vector& p) {
    return Vector{p[0] * std::cos(p[1]), p[0] * std::sin(p[1])};
  };
  const Vector x{1.5, 0.4};
  const auto grad = chain_rule_gradient(f, G, x);
  EXPECT_NEAR(grad[0], x[0] * std::sin(2 * x[1]), 1e-7);
  EXPECT_NEAR(grad[1], x[0] * x[0] * std::cos(2 * x[1]), 1e-7);
  // Polar -> Cartesian -> polar is the identity, so its Jacobian is I.
  const auto F = [](const Vector& v) {
    return Vector{std::hypot(v[0], v[1]), std::atan2(v[1], v[0])};
  };
  const auto J = chain_rule_jacobian(F, G, x);
  EXPECT_NEAR(J(0, 0), 1.0, 1e-7);
  EXPECT_NEAR(J(0, 1), 0.0, 1e-7);
  EXPECT_NEAR(J(1, 0), 0.0, 1e-7);
  EXPECT_NEAR(J(1, 1), 1.0, 1e-7);
}

// ---------------------------------------------------------------- 8.2 -----
TEST(Ch08DoubleIntegrals, RectangleAndRegion) {
  // integral over [0,1]x[0,2] of x y^2 = 1/2 * 8/3.
  EXPECT_NEAR(
      double_integral([](double x, double y) { return x * y * y; }, 0, 1, 0, 2),
      4.0 / 3, 1e-12);
  // Fubini: the order does not matter.
  const auto f = [](double x, double y) {
    return std::exp(-x * y) * std::cos(x + y);
  };
  EXPECT_NEAR(
      double_integral(f, 0, 1, 0, 2),
      double_integral([&](double y, double x) { return f(x, y); }, 0, 2, 0, 1),
      1e-12);
  // Triangle 0 <= y <= x <= 1, f = 1: area 1/2; f = y: 1/6.
  const auto zero = [](double) { return 0.0; };
  const auto diag = [](double x) { return x; };
  EXPECT_NEAR(double_integral_region([](double, double) { return 1.0; }, 0, 1,
                                     zero, diag),
              0.5, 1e-12);
  EXPECT_NEAR(double_integral_region([](double, double y) { return y; }, 0, 1,
                                     zero, diag),
              1.0 / 6, 1e-12);
}

// -------------------------------------------------------------- 8.3 -------
TEST(Ch08DoubleIntegrals, ChangeOfVariables) {
  // Unit disk area by polar coordinates, by the general formula, and via |det
  // J| = r.
  EXPECT_NEAR(polar_integral([](double, double) { return 1.0; }, 0, 1), pi,
              1e-10);
  EXPECT_NEAR(
      jacobian_determinant([](double r, double t) { return from_polar(r, t); },
                           2.0, 0.7),
      2.0, 1e-8);
  EXPECT_NEAR(
      change_of_variables_integral(
          [](double, double) { return 1.0; },
          [](double r, double t) { return from_polar(r, t); }, 0, 1, 0, 2 * pi),
      pi, 1e-8);
  // Linear map (u, v) -> (u + v, u - v), |det| = 2, over [0,1]^2: integral of
  // x^2.
  const auto lin = [](double u, double v) { return Point2{u + v, u - v}; };
  const double exact =
      2 * double_integral([](double u, double v) { return (u + v) * (u + v); },
                          0, 1, 0, 1);
  EXPECT_NEAR(change_of_variables_integral(
                  [](double x, double) { return x * x; }, lin, 0, 1, 0, 1),
              exact, 1e-8);
  // Annulus integral of x^2 + y^2: 2 pi (r1^4 - r0^4) / 4.
  EXPECT_NEAR(
      polar_integral([](double x, double y) { return x * x + y * y; }, 1, 2),
      pi * 15 / 2, 1e-9);
}

// ---------------------------------------------------------------- 8.6 -----
TEST(Ch08DoubleIntegrals, NormalDensityIntegral) {
  for (double R : {0.5, 1.0, 3.0}) {
    const double disk = polar_integral(
        [](double x, double y) { return std::exp(-0.5 * (x * x + y * y)); }, 0,
        R);
    EXPECT_NEAR(disk, gaussian_disk_integral(R), 1e-8);
    // disk(R) <= square(R) <= disk(R sqrt 2).
    EXPECT_LT(gaussian_disk_integral(R), gaussian_square_integral(R));
    EXPECT_LT(gaussian_square_integral(R),
              gaussian_disk_integral(R * std::numbers::sqrt2));
  }
  EXPECT_NEAR(gaussian_disk_integral(40), 2 * pi, 1e-15);
  EXPECT_NEAR(std::sqrt(gaussian_square_integral(12)), std::sqrt(2 * pi),
              1e-12);
}

// ---------------------------------------------------------------- 8.4 -----
TEST(Ch08Extrema, Classify) {
  const auto bowl = [](const Vector& v) {
    return (v[0] - 1) * (v[0] - 1) + 2 * (v[1] + 0.5) * (v[1] + 0.5) +
           v[0] * v[1];
  };
  const auto x = find_critical_point(bowl, {0.0, 0.0});
  // grad = (2(x-1) + y, 4(y+1/2) + x) = 0 -> x = 10/7, y = -6/7.
  EXPECT_NEAR(x[0], 10.0 / 7, 1e-7);
  EXPECT_NEAR(x[1], -6.0 / 7, 1e-7);
  EXPECT_EQ(classify_critical_point(bowl, x), CriticalPointType::Minimum);

  // -bowl has a maximum there; x^2 - y^2 a saddle at 0.
  const auto neg = [&](const Vector& v) { return -bowl(v); };
  EXPECT_EQ(classify_critical_point(neg, x), CriticalPointType::Maximum);
  const auto saddle = [](const Vector& v) { return v[0] * v[0] - v[1] * v[1]; };
  EXPECT_EQ(classify_critical_point(saddle, {0, 0}), CriticalPointType::Saddle);
  // f = x^4 + y^2 at 0: Hessian singular, test inconclusive.
  EXPECT_EQ(classify_critical_point(
                [](const Vector& v) { return std::pow(v[0], 4) + v[1] * v[1]; },
                {0, 0}),
            CriticalPointType::Degenerate);

  // f(x, y) = x^3 - 3x + y^2: (1, 0) minimum, (-1, 0) saddle.
  const auto cubic = [](const Vector& v) {
    return v[0] * v[0] * v[0] - 3 * v[0] + v[1] * v[1];
  };
  const auto m = find_critical_point(cubic, {2.0, 1.0});
  EXPECT_NEAR(m[0], 1.0, 1e-7);
  EXPECT_EQ(classify_critical_point(cubic, m), CriticalPointType::Minimum);
  const auto s = find_critical_point(cubic, {-2.0, 1.0});
  EXPECT_NEAR(s[0], -1.0, 1e-7);
  EXPECT_EQ(classify_critical_point(cubic, s), CriticalPointType::Saddle);

  // Three variables: diag(2, 3, -1) is indefinite; diag(-1, -2, -3) negative
  // definite.
  EXPECT_EQ(classify_hessian({{2, 0, 0}, {0, 3, 0}, {0, 0, -1}}),
            CriticalPointType::Saddle);
  EXPECT_EQ(classify_hessian({{-1, 0, 0}, {0, -2, 0}, {0, 0, -3}}),
            CriticalPointType::Maximum);
  EXPECT_NEAR(determinant({{0, 1}, {1, 0}}), -1.0, 1e-15);
  const auto D = leading_principal_minors({{2, 1}, {1, 2}});
  EXPECT_DOUBLE_EQ(D[0], 2.0);
  EXPECT_DOUBLE_EQ(D[1], 3.0);
}
