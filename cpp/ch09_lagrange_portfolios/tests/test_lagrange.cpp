#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch09/lagrange.hpp>

using namespace p4mfe::ch09;

// ---------------------------------------------------------------- 9.1 -----
TEST(Ch09Lagrange, LinearOnCircle) {
  // Extrema of x + 2y on x^2 + y^2 = 5: max 5 at (1, 2), min -5 at (-1, -2).
  const auto f = [](const Vector& v) { return v[0] + 2 * v[1]; };
  const auto g = [](const Vector& v) {
    return Vector{v[0] * v[0] + v[1] * v[1] - 5};
  };
  const auto mx = lagrange_stationary_point(f, g, {0.8, 1.5});
  EXPECT_NEAR(mx.x[0], 1.0, 1e-8);
  EXPECT_NEAR(mx.x[1], 2.0, 1e-8);
  EXPECT_NEAR(mx.lambda[0], -0.5, 1e-8);  // grad f + lambda grad g = 0
  EXPECT_EQ(classify_constrained(f, g, mx.x, mx.lambda),
            CriticalPointType::Maximum);
  const auto mn = lagrange_stationary_point(f, g, {-1.2, -1.8});
  EXPECT_NEAR(f(mn.x), -5.0, 1e-8);
  EXPECT_EQ(classify_constrained(f, g, mn.x, mn.lambda),
            CriticalPointType::Minimum);
  // The Lagrangian's gradient vanishes there.
  for (double v : lagrangian_gradient(f, g, mx.x, mx.lambda))
    EXPECT_NEAR(v, 0.0, 1e-8);
}

// -------------------------------------------------------------- 9.1.1 -----
TEST(Ch09Lagrange, TwoConstraints) {
  // Closest point to the origin on the line x + y + z = 3, x - y = 1: (1.5,
  // 0.5, 1).
  const auto f = [](const Vector& v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  };
  const auto g = [](const Vector& v) {
    return Vector{v[0] + v[1] + v[2] - 3, v[0] - v[1] - 1};
  };
  const auto r = lagrange_stationary_point(f, g, {0, 0, 0});
  EXPECT_NEAR(r.x[0], 1.5, 1e-8);
  EXPECT_NEAR(r.x[1], 0.5, 1e-8);
  EXPECT_NEAR(r.x[2], 1.0, 1e-8);
  ASSERT_EQ(r.lambda.size(), 2u);
  EXPECT_EQ(classify_constrained(f, g, r.x, r.lambda),
            CriticalPointType::Minimum);
}

TEST(Ch09Lagrange, BoxOfMaximumVolume) {
  // Max xyz with surface area 2(xy + yz + zx) = 6: the unit cube.
  const auto f = [](const Vector& v) { return v[0] * v[1] * v[2]; };
  const auto g = [](const Vector& v) {
    return Vector{2 * (v[0] * v[1] + v[1] * v[2] + v[2] * v[0]) - 6};
  };
  const auto r = lagrange_stationary_point(f, g, {0.8, 1.2, 1.1});
  for (double v : r.x) EXPECT_NEAR(v, 1.0, 1e-8);
  EXPECT_EQ(classify_constrained(f, g, r.x, r.lambda),
            CriticalPointType::Maximum);
}

TEST(Ch09Lagrange, SaddleAndErrors) {
  // x^2 - y^2 - z^2... on the plane z = 0, f = x^2 - y^2 has a saddle at 0.
  const auto f = [](const Vector& v) {
    return v[0] * v[0] - v[1] * v[1] + v[2];
  };
  const auto g = [](const Vector& v) { return Vector{v[2]}; };
  const auto r = lagrange_stationary_point(f, g, {0.3, -0.2, 0.1});
  EXPECT_NEAR(r.x[0], 0.0, 1e-8);
  EXPECT_NEAR(r.lambda[0], -1.0, 1e-8);
  EXPECT_EQ(classify_constrained(f, g, r.x, r.lambda),
            CriticalPointType::Saddle);
  // As many constraints as unknowns is not a Lagrange problem.
  EXPECT_THROW((void)lagrange_stationary_point(
                   f, [](const Vector& v) { return Vector{v[0], v[1], v[2]}; },
                   {0, 0, 0}),
               std::invalid_argument);
}
