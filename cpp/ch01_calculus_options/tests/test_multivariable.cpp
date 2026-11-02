#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch01/multivariable.hpp>

using namespace p4mfe::ch01;

namespace {
// f(x, y) = x^2 y + e^(x y)
double F(const Vector& v) { return v[0] * v[0] * v[1] + std::exp(v[0] * v[1]); }
}  // namespace

TEST(Ch01Multivariable, GradientTwoVariables) {
  const double x = 0.5, y = -1.2;
  const auto g = Gradient(F, {x, y});
  EXPECT_NEAR(g[0], 2 * x * y + y * std::exp(x * y), 1e-8);
  EXPECT_NEAR(g[1], x * x + x * std::exp(x * y), 1e-8);
}

TEST(Ch01Multivariable, HessianTwoVariables) {
  const double x = 0.5, y = -1.2;
  const double e = std::exp(x * y);
  const auto hessian = Hessian(F, {x, y});
  EXPECT_NEAR(hessian[0][0], 2 * y + y * y * e, 1e-6);
  EXPECT_NEAR(hessian[1][1], x * x * e, 1e-6);
  EXPECT_NEAR(hessian[0][1], 2 * x + e + x * y * e, 1e-6);
  EXPECT_DOUBLE_EQ(hessian[0][1], hessian[1][0]);
}

TEST(Ch01Multivariable, ThreeVariables) {
  // f(x, y, z) = x y z + z^2: gradient (yz, xz, xy + 2z)
  const auto g =
      Gradient([](const Vector& v) { return v[0] * v[1] * v[2] + v[2] * v[2]; },
               {1.0, 2.0, 3.0});
  EXPECT_NEAR(g[0], 6.0, 1e-8);
  EXPECT_NEAR(g[1], 3.0, 1e-8);
  EXPECT_NEAR(g[2], 8.0, 1e-8);
}

TEST(Ch01Multivariable, RejectsBadIndex) {
  EXPECT_THROW((void)Partial(F, {1.0, 2.0}, 2), std::out_of_range);
}
