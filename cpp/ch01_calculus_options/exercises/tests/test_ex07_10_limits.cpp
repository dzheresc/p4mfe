#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch01/exercises/ex07_10_limits.hpp>

using namespace p4mfe::ch01::ex;

TEST(Ch01Ex07, BoundsAroundE) {
  for (double x : {1.0, 1.5, 2.0, 10.0, 1e3, 1e6}) {
    const auto [lo, hi] = Ex07EBounds(x);
    EXPECT_LT(lo, std::numbers::e) << x;
    EXPECT_GT(hi, std::numbers::e) << x;
    // The gap is (1+1/x)^x / x < e / x.
    EXPECT_LT(hi - lo, std::numbers::e / x);
  }
  const auto [lo1, hi1] = Ex07EBounds(1.0);
  EXPECT_DOUBLE_EQ(lo1, 2.0);
  EXPECT_DOUBLE_EQ(hi1, 4.0);
}

TEST(Ch01Ex08, TaylorRemainders) {
  for (double x : {1e-1, 1e-3, 1e-6, -1e-6, -1e-3}) {
    EXPECT_NEAR(Ex08SqrtQuotient(x), kEx08SqrtLimit,
                std::abs(x));  // next term x/16
    EXPECT_NEAR(Ex08ExpQuotient(x), kEx08ExpLimit,
                std::abs(x));  // next term x/24
  }
  EXPECT_NEAR(Ex08SqrtQuotient(1e-12), -0.125, 1e-12);
  EXPECT_NEAR(Ex08ExpQuotient(1e-12), 1.0 / 6.0, 1e-12);
  // The stable forms agree with the naive ones where the naive ones are
  // accurate.
  const double x = 0.3;
  EXPECT_NEAR(Ex08SqrtQuotient(x), (std::sqrt(1 + x) - 1 - x / 2) / (x * x),
              1e-12);
  EXPECT_NEAR(Ex08ExpQuotient(x),
              (std::exp(x) - 1 - x - x * x / 2) / (x * x * x), 1e-12);
  EXPECT_NEAR(Ex08ExpQuotient(0.005),
              (std::expm1(0.005) - 0.005 - 0.0000125) / 1.25e-7, 1e-6);
}

TEST(Ch01Ex09, LimitsAtInfinity) {
  for (double x : {1e3, 1e6, 1e9})
    EXPECT_NEAR(Ex09ExprI(x), kEx09LimitI, 10.0 / x);
  // Naive check where it is still accurate.
  const double x = 100.0;
  EXPECT_NEAR(Ex09ExprI(x), 1.0 / (std::sqrt(x * x - 4 * x + 1) - x), 1e-12);
  EXPECT_NEAR(Ex09ExprIi(x), 1.0 / (std::sqrt(x * x - 4 * x + 1) - x + 2),
              1e-9);
  // (ii) diverges to -infinity like -2x/3.
  EXPECT_TRUE(std::isinf(kEx09LimitIi) && kEx09LimitIi < 0);
  for (double y : {1e3, 1e6, 1e9})
    EXPECT_NEAR(Ex09ExprIi(y) / y, -2.0 / 3.0, 2.0 / y);
}

TEST(Ch01Ex10, InverseE) {
  for (double x : {1e2, 1e4, 1e8})
    EXPECT_NEAR(Ex10Expr(x), 1.0 / std::numbers::e, 1.0 / x);
  // Monotone increasing towards 1/e from below.
  EXPECT_LT(Ex10Expr(10.0), Ex10Expr(100.0));
  EXPECT_LT(Ex10Expr(1e6), 1.0 / std::numbers::e);
}
