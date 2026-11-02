#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch01/exercises/ex07_10_limits.hpp>

using namespace p4mfe::ch01::ex;

TEST(Ch01Ex07, BoundsAroundE) {
  for (double x : {1.0, 1.5, 2.0, 10.0, 1e3, 1e6}) {
    const auto [lo, hi] = ex07_e_bounds(x);
    EXPECT_LT(lo, std::numbers::e) << x;
    EXPECT_GT(hi, std::numbers::e) << x;
    // The gap is (1+1/x)^x / x < e / x.
    EXPECT_LT(hi - lo, std::numbers::e / x);
  }
  const auto [lo1, hi1] = ex07_e_bounds(1.0);
  EXPECT_DOUBLE_EQ(lo1, 2.0);
  EXPECT_DOUBLE_EQ(hi1, 4.0);
}

TEST(Ch01Ex08, TaylorRemainders) {
  for (double x : {1e-1, 1e-3, 1e-6, -1e-6, -1e-3}) {
    EXPECT_NEAR(ex08_sqrt_quotient(x), ex08_sqrt_limit,
                std::abs(x));  // next term x/16
    EXPECT_NEAR(ex08_exp_quotient(x), ex08_exp_limit,
                std::abs(x));  // next term x/24
  }
  EXPECT_NEAR(ex08_sqrt_quotient(1e-12), -0.125, 1e-12);
  EXPECT_NEAR(ex08_exp_quotient(1e-12), 1.0 / 6.0, 1e-12);
  // The stable forms agree with the naive ones where the naive ones are
  // accurate.
  const double x = 0.3;
  EXPECT_NEAR(ex08_sqrt_quotient(x), (std::sqrt(1 + x) - 1 - x / 2) / (x * x),
              1e-12);
  EXPECT_NEAR(ex08_exp_quotient(x),
              (std::exp(x) - 1 - x - x * x / 2) / (x * x * x), 1e-12);
  EXPECT_NEAR(ex08_exp_quotient(0.005),
              (std::expm1(0.005) - 0.005 - 0.0000125) / 1.25e-7, 1e-6);
}

TEST(Ch01Ex09, LimitsAtInfinity) {
  for (double x : {1e3, 1e6, 1e9})
    EXPECT_NEAR(ex09_expr_i(x), ex09_limit_i, 10.0 / x);
  // Naive check where it is still accurate.
  const double x = 100.0;
  EXPECT_NEAR(ex09_expr_i(x), 1.0 / (std::sqrt(x * x - 4 * x + 1) - x), 1e-12);
  EXPECT_NEAR(ex09_expr_ii(x), 1.0 / (std::sqrt(x * x - 4 * x + 1) - x + 2),
              1e-9);
  // (ii) diverges to -infinity like -2x/3.
  EXPECT_TRUE(std::isinf(ex09_limit_ii) && ex09_limit_ii < 0);
  for (double y : {1e3, 1e6, 1e9})
    EXPECT_NEAR(ex09_expr_ii(y) / y, -2.0 / 3.0, 2.0 / y);
}

TEST(Ch01Ex10, InverseE) {
  for (double x : {1e2, 1e4, 1e8})
    EXPECT_NEAR(ex10_expr(x), 1.0 / std::numbers::e, 1.0 / x);
  // Monotone increasing towards 1/e from below.
  EXPECT_LT(ex10_expr(10.0), ex10_expr(100.0));
  EXPECT_LT(ex10_expr(1e6), 1.0 / std::numbers::e);
}
