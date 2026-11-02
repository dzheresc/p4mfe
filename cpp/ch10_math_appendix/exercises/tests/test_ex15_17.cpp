#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch10/exercises/ex15_17_nested_limits.hpp>

using namespace p4mfe::ch10::ex;

TEST(Ch10Ex15, NestedRadical) {
  EXPECT_DOUBLE_EQ(Ex15Limit(2.0), 2.0);
  EXPECT_DOUBLE_EQ(Ex15Limit(6.0), 3.0);
  EXPECT_NEAR(Ex15Limit(1.0), std::numbers::phi, 1e-15);
  for (double a : {0.01, 0.5, 1.0, 2.0, 6.0, 100.0}) {
    const double limit = Ex15Limit(a);
    EXPECT_NEAR(limit * limit, limit + a, 1e-12 * limit * limit);
    double prev = 0.0;
    for (unsigned n = 1; n <= 10; ++n) {
      const double x = Ex15Iterate(a, n);
      EXPECT_GT(x, prev);   // increasing
      EXPECT_LT(x, limit);  // bounded by L
      prev = x;
    }
    EXPECT_NEAR(Ex15Iterate(a, 200), limit, 1e-13 * limit) << a;
  }
  EXPECT_THROW((void)Ex15Limit(0.0), std::domain_error);
}

TEST(Ch10Ex16, ContinuedFraction) {
  EXPECT_NEAR(Ex16Limit(1.0), std::numbers::phi, 1e-15);
  EXPECT_NEAR(Ex16Limit(2.0), 1.0 + std::numbers::sqrt2, 1e-15);
  for (double a : {0.1, 0.5, 1.0, 2.0, 10.0}) {
    const double limit = Ex16Limit(a);
    EXPECT_NEAR(limit, a + 1.0 / limit, 1e-14 * limit);
    for (unsigned n = 1; n <= 5;
         n += 2) {  // odd iterates below, even above (before round-off)
      EXPECT_LT(Ex16Iterate(a, n), limit) << a << ' ' << n;
      EXPECT_GT(Ex16Iterate(a, n + 1), limit) << a << ' ' << n;
    }
    // Contraction factor 1/(1 + a^2) per step; 2000 steps is plenty for a =
    // 0.1.
    EXPECT_NEAR(Ex16Iterate(a, 2000), limit, 1e-13 * limit) << a;
  }
}

TEST(Ch10Ex17, TowerEqualToTwo) {
  const double x = Ex17XFor(2.0);
  EXPECT_NEAR(x, std::numbers::sqrt2, 1e-15);
  EXPECT_NEAR(Ex17XFor(4.0), std::numbers::sqrt2,
              1e-15);  // same x, but the tower is 2
  double prev = 0.0;
  for (unsigned n = 1; n <= 30; ++n) {
    const double t = Ex17Tower(x, n);
    EXPECT_GT(t, prev);
    EXPECT_LT(t, 2.0);
    prev = t;
  }
  EXPECT_NEAR(Ex17Tower(x, 200), 2.0, 1e-13);  // rate ln 2 per step
}

TEST(Ch10Ex17, LargestXAndB) {
  const double e = std::numbers::e;
  EXPECT_NEAR(kEx17XMax, std::exp(1.0 / e), 1e-15);
  EXPECT_NEAR(Ex17XFor(e), kEx17XMax, 1e-15);
  // b^(1/b) is maximal at b = e.
  for (double b : {0.5, 1.0, 2.0, 2.7, 2.72, 3.0, 4.0, 10.0})
    EXPECT_LT(Ex17XFor(b), kEx17XMax) << b;
  // At x = e^(1/e) the tower increases to e, slowly: e - t_n ~ 2e/n.
  const double t = Ex17Tower(kEx17XMax, 100000);
  EXPECT_LT(t, e);
  EXPECT_GT(t, e - 1e-3);
  EXPECT_NEAR((e - t) * 100000.0 / (2.0 * e), 1.0, 0.05);
  // Just above e^(1/e) the tower diverges; just below it converges to b < e.
  EXPECT_EQ(Ex17Tower(kEx17XMax + 1e-3, 100000), HUGE_VAL);
  const double below = Ex17Tower(1.44, 100000);
  EXPECT_LT(below, e);
  EXPECT_NEAR(std::pow(1.44, below), below, 1e-12);
  // For x in (0, 1), b < 1 (e.g. x = 0.5 converges to 0.6412).
  const double half = Ex17Tower(0.5, 200);
  EXPECT_NEAR(std::pow(0.5, half), half, 1e-14);
  EXPECT_LT(half, 1.0);
}
