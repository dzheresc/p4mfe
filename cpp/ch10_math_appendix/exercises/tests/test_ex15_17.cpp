#include <p4mfe/ch10/exercises/ex15_17_nested_limits.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace p4mfe::ch10::ex;

TEST(Ch10Ex15, NestedRadical) {
    EXPECT_DOUBLE_EQ(ex15_limit(2.0), 2.0);
    EXPECT_DOUBLE_EQ(ex15_limit(6.0), 3.0);
    EXPECT_NEAR(ex15_limit(1.0), std::numbers::phi, 1e-15);
    for (double a : {0.01, 0.5, 1.0, 2.0, 6.0, 100.0}) {
        const double L = ex15_limit(a);
        EXPECT_NEAR(L * L, L + a, 1e-12 * L * L);
        double prev = 0.0;
        for (unsigned n = 1; n <= 10; ++n) {
            const double x = ex15_iterate(a, n);
            EXPECT_GT(x, prev);  // increasing
            EXPECT_LT(x, L);     // bounded by L
            prev = x;
        }
        EXPECT_NEAR(ex15_iterate(a, 200), L, 1e-13 * L) << a;
    }
    EXPECT_THROW((void)ex15_limit(0.0), std::domain_error);
}

TEST(Ch10Ex16, ContinuedFraction) {
    EXPECT_NEAR(ex16_limit(1.0), std::numbers::phi, 1e-15);
    EXPECT_NEAR(ex16_limit(2.0), 1.0 + std::numbers::sqrt2, 1e-15);
    for (double a : {0.1, 0.5, 1.0, 2.0, 10.0}) {
        const double L = ex16_limit(a);
        EXPECT_NEAR(L, a + 1.0 / L, 1e-14 * L);
        for (unsigned n = 1; n <= 5; n += 2) {  // odd iterates below, even above (before round-off)
            EXPECT_LT(ex16_iterate(a, n), L) << a << ' ' << n;
            EXPECT_GT(ex16_iterate(a, n + 1), L) << a << ' ' << n;
        }
        // Contraction factor 1/(1 + a^2) per step; 2000 steps is plenty for a = 0.1.
        EXPECT_NEAR(ex16_iterate(a, 2000), L, 1e-13 * L) << a;
    }
}

TEST(Ch10Ex17, TowerEqualToTwo) {
    const double x = ex17_x_for(2.0);
    EXPECT_NEAR(x, std::numbers::sqrt2, 1e-15);
    EXPECT_NEAR(ex17_x_for(4.0), std::numbers::sqrt2, 1e-15);  // same x, but the tower is 2
    double prev = 0.0;
    for (unsigned n = 1; n <= 30; ++n) {
        const double t = ex17_tower(x, n);
        EXPECT_GT(t, prev);
        EXPECT_LT(t, 2.0);
        prev = t;
    }
    EXPECT_NEAR(ex17_tower(x, 200), 2.0, 1e-13);  // rate ln 2 per step
}

TEST(Ch10Ex17, LargestXAndB) {
    const double e = std::numbers::e;
    EXPECT_NEAR(ex17_x_max, std::exp(1.0 / e), 1e-15);
    EXPECT_NEAR(ex17_x_for(e), ex17_x_max, 1e-15);
    // b^(1/b) is maximal at b = e.
    for (double b : {0.5, 1.0, 2.0, 2.7, 2.72, 3.0, 4.0, 10.0}) EXPECT_LT(ex17_x_for(b), ex17_x_max) << b;
    // At x = e^(1/e) the tower increases to e, slowly: e - t_n ~ 2e/n.
    const double t = ex17_tower(ex17_x_max, 100000);
    EXPECT_LT(t, e);
    EXPECT_GT(t, e - 1e-3);
    EXPECT_NEAR((e - t) * 100000.0 / (2.0 * e), 1.0, 0.05);
    // Just above e^(1/e) the tower diverges; just below it converges to b < e.
    EXPECT_EQ(ex17_tower(ex17_x_max + 1e-3, 100000), HUGE_VAL);
    const double below = ex17_tower(1.44, 100000);
    EXPECT_LT(below, e);
    EXPECT_NEAR(std::pow(1.44, below), below, 1e-12);
    // For x in (0, 1), b < 1 (e.g. x = 0.5 converges to 0.6412).
    const double half = ex17_tower(0.5, 200);
    EXPECT_NEAR(std::pow(0.5, half), half, 1e-14);
    EXPECT_LT(half, 1.0);
}
