#include <p4mfe/ch07/finite_differences.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace p4mfe::ch07;

namespace {
const auto f = [](double x) { return std::exp(-x * x / 2) * std::sin(x); };
// f'(x) and f''(x) by hand.
double df(double x) { return std::exp(-x * x / 2) * (std::cos(x) - x * std::sin(x)); }
double d2f(double x) {
    return std::exp(-x * x / 2) * ((x * x - 2) * std::sin(x) - 2 * x * std::cos(x));
}
}  // namespace

// ---------------------------------------------------------------- 7.1 -----
TEST(Ch07FiniteDifferences, Schemes) {
    const double a = 0.7, h = 1e-3;
    EXPECT_NEAR(forward_difference(f, a, h), df(a), 1e-3);
    EXPECT_NEAR(backward_difference(f, a, h), df(a), 1e-3);
    EXPECT_NEAR(central_difference(f, a, h), df(a), 1e-6);
    EXPECT_NEAR(second_central_difference(f, a, h), d2f(a), 1e-5);
    // Forward and backward errors are +-h/2 f''(a) to leading order, so their
    // average is the central difference.
    EXPECT_NEAR(forward_difference(f, a, h) - df(a), 0.5 * h * d2f(a), 1e-6);
    EXPECT_NEAR(backward_difference(f, a, h) - df(a), -0.5 * h * d2f(a), 1e-6);
    EXPECT_NEAR(0.5 * (forward_difference(f, a, h) + backward_difference(f, a, h)), central_difference(f, a, h),
                1e-12);
    EXPECT_EQ(first_difference(f, a, h, FdScheme::Backward), backward_difference(f, a, h));
    EXPECT_THROW((void)central_difference(f, a, 0.0), std::invalid_argument);
}

TEST(Ch07FiniteDifferences, ExactForLowDegreePolynomials) {
    // One-sided differences are exact for linear functions, central ones for
    // quadratics (first derivative) and cubics (second derivative).
    const auto lin = [](double x) { return 3 * x - 1; };
    const auto quad = [](double x) { return x * x - 3 * x; };
    const auto cubic = [](double x) { return x * x * x; };
    EXPECT_NEAR(forward_difference(lin, 2.0, 0.5), 3.0, 1e-12);
    EXPECT_NEAR(central_difference(quad, 2.0, 0.5), 1.0, 1e-12);
    EXPECT_NEAR(second_central_difference(cubic, 2.0, 0.5), 12.0, 1e-12);
}

TEST(Ch07FiniteDifferences, ObservedOrder) {
    const double a = 0.3;
    EXPECT_EQ(truncation_order(FdScheme::Forward), 1);
    EXPECT_EQ(truncation_order(FdScheme::Central), 2);
    EXPECT_NEAR(fd_observed_order(f, a, df(a), FdScheme::Forward, 1e-2), 1.0, 0.05);
    EXPECT_NEAR(fd_observed_order(f, a, df(a), FdScheme::Backward, 1e-2), 1.0, 0.05);
    EXPECT_NEAR(fd_observed_order(f, a, df(a), FdScheme::Central, 1e-2), 2.0, 0.05);
    EXPECT_NEAR(fd_observed_order(f, a, d2f(a), FdScheme::Central, 1e-2, 2), 2.0, 0.05);
}

// -------------------------------------------------------------- 7.3.1 -----
TEST(Ch07FiniteDifferences, RoundOffAndOptimalStep) {
    const double a = 1.0;
    const auto rows = fd_error_table(f, a, df(a), FdScheme::Central, 0.1, 12);
    ASSERT_EQ(rows.size(), 12u);
    EXPECT_DOUBLE_EQ(rows[3].h, 1e-4);
    // Error falls ~100x per row while truncation dominates...
    EXPECT_NEAR(rows[0].error / rows[1].error, 100.0, 5.0);
    // ...then round-off wins: h = 1e-11 is far worse than h = 1e-5.
    EXPECT_GT(std::abs(rows[10].error), 100 * std::abs(rows[4].error));
    // The optimal steps.
    EXPECT_NEAR(optimal_step(FdScheme::Forward), 1.49e-8, 1e-10);
    EXPECT_NEAR(optimal_step(FdScheme::Central), 6.06e-6, 1e-8);
    EXPECT_NEAR(optimal_step(FdScheme::Central, 2), 1.22e-4, 1e-6);
    EXPECT_DOUBLE_EQ(optimal_step(FdScheme::Central, 1, -50.0), 50 * optimal_step(FdScheme::Central));
    // Errors at the optimal steps are near the best achievable.
    for (FdScheme s : {FdScheme::Forward, FdScheme::Central}) {
        const double h = optimal_step(s, 1, a);
        EXPECT_LT(std::abs(first_difference(f, a, h, s) - df(a)), s == FdScheme::Central ? 1e-10 : 1e-7);
    }
    EXPECT_LT(std::abs(second_central_difference(f, a, optimal_step(FdScheme::Central, 2, a)) - d2f(a)), 1e-7);
    // Second differences are more fragile: at h = 1e-7 round-off is ~eps/h^2.
    EXPECT_GT(std::abs(second_central_difference(f, a, 1e-7) - d2f(a)), 1e3 * std::abs(central_difference(f, a, 1e-7) - df(a)));
}
