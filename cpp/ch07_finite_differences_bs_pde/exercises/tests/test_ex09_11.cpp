#include <p4mfe/ch07/exercises/ex09_euler_convergence.hpp>
#include <p4mfe/ch07/exercises/ex10_bvp_linear_system.hpp>
#include <p4mfe/ch07/exercises/ex11_first_order_system.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace p4mfe;
using namespace p4mfe::ch07;
using namespace p4mfe::ch07::ex;

TEST(Ch07Ex09, ClosedFormOfEuler) {
    for (std::size_t n : {1u, 5u, 64u}) {
        const auto s = ex09_solve(n);
        for (std::size_t i = 0; i <= n; ++i) EXPECT_NEAR(s.y[i], ex09_closed_form(i, n), 1e-13 * s.y[i]);
    }
}

TEST(Ch07Ex09, FirstOrderConvergence) {
    const double e = std::numbers::e;
    for (std::size_t n : {10u, 100u, 1000u}) {
        const auto s = ex09_solve(n);
        const double err = ex09_max_error(n);
        // The largest error is at x = 1 and is below the bound e/(2n).
        EXPECT_NEAR(err, e - s.y[n], 1e-14);
        EXPECT_LT(err, ex09_bound(n));
        // n max|e_i| -> e/2, with the O(1/n) correction -11e/(24n).
        const double dn = static_cast<double>(n);
        EXPECT_NEAR(dn * err, e / 2.0 - 11.0 * e / (24.0 * dn), 2.0 / (dn * dn)) << n;
        for (std::size_t i = 0; i <= n; ++i) EXPECT_LE(s.y[i], std::exp(s.x[i]) + 1e-15);  // e_i <= 0
    }
    EXPECT_NEAR(std::log2(ex09_max_error(100) / ex09_max_error(200)), 1.0, 0.01);
}

TEST(Ch07Ex10, LinearSystemForSixIntervals) {
    const auto s = ex10_system(6);
    const double lower[] = {0.0, 13.0, 28.5, 50.0, 77.5};
    const double diag[] = {-5.0, -23.0, -53.0, -95.0, -149.0};
    const double upper[] = {2.5, 11.0, 25.5, 46.0, 0.0};
    const double rhs[] = {-3.5, 0.0, 0.0, 0.0, -36.25};
    ASSERT_EQ(s.diag.size(), 5u);
    for (std::size_t k = 0; k < 5; ++k) {
        EXPECT_DOUBLE_EQ(s.lower[k], lower[k]);
        EXPECT_DOUBLE_EQ(s.diag[k], diag[k]);
        EXPECT_DOUBLE_EQ(s.upper[k], upper[k]);
        EXPECT_DOUBLE_EQ(s.rhs[k], rhs[k]);
    }
    const auto sol = ex10_solve(6);
    const double Y[] = {1.385311, 1.370622, 1.228661, 1.021815, 0.774770};
    for (std::size_t i = 1; i <= 5; ++i) EXPECT_NEAR(sol.y[i], Y[i - 1], 5e-7);
    // A Y = b.
    for (std::size_t k = 0; k < 5; ++k) {
        double r = diag[k] * sol.y[k + 1];
        if (k > 0) r += lower[k] * sol.y[k];
        if (k < 4) r += upper[k] * sol.y[k + 2];
        EXPECT_NEAR(r, rhs[k], 1e-12);
    }
}

TEST(Ch07Ex10, SameAsGenericBvpSolver) {
    // y'' + p y' + q y = 0 with p = -1/(3x), q = 1/(3x^2): the same equations scaled by 3x_i^2.
    for (std::size_t n : {6u, 20u}) {
        const auto a = ex10_solve(n);
        const auto b = solve_bvp([](double x) { return -1.0 / (3.0 * x); },
                                 [](double x) { return 1.0 / (3.0 * x * x); }, [](double) { return 0.0; }, 0.0, 1.0,
                                 1.0, 0.5, n);
        for (std::size_t i = 0; i <= n; ++i) EXPECT_NEAR(a.y[i], b.y[i], 1e-10);
    }
}

TEST(Ch07Ex10, NoConvergenceBecauseNoSolution) {
    // Every solution of the ODE is c1 x + c2 x^(1/3) (zero at x = 0) ...
    for (double m : {1.0, 1.0 / 3.0}) EXPECT_NEAR(3.0 * m * (m - 1.0) - m + 1.0, 0.0, 1e-15);
    // ... so y(0) = 1 cannot hold, and the discrete solutions keep growing.
    double prev = 0.0;
    for (std::size_t n : {6u, 12u, 24u, 48u, 96u}) {
        const double mid = ex10_solve(n).y[n / 2];
        EXPECT_GT(mid, prev + 0.2) << n;
        prev = mid;
    }
    EXPECT_GT(prev, 2.5);
}

TEST(Ch07Ex11, SystemForm) {
    // f(x, Y) = (y', 2y' - x^2 y).
    const auto d = ex11_f(2.0, {3.0, 5.0});
    EXPECT_DOUBLE_EQ(d[0], 5.0);
    EXPECT_DOUBLE_EQ(d[1], -4.0 * 3.0 + 2.0 * 5.0);
    // The first component of an accurate solution of the system solves the ODE.
    const auto ref = ex11_rk4({1.0, 0.0}, 0.0, 1.0, 1000);
    for (std::size_t k : {1u, 250u, 500u, 999u}) EXPECT_NEAR(ex11_ode_residual(ref, k), 0.0, 1e-4) << k;
    // and its second component is the derivative of the first.
    const double h = 1e-3;
    for (std::size_t k : {100u, 700u})
        EXPECT_NEAR((ref.Y[k + 1][0] - ref.Y[k - 1][0]) / (2 * h), ref.Y[k][1], 1e-5);
    // Forward Euler on the system is first order.
    const auto err = [&](std::size_t n) {
        const auto s = ex11_euler({1.0, 0.0}, 0.0, 1.0, n);
        return std::abs(s.Y[n][0] - ref.Y[1000][0]);
    };
    EXPECT_NEAR(std::log2(err(100) / err(200)), 1.0, 0.05);
}
