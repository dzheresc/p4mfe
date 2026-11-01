#include <p4mfe/ch05/newton_nd.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

using namespace p4mfe::ch05;

namespace {
// A system with the root (1, 2, 3), where the Jacobian is nonsingular.
Vector F3(const Vector& x) {
    return {x[0] * x[0] * x[0] + x[1] * x[2] - 7.0, x[0] * x[0] * x[1] + x[2] - 5.0, x[0] + x[1] + x[2] * x[2] - 12.0};
}
Matrix J3(const Vector& x) {
    return {{3 * x[0] * x[0], x[2], x[1]}, {2 * x[0] * x[1], x[0] * x[0], 1.0}, {1.0, 1.0, 2 * x[2]}};
}
}  // namespace

TEST(Ch05NewtonNd, SolveLinear) {
    // Needs pivoting: A(0, 0) = 0.
    const Matrix A{{0, 2, 1}, {1, 1, 1}, {2, 1, 3}};
    const auto x = solve_linear(A, {7, 6, 13});
    EXPECT_NEAR(x[0], 1, 1e-14);
    EXPECT_NEAR(x[1], 2, 1e-14);
    EXPECT_NEAR(x[2], 3, 1e-14);
    EXPECT_THROW((void)solve_linear(Matrix{{1, 2}, {2, 4}}, {1, 2}), std::runtime_error);
}

TEST(Ch05NewtonNd, NewtonFindsRoot) {
    const auto res = newton_nd(F3, J3, {1.3, 1.7, 2.6}, {1e-12, 1e-12});
    EXPECT_NEAR(res.x[0], 1, 1e-12);
    EXPECT_NEAR(res.x[1], 2, 1e-12);
    EXPECT_NEAR(res.x[2], 3, 1e-12);
    EXPECT_LE(res.iterations, 8u);
}

TEST(Ch05NewtonNd, QuadraticConvergence) {
    // ||x_{k+1} - x*|| / ||x_k - x*||^2 stays bounded.
    const Vector root{1, 2, 3};
    Vector x{1.3, 1.7, 2.6};
    double e_prev = 0.0;
    for (int k = 0; k < 4; ++k) {
        const auto step = newton_nd(F3, J3, x, {1e300, 1e300, 1});  // one step
        x = step.x;
        double e = 0.0;
        for (int i = 0; i < 3; ++i) e += (x[i] - root[i]) * (x[i] - root[i]);
        e = std::sqrt(e);
        if (k > 0 && e > 1e-13) {
            EXPECT_LT(e / (e_prev * e_prev), 5.0);
        }
        e_prev = e;
    }
}

TEST(Ch05NewtonNd, FiniteDifferenceJacobian) {
    const Vector x{0.7, -1.2, 2.1};
    const Matrix J = J3(x);
    const Matrix Jf = finite_difference_jacobian(F3, x, 1e-6);
    const Matrix Jc = finite_difference_jacobian(F3, x, 1e-5, Difference::Central);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(Jf(i, j), J(i, j), 1e-4);  // O(h)
            EXPECT_NEAR(Jc(i, j), J(i, j), 1e-8);  // O(h^2)
        }
}

TEST(Ch05NewtonNd, ApproximateNewtonMatchesNewton) {
    const auto exact = newton_nd(F3, J3, {1.3, 1.7, 2.6}, {1e-12, 1e-12});
    const auto approx = approximate_newton_nd(F3, {1.3, 1.7, 2.6}, {1e-12, 1e-12});
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(approx.x[i], exact.x[i], 1e-12);
    EXPECT_GE(approx.iterations, exact.iterations);
}

TEST(Ch05NewtonNd, TwoDimensional) {
    // Circle x^2 + y^2 = 4 meets y = 1 - e^x at two points; start near the one with x < 0.
    const auto F = [](const Vector& v) { return Vector{v[0] * v[0] + v[1] * v[1] - 4, std::exp(v[0]) + v[1] - 1}; };
    const auto res = approximate_newton_nd(F, {-2.0, 1.0}, {1e-12, 1e-12});
    const auto f = F(res.x);
    EXPECT_LT(norm2(f), 1e-12);
    EXPECT_LT(res.x[0], 0.0);
}
