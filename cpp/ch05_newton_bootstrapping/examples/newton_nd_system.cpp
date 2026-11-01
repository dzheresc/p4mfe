// Section 5.2: Newton's method and the approximate Newton's method for
//   x1^3 + x2 x3 = 7,  x1^2 x2 + x3 = 5,  x1 + x2 + x3^2 = 12,
// which has the root (1, 2, 3) and another near (1.708, 0.649, 3.105).
// Prints ||F(x_k)|| per iteration for the exact Jacobian and for forward-
// difference Jacobians with several steps h, then shows that the starting
// point decides which root Newton's method finds.

#include <p4mfe/ch05/newton_nd.hpp>

#include <cstdio>

int main() {
    using namespace p4mfe::ch05;
    const auto F = [](const Vector& x) {
        return Vector{x[0] * x[0] * x[0] + x[1] * x[2] - 7.0, x[0] * x[0] * x[1] + x[2] - 5.0,
                      x[0] + x[1] + x[2] * x[2] - 12.0};
    };
    const auto J = [](const Vector& x) {
        return Matrix{{3 * x[0] * x[0], x[2], x[1]}, {2 * x[0] * x[1], x[0] * x[0], 1.0}, {1.0, 1.0, 2 * x[2]}};
    };
    const Vector x0{1.5, 1.5, 2.5};
    const Tolerances one_step{1e300, 1e300, 1};

    std::printf("%4s  %12s  %12s  %12s  %12s\n", "k", "exact J", "h = 1e-2", "h = 1e-4", "h = 1e-8");
    Vector xs[4] = {x0, x0, x0, x0};
    const double hs[3] = {1e-2, 1e-4, 1e-8};
    for (int k = 0; k <= 8; ++k) {
        std::printf("%4d", k);
        for (int m = 0; m < 4; ++m) std::printf("  %12.3e", norm2(F(xs[m])));
        std::printf("\n");
        xs[0] = newton_nd(F, J, xs[0], one_step).x;
        for (int m = 1; m < 4; ++m) xs[m] = approximate_newton_nd(F, xs[m], one_step, hs[m - 1]).x;
    }
    std::printf("\n");
    for (const Vector& start : {x0, Vector{1.3, 1.7, 2.6}}) {
        const auto res = newton_nd(F, J, start, {1e-12, 1e-12});
        std::printf("From (%.1f, %.1f, %.1f): x = (%.12f, %.12f, %.12f) after %zu iterations\n", start[0], start[1],
                    start[2], res.x[0], res.x[1], res.x[2], res.iterations);
    }
}
