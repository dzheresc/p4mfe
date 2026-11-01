#include <p4mfe/ch10/exercises/ex05_07_linear_recursions.hpp>
#include <p4mfe/ch10/exercises/ex08_double_root.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

using namespace p4mfe::ch10;
using namespace p4mfe::ch10::ex;

TEST(Ch10Ex05, SequenceIsAlternatingSign) {
    const auto x = recursion_terms(ex05_recursion, ex05_init, 80);
    for (unsigned n = 0; n < x.size(); ++n) EXPECT_EQ(x[n], ex05_closed_form(n)) << n;
    // Constants of c1 (-1)^n + c2 phi^n + c3 psi^n are (1, 0, 0).
    const Vector c = ex05_constants();
    EXPECT_NEAR(c[0], 1.0, 1e-12);
    EXPECT_NEAR(c[1], 0.0, 1e-12);
    EXPECT_NEAR(c[2], 0.0, 1e-12);
    // The three roots of z^3 - 2z - 1.
    const double phi = 0.5 * (1.0 + std::sqrt(5.0)), psi = 0.5 * (1.0 - std::sqrt(5.0));
    for (double z : {-1.0, phi, psi}) EXPECT_NEAR(z * z * z - 2.0 * z - 1.0, 0.0, 1e-12) << z;
    // Other initial values excite phi^n: the general form still fits exactly.
    const auto basis = ex05_basis();
    const std::array<double, 3> init{2.0, 0.5, -1.0};
    const Vector c2 = ex_fit_constants(basis, {init.begin(), init.end()});
    const auto y = recursion_terms(ex05_recursion, init, 30);
    for (unsigned n = 0; n < y.size(); ++n)
        EXPECT_NEAR(ex_evaluate(basis, c2, n), y[n], 1e-9 * std::max(1.0, std::abs(y[n]))) << n;
}

TEST(Ch10Ex06, ClosedForm) {
    const std::array<double, 2> a{-3.0, 4.0}, init{1.0, 5.0};
    const auto x = recursion_terms(a, init, 30);
    const auto r = ex06_second_order();
    double direct = 1.0;
    for (unsigned n = 0; n < x.size(); ++n) {
        EXPECT_EQ(x[n], direct) << n;  // integers below 2^53
        EXPECT_EQ(ex06_closed_form(n), direct) << n;
        EXPECT_NEAR(r(n), direct, 1e-12 * direct) << n;
        EXPECT_NEAR(ex06_first_order(n), direct, 1e-12 * direct) << n;
        direct = 3.0 * direct + 2.0;
    }
    EXPECT_NEAR(std::min(r.lambda1.real(), r.lambda2.real()), 1.0, 1e-12);
    EXPECT_NEAR(std::max(r.lambda1.real(), r.lambda2.real()), 3.0, 1e-12);
}

TEST(Ch10Ex07, ClosedForm) {
    const auto direct = ex07_direct(30);
    EXPECT_EQ(direct[1], 5.0);
    EXPECT_EQ(direct[2], 18.0);
    const auto x = recursion_terms(ex07_recursion, ex07_init, 30);
    for (unsigned n = 0; n < x.size(); ++n) {
        EXPECT_EQ(x[n], direct[n]) << n;
        EXPECT_EQ(ex07_closed_form(n), direct[n]) << n;
    }
    const Vector c = ex07_constants();
    EXPECT_NEAR(c[0], -1.25, 1e-12);
    EXPECT_NEAR(c[1], -0.5, 1e-12);
    EXPECT_NEAR(c[2], 2.25, 1e-12);
    // (z - 1)^2 (z - 3) = z^3 - 5z^2 + 7z - 3.
    for (double z : {-1.0, 0.5, 2.0, 4.0})
        EXPECT_NEAR(z * z * z - ex07_recursion[2] * z * z - ex07_recursion[1] * z - ex07_recursion[0],
                    (z - 1.0) * (z - 1.0) * (z - 3.0), 1e-12);
}

TEST(Ch10Ex08, DoubleRootGivesNLambdaN) {
    // P(z) = (z - 2)^2 (z + 1) = z^3 - 3z^2 + 4: double root 2, simple root -1.
    const std::array<double, 4> a{4.0, 0.0, -3.0, 1.0};
    for (unsigned n = 0; n <= 20; ++n) {
        const double scale = std::pow(2.0, n + 3.0) * (n + 3.0);
        EXPECT_NEAR(ex08_residual(a, 2.0, 1.7, n), 0.0, 1e-13 * scale) << n;
        EXPECT_NEAR(ex08_hint(a, 2.0, 1.7, n), 0.0, 1e-13 * scale) << n;
        // At the simple root the residual is C lambda^(n+1) P'(lambda) != 0, as in the hint.
        const double r = ex08_residual(a, -1.0, 1.7, n);
        EXPECT_NEAR(r, ex08_hint(a, -1.0, 1.7, n), 1e-12) << n;
        EXPECT_NEAR(std::abs(r), 1.7 * 9.0, 1e-12) << n;  // |P'(-1)| = |3 + 6| = 9
    }
    // A generic (non-root) lambda: the identity of the hint holds for every lambda.
    const std::array<double, 4> b{0.5, -1.2, 0.3, 2.0};
    for (unsigned n = 0; n <= 10; ++n)
        EXPECT_NEAR(ex08_residual(b, 0.8, -2.0, n), ex08_hint(b, 0.8, -2.0, n), 1e-12) << n;
    // The sequence n 2^n satisfies x_{n+3} = 3 x_{n+2} - 4 x_n.
    const std::array<double, 3> rec{-4.0, 0.0, 3.0}, init{0.0, 2.0, 8.0};
    const auto x = recursion_terms(rec, init, 40);
    for (unsigned n = 0; n < x.size(); ++n) EXPECT_EQ(x[n], n * std::pow(2.0, n)) << n;
}
