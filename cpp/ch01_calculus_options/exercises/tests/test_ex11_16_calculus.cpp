#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/exercises/ex11_delta_integral.hpp>
#include <p4mfe/ch01/exercises/ex12_13_integral_limits.hpp>
#include <p4mfe/ch01/exercises/ex14_bond_yield_derivatives.hpp>
#include <p4mfe/ch01/exercises/ex15_quadratic_taylor.hpp>
#include <p4mfe/ch01/exercises/ex16_heat_equation.hpp>
#include <p4mfe/ch01/multivariable.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

using namespace p4mfe::ch01;
using namespace p4mfe::ch01::ex;

TEST(Ch01Ex11, DerivativeOfIntegral) {
    const Ex11Params p{100.0, 0.5, 0.3, 0.04};
    const auto g_by_integral = [&](double x) {
        return riemann_sum([](double y) { return std::exp(-0.5 * y * y); }, 0.0, ex11_b(p, x), 4000) /
               std::sqrt(2.0 * std::numbers::pi);
    };
    for (double x : {60.0, 90.0, 100.0, 120.0, 180.0}) {
        EXPECT_NEAR(g_by_integral(x), ex11_g(p, x), 1e-7);
        EXPECT_NEAR(derivative([&](double s) { return ex11_g(p, s); }, x), ex11_g_prime(p, x), 1e-8);
        // Leibniz rule from section 1.3 (integrand independent of x).
        EXPECT_NEAR(integral_derivative([](double y, double) { return std::exp(-0.5 * y * y); },
                                        [](double) { return 0.0; }, [&](double s) { return ex11_b(p, s); }, x) /
                        std::sqrt(2.0 * std::numbers::pi),
                    ex11_g_prime(p, x), 1e-8);
    }
}

TEST(Ch01Ex12, LocalAverage) {
    const auto f = [](double x) { return std::cos(x) + x * x * x; };
    const double a = 0.7;
    double prev_err = 0.0;
    for (double h : {0.4, 0.2, 0.1, 0.05}) {
        const double err = std::abs(ex12_local_average(f, a, h, 2000) - f(a));
        if (prev_err > 0.0) {
            EXPECT_NEAR(prev_err / err, 4.0, 0.2);  // O(h^2)
        }
        prev_err = err;
    }
    // Merely continuous f (a kink at a) still converges.
    const auto g = [](double x) { return std::abs(x - 1.0); };
    EXPECT_NEAR(ex12_local_average(g, 1.0, 1e-4, 2000), 0.0, 1e-4);
}

TEST(Ch01Ex13, GaussianToDelta) {
    // g = cos: integral phi(z) cos(mu + sigma z) dz = cos(mu) e^(-sigma^2/2) exactly.
    const double mu = 0.4;
    for (double sigma : {1.0, 0.1, 0.01}) {
        EXPECT_NEAR(ex13_gaussian_average([](double x) { return std::cos(x); }, mu, sigma),
                    std::cos(mu) * std::exp(-0.5 * sigma * sigma), 1e-10);
    }
    // Bounded but not smooth: g = sign-like clamp.
    const auto g = [](double x) { return std::tanh(5.0 * x) + 1.0 / (1.0 + x * x); };
    double prev = 1e9;
    for (double sigma : {0.5, 0.1, 0.02, 0.004}) {
        const double err = std::abs(ex13_gaussian_average(g, 0.3, sigma) - g(0.3));
        EXPECT_LT(err, prev);
        prev = err;
    }
    EXPECT_LT(prev, 1e-3);
}

TEST(Ch01Ex14, BondPriceDerivatives) {
    // 3-year 6% semiannual bond.
    std::vector<CashFlow> cfs;
    for (int i = 1; i <= 6; ++i) cfs.push_back({0.5 * i, i == 6 ? 103.0 : 3.0});
    const double y = 0.055;
    const auto f = [&](double s) { return ex14_continuous(cfs, s).value; };
    const auto d = ex14_continuous(cfs, y);
    EXPECT_NEAR(derivative(f, y), d.first, 1e-6);
    EXPECT_NEAR(second_derivative(f, y), d.second, 1e-3);
    EXPECT_LT(d.first, 0.0);
    EXPECT_GT(d.second, 0.0);
    for (double m : {1.0, 2.0, 12.0}) {
        const auto g = [&](double s) { return ex14_discrete(cfs, s, m).value; };
        const auto e = ex14_discrete(cfs, y, m);
        EXPECT_NEAR(derivative(g, y), e.first, 1e-6);
        EXPECT_NEAR(second_derivative(g, y), e.second, 1e-3);
    }
    // m -> infinity recovers continuous compounding.
    const auto big = ex14_discrete(cfs, y, 1e7);
    EXPECT_NEAR(big.value, d.value, 1e-4);
    EXPECT_NEAR(big.first, d.first, 1e-4);
    EXPECT_NEAR(big.second, d.second, 1e-3);
}

TEST(Ch01Ex15, GradientHessianTaylor) {
    const auto g = ex15_gradient(ex15_a);
    EXPECT_EQ(g, (Vec3{5.0, -1.0, -3.0}));
    EXPECT_DOUBLE_EQ(ex15_f(ex15_a), 3.0);
    // Agrees with the numerical gradient and Hessian of section 1.6.
    const auto fv = [](const Vector& x) { return ex15_f({x[0], x[1], x[2]}); };
    const Vector a{1.0, -1.0, 0.0};
    const auto ng = gradient(fv, a);
    const auto nh = hessian(fv, a);
    const auto H = ex15_hessian();
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(ng[i], g[i], 1e-8);
        for (std::size_t j = 0; j < 3; ++j) EXPECT_NEAR(nh[i][j], H[i][j], 1e-5);
    }
    // The quadratic Taylor expansion is exact.
    for (const Vec3& x : {Vec3{0, 0, 0}, Vec3{2, 3, -1}, Vec3{-1.5, 0.25, 4}, Vec3{10, -7, 3}})
        EXPECT_NEAR(ex15_taylor2(x), ex15_f(x), 1e-12 * std::max(1.0, std::abs(ex15_f(x))));
    static_assert(ex15_taylor2({2, 3, -1}) == ex15_f({2, 3, -1}));
}

TEST(Ch01Ex16, HeatEquation) {
    for (double t : {0.1, 0.5, 2.0}) {
        for (double x : {-2.0, -0.3, 0.0, 0.8, 3.0}) {
            const double ut = derivative([&](double s) { return ex16_u(x, s); }, t);
            const double uxx = second_derivative([&](double s) { return ex16_u(s, t); }, x);
            EXPECT_NEAR(ut, ex16_u_t(x, t), 1e-7);
            EXPECT_NEAR(uxx, ex16_u_xx(x, t), 1e-5);
            EXPECT_NEAR(derivative([&](double s) { return ex16_u(s, t); }, x), ex16_u_x(x, t), 1e-8);
            EXPECT_DOUBLE_EQ(ex16_u_t(x, t), ex16_u_xx(x, t));
        }
        // u(., t) is the N(0, 2t) density: unit mass.
        EXPECT_NEAR(riemann_sum([&](double s) { return ex16_u(s, t); }, -40.0, 40.0, 20000), 1.0, 1e-10);
    }
}
