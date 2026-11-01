#include <p4mfe/ch08/exercises/ex09_10_heat_change_of_variables.hpp>
#include <p4mfe/ch08/exercises/ex11_barrier_rebate.hpp>
#include <p4mfe/ch08/exercises/ex12_13_asian_pdes.hpp>
#include <p4mfe/ch08/heat_equation.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace p4mfe;
using namespace p4mfe::ch08::ex;
using ch03::OptionType;

TEST(Ch08Ex09, InitialConditions) {
    const double K = 50.0, sigma = 0.3, r = 0.05, q = 0.02;
    const auto c = ex09_coefficients(sigma, r, q);
    for (double x = -1.0; x <= 1.0; x += 0.1) {
        EXPECT_NEAR(ex09_u(OptionType::Call, x, 0.0, K, sigma, r, q), ex09_call_u0(x, K, c.a), 1e-12);
        EXPECT_NEAR(ex09_u(OptionType::Put, x, 0.0, K, sigma, r, q), ex09_put_u0(x, K, c.a), 1e-12);
    }
    // The library's transform: a' = -a, b' = -b.
    const auto lib = ch08::heat_transform(sigma, r, q);
    EXPECT_NEAR(lib.a, -c.a, 1e-14);
    EXPECT_NEAR(lib.b, -c.b, 1e-12);
}

TEST(Ch08Ex09, HeatEquation) {
    const double K = 50.0;
    for (double q : {0.0, 0.03})
        for (OptionType type : {OptionType::Call, OptionType::Put})
            for (double x : {-0.3, 0.0, 0.2})
                for (double tau : {0.01, 0.05}) {
                    // u_xx is of order K / sqrt(tau) here; the residual is the
                    // O(h^2) central difference error, h = 1e-4.
                    EXPECT_NEAR(ex09_heat_residual(type, x, tau, K, 0.3, 0.05, q), 0.0, 1e-2)
                        << x << " " << tau;
                }
}

TEST(Ch08Ex10, SolutionOfTheSystem) {
    for (double q : {0.0, 0.01, 0.04})
        for (double sigma : {0.2, 0.4}) {
            const double r = 0.05;
            const auto s = ex10_solve(sigma, r, q);
            const auto res = ex10_residuals(s.a, s.b, sigma, r, q);
            EXPECT_NEAR(res.first, 0.0, 1e-13);
            EXPECT_NEAR(res.second, 0.0, 1e-12);
            const auto c = ex09_coefficients(sigma, r, q);
            EXPECT_NEAR(s.a, c.a, 1e-14);
            EXPECT_NEAR(s.b, c.b, 1e-12);
            // The printed b solves the system only for q = 0.
            const auto printed = ex10_residuals(s.a, ex10_printed_b(sigma, r, q), sigma, r, q);
            EXPECT_NEAR(printed.second, -2.0 * q / (sigma * sigma), 1e-12);
        }
}

TEST(Ch08Ex11, BoundaryCondition) {
    const double S = 40.0, K = 40.0, B = 35.0, T = 0.5, sigma = 0.3, r = 0.05, R = 2.0;
    const auto c = ex09_coefficients(sigma, r);
    // u at the barrier corresponds to V = R.
    for (double tau : {0.0, 0.01, 0.0225}) {
        const double u = ex11_u_boundary(tau, B, K, R, sigma, r);
        EXPECT_NEAR(std::exp(-c.a * std::log(B / K) - c.b * tau) * u, R, 1e-13);
    }
    EXPECT_DOUBLE_EQ(ex11_rebate_value(B, B, T, sigma, r), 1.0);
    // Heat equation with the derived boundary condition vs. closed form.
    for (double q : {0.0, 0.02})
        for (double rebate : {0.0, R}) {
            const double cf = ex11_down_and_out_call_with_rebate(S, K, B, rebate, T, sigma, r, q);
            EXPECT_NEAR(ex11_heat_fd(S, K, B, rebate, T, sigma, r, q, 400), cf, 2e-4) << q << " " << rebate;
        }
    EXPECT_NEAR(ex11_down_and_out_call_with_rebate(S, K, B, 0.0, T, sigma, r), 3.398883, 1e-6);
    EXPECT_NEAR(ex11_down_and_out_call_with_rebate(S, K, B, R, T, sigma, r), 4.439140, 1e-6);
}

TEST(Ch08Ex12, GeometricAsianPdes) {
    const AsianMarket m;
    const auto F = [&](double y, double t) { return ex12_geometric_asian_call(y, t, m); };
    for (double y : {std::log(45.0), std::log(50.0), std::log(56.0)})
        for (double t : {0.2, 0.6})
            EXPECT_NEAR(ex12_F_residual(F, y, t, m.T, m.sigma, m.r), 0.0, 1e-5) << y << " " << t;
    const auto V = [&](double S, double I, double t) { return F(ex12_y(S, I, t, m.T), t); };
    for (double S : {44.0, 50.0, 57.0}) {
        const double t = 0.3, I = t * std::log(48.0);
        EXPECT_NEAR(ex12_V_residual(V, S, I, t, m.sigma, m.r, 1e-5), 0.0, 2e-5) << S;
    }
    // At t = 0, I = 0 the price is the classic geometric Asian call:
    // Black-Scholes with volatility sigma/sqrt(3) and yield (r + sigma^2/6)/2.
    const double S = 50.0;
    const double bs = ch03::bs_call(S, m.K, m.T, m.sigma / std::sqrt(3.0), m.r,
                                    0.5 * (m.r + m.sigma * m.sigma / 6.0));
    EXPECT_NEAR(V(S, 0.0, 0.0), bs, 1e-12);
}

TEST(Ch08Ex12, OperatorIdentity) {
    // For any smooth F, residual_(8.52)(F(y(S, I, t), t)) = residual_F(F).
    const double T = 1.0, sigma = 0.25, r = 0.03;
    const auto F = [](double y, double t) { return std::sin(y) * std::exp(t) + y * y * y / 10.0; };
    const auto V = [&](double S, double I, double t) { return F(ex12_y(S, I, t, T), t); };
    for (double S : {30.0, 60.0})
        for (double t : {0.25, 0.7}) {
            const double I = 0.4;
            EXPECT_NEAR(ex12_V_residual(V, S, I, t, sigma, r), ex12_F_residual(F, ex12_y(S, I, t, T), t, T, sigma, r),
                        1e-5);
        }
}

TEST(Ch08Ex13, ArithmeticAsianPdes) {
    const double T = 1.0, sigma = 0.3, r = 0.05;
    const auto H = [&](double R, double t) { return ex13_H_linear(R, t, T, r); };
    const auto V = [&](double S, double I, double t) { return S * H(I / S, t); };
    for (double R : {0.0, 0.3, 1.2}) EXPECT_NEAR(ex13_H_residual(H, R, 0.4, sigma, r), 0.0, 1e-7);
    for (double S : {40.0, 55.0}) EXPECT_NEAR(ex13_V_residual(V, S, 12.0, 0.4, sigma, r), 0.0, 1e-5 * S);
    // Generic smooth H: residual_(8.53)(S H(I/S, t)) = S * residual_(8.55)(H).
    const auto G = [](double R, double t) { return std::exp(-R * R) * std::cos(t) + R * R * R; };
    const auto W = [&](double S, double I, double t) { return S * G(I / S, t); };
    for (double S : {20.0, 45.0})
        for (double I : {5.0, 30.0}) {
            const double t = 0.5;
            EXPECT_NEAR(ex13_V_residual(W, S, I, t, sigma, r), S * ex13_H_residual(G, I / S, t, sigma, r),
                        1e-4 * S * (1.0 + std::pow(I / S, 3)))
                << S << " " << I;
        }
}
