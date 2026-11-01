#include <p4mfe/ch07/exercises/ex12_gamma_theta.hpp>
#include <p4mfe/ch07/exercises/ex13_14_bs_pde_solutions.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace p4mfe;
using namespace p4mfe::ch07;
using namespace p4mfe::ch07::ex;

TEST(Ch07Ex12, ExactWhenRatesAreZero) {
    for (double S : {30.0, 50.0, 80.0})
        for (double tau : {0.1, 1.0, 3.0}) {
            const Ex12Market m{S, 52.0, tau, 0.3, 0.0};
            EXPECT_NEAR(ex12_lhs(m), 0.0, 1e-12) << S << " " << tau;
            EXPECT_NEAR(ex12_lhs(m, ch03::OptionType::Put), 0.0, 1e-12) << S << " " << tau;
        }
}

TEST(Ch07Ex12, FormulaWhenRateIsNonzero) {
    for (double r : {-0.01, 0.02, 0.05, 0.2})
        for (double S : {30.0, 50.0, 80.0}) {
            const Ex12Market m{S, 52.0, 0.75, 0.3, r};
            EXPECT_NEAR(ex12_lhs(m), ex12_rhs(m), 1e-12) << r << " " << S;
            if (r > 0) {
                EXPECT_GT(ex12_lhs(m), 0.0);
                EXPECT_LT(ex12_lhs(m), 1.0);
            }
        }
    // -> 0 as r -> 0
    EXPECT_LT(std::abs(ex12_lhs({50.0, 52.0, 0.75, 0.3, 1e-6})), 1e-4);
}

TEST(Ch07Ex13, CallSolvesThePde) {
    const PdeMarket m;
    for (double S : {20.0, 40.0, 60.0})
        for (double t : {0.0, 0.5, 0.9}) {
            EXPECT_NEAR(ex13_residual_greeks(S, t, m), 0.0, 1e-12) << S << " " << t;
            EXPECT_NEAR(ex13_residual_fd(S, t, m), 0.0, 1e-5) << S << " " << t;
        }
    // A non-solution for contrast: the call price with the wrong volatility in the PDE.
    EXPECT_GT(std::abs(bs_pde_residual_fd([&](double s, double u) { return ex13_call(s, u, m); }, 40.0, 0.5, 0.4,
                                          m.r, m.q)),
              0.1);
}

TEST(Ch07Ex14, ForwardSolvesThePde) {
    for (double sigma : {0.1, 0.3, 0.8}) {
        const PdeMarket m{40.0, 1.0, sigma, 0.05, 0.02};
        for (double S : {20.0, 40.0, 60.0})
            for (double t : {0.0, 0.5, 0.9}) {
                EXPECT_NEAR(ex14_residual_exact(S, t, m), 0.0, 1e-12) << S << " " << t;
                EXPECT_NEAR(ex14_residual_fd(S, t, m), 0.0, 1e-6) << S << " " << t;
            }
    }
    // f = C - P (put-call parity), so the put solves the PDE too.
    const PdeMarket m;
    const double S = 45.0, t = 0.3;
    const double put = ch03::bs_put(S, m.K, m.T - t, m.sigma, m.r, m.q);
    EXPECT_NEAR(ex13_call(S, t, m) - put, ex14_forward(S, t, m), 1e-12);
}
