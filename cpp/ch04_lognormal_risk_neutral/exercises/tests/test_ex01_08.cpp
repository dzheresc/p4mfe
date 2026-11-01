#include <p4mfe/ch04/exercises/ex01_02_normal_averages.hpp>
#include <p4mfe/ch04/exercises/ex03_06_binomial_calibration.hpp>
#include <p4mfe/ch04/exercises/ex07_08_confidence_roulette.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace p4mfe::ch04;
using namespace p4mfe::ch04::ex;

TEST(Ch04Ex01, SumOfIndependentStandardNormals) {
    const auto s = ex01_sum_independent();
    EXPECT_DOUBLE_EQ(s.mean, 0.0);
    EXPECT_DOUBLE_EQ(s.variance, 2.0);
    // Sample check: the sum of two independent draws has variance 2.
    std::mt19937_64 gen(42);
    std::normal_distribution<double> Z;
    const int n = 200000;
    double s1 = 0, s2 = 0;
    for (int i = 0; i < n; ++i) {
        const double x = Z(gen) + Z(gen);
        s1 += x;
        s2 += x * x;
    }
    EXPECT_NEAR(s1 / n, 0.0, 0.02);
    EXPECT_NEAR(s2 / n, 2.0, 0.03);
    // As written (X2 = -X1), the sum is identically zero.
    for (double z : {-2.5, -0.1, 0.0, 1.7}) EXPECT_EQ(ex01_sum_as_written(z), 0.0);
}

TEST(Ch04Ex02, AverageOfUncorrelated) {
    for (std::size_t n : {1u, 4u, 25u, 100u}) {
        const auto a = ex02_average(3.0, 2.0, n);
        EXPECT_NEAR(a.mean, 3.0, 1e-12);
        EXPECT_NEAR(a.variance, 4.0 / static_cast<double>(n), 1e-12);
    }
}

TEST(Ch04Ex03, BinomialMoments) {
    const double S = 40, u = 1.1, d = 0.92, p = 0.55;
    EXPECT_NEAR(ex03_moment(1, S, u, d, p), (p * u + (1 - p) * d) * S, 1e-12);
    EXPECT_NEAR(ex03_moment(2, S, u, d, p), (p * u * u + (1 - p) * d * d) * S * S, 1e-10);
}

TEST(Ch04Ex04, LognormalMomentsByIntegration) {
    const double S = 40, r = 0.05, sigma = 0.3, dt = 0.25;
    EXPECT_NEAR(risk_neutral_expectation([](double x) { return x; }, S, dt, sigma, r), ex04_first_moment(S, r, dt),
                1e-7);
    EXPECT_NEAR(risk_neutral_expectation([](double x) { return x * x; }, S, dt, sigma, r),
                ex04_second_moment(S, r, sigma, dt), 1e-5);
}

TEST(Ch04Ex05, CoxRossRubinstein) {
    for (double dt : {1.0 / 252, 1.0 / 12, 0.5}) {
        const double r = 0.05, sigma = 0.25;
        const auto s = ex05_crr(r, sigma, dt);
        EXPECT_NEAR(s.u * s.d, 1.0, 1e-12);
        EXPECT_GT(s.u, 1.0);
        EXPECT_GT(s.p, 0.0);
        EXPECT_LT(s.p, 1.0);
        const auto res = moment_residuals(s, r, sigma, dt);
        EXPECT_NEAR(res.first, 0.0, 1e-12);
        EXPECT_NEAR(res.second, 0.0, 1e-12);
        EXPECT_NEAR(ex05_residual_460(s.u, s.d, r, sigma, dt), 0.0, 1e-12);
        // Small dt: u ~ e^(sigma sqrt(dt)).
        if (dt < 0.01) {
            EXPECT_NEAR(std::log(s.u), sigma * std::sqrt(dt), 1e-4);
        }
    }
    // (i): any (u, d) satisfying (4.60), with p from (4.59), satisfies (4.57)-(4.58).
    const double r = 0.03, sigma = 0.2, dt = 0.1, g = std::exp(r * dt);
    const double d = 0.9, rhs = g * g * std::expm1(sigma * sigma * dt);
    const double u = g + rhs / (g - d);
    const BinomialStep s{u, d, (g - d) / (u - d)};
    const auto res = moment_residuals(s, r, sigma, dt);
    EXPECT_NEAR(res.first, 0.0, 1e-12);
    EXPECT_NEAR(res.second, 0.0, 1e-12);
}

TEST(Ch04Ex06, EqualProbabilityTree) {
    const double r = 0.04, sigma = 0.3, dt = 1.0 / 12;
    const auto s = ex06_equal_probability(r, sigma, dt);
    EXPECT_EQ(s.p, 0.5);
    const auto res = moment_residuals(s, r, sigma, dt);
    EXPECT_NEAR(res.first, 0.0, 1e-12);
    EXPECT_NEAR(res.second, 0.0, 1e-12);
    EXPECT_THROW((void)ex06_equal_probability(r, 1.0, 1.0), std::domain_error);
}

TEST(Ch04Ex07, ConfidenceIntervals) {
    const auto a = ex07_interval(0.95, 1.0 / 24, false);
    EXPECT_NEAR(a.lo, 44.3727, 1e-4);
    EXPECT_NEAR(a.hi, 56.4114, 1e-4);
    const auto b = ex07_interval(0.99, 1.0, false);
    EXPECT_NEAR(b.lo, 23.4360, 1e-4);
    EXPECT_NEAR(b.hi, 109.9221, 1e-4);
    const auto c = ex07_interval(0.95, 0.5, true);
    EXPECT_NEAR(c.lo, 32.7451, 1e-4);
    EXPECT_NEAR(c.hi, 75.2107, 1e-4);
    const auto d = ex07_interval(0.99, 2.0 / 12, true);
    EXPECT_NEAR(d.lo, 36.3811, 1e-4);
    EXPECT_NEAR(d.hi, 68.3743, 1e-4);
    // Probability inside the interval is the confidence level.
    const Ex07Asset A;
    for (double T : ex07_horizons) {
        for (bool rn : {false, true}) {
            const auto i = ex07_interval(0.99, T, rn);
            const double mu = rn ? A.r : A.mu;
            EXPECT_NEAR(price_cdf(i.hi, A.S0, T, A.sigma, mu, A.q) - price_cdf(i.lo, A.S0, T, A.sigma, mu, A.q), 0.99,
                        1e-9);
            // Risk-neutral interval = real-world one scaled by e^(-(mu - r) T).
            if (rn) {
                EXPECT_NEAR(i.lo, ex07_interval(0.99, T, false).lo * std::exp(-(A.mu - A.r) * T), 1e-9);
            }
        }
    }
}

TEST(Ch04Ex08, Roulette) {
    const auto e = ex08_exact();
    EXPECT_NEAR(e.win, 0.076689, 1e-6);
    EXPECT_NEAR(e.lose, 0.354794, 1e-6);
    const auto n = ex08_normal();
    EXPECT_NEAR(n.win, 0.063202, 1e-6);
    EXPECT_NEAR(n.lose, 0.317628, 1e-6);
    const auto c = ex08_normal(true);
    EXPECT_NEAR(c.win, 0.076603, 1e-6);
    EXPECT_NEAR(c.lose, 0.354127, 1e-6);
    // pmf sums to one.
    double total = 0;
    for (int k = 0; k <= 100; ++k) total += binomial_pmf(k, 100, ex08_p_black);
    EXPECT_NEAR(total, 1.0, 1e-12);
}
