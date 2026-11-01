#include <p4mfe/ch10/big_o.hpp>
#include <p4mfe/ch10/exercises/ex09_10_big_o.hpp>
#include <p4mfe/ch10/exercises/ex11_binomial_identities.hpp>
#include <p4mfe/ch10/exercises/ex12_13_harmonic_series.hpp>
#include <p4mfe/ch10/exercises/ex14_power_series_radius.hpp>
#include <p4mfe/ch10/power_series.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

using namespace p4mfe::ch10;
using namespace p4mfe::ch10::ex;

TEST(Ch10Ex09, SumsOfBigOAndLittleO) {
    std::vector<double> xs;
    for (double x = 0.1; x > 1e-6; x *= 0.37) xs.push_back(x);
    const auto x2 = [](double x) { return x * x; };
    const auto big_sum = [](double x) { return ex09_f_big(x) + ex09_g_big(x); };
    for (double r : order_ratios(ex09_f_big, x2, xs)) EXPECT_LE(r, 1.0 + 1e-12);
    for (double r : order_ratios(ex09_g_big, x2, xs)) EXPECT_NEAR(r, 3.0, 1e-12);
    for (double r : order_ratios(big_sum, x2, xs)) EXPECT_LE(r, 4.0 + 1e-12);  // C_f + C_g
    // Not o(x^2): the ratio of the sum stays >= 2.
    for (double r : order_ratios(big_sum, x2, xs)) EXPECT_GE(r, 2.0 - 1e-12);
    const auto small_sum = [](double x) { return ex09_f_small(x) + ex09_g_small(x); };
    const auto ratios = order_ratios(small_sum, x2, xs);
    for (std::size_t i = 1; i < ratios.size(); ++i) EXPECT_LT(ratios[i], ratios[i - 1]);
    EXPECT_LT(ratios.back(), 2e-3);
    // f + g = o(x^2) at the slower of the two rates, x^(1/2).
    EXPECT_NEAR(observed_power(small_sum, 1e-6), 2.5, 0.01);
}

TEST(Ch10Ex10, OrdersOfSumsOfSquaresAndCubes) {
    for (std::uint64_t n = 1; n <= 10000; n = n < 20 ? n + 1 : n * 3 / 2) {
        const double N = static_cast<double>(n);
        EXPECT_NEAR(ex10_ratio_k2(n), 1.0 / 3.0 + 1.0 / (2.0 * N) + 1.0 / (6.0 * N * N), 1e-13) << n;
        EXPECT_NEAR(ex10_remainder_ratio_k2(n), 0.5 + 1.0 / (6.0 * N), 1e-13) << n;
        EXPECT_NEAR(ex10_ratio_k3(n), 0.25 + 1.0 / (2.0 * N) + 1.0 / (4.0 * N * N), 1e-13) << n;
        EXPECT_NEAR(ex10_remainder_ratio_k3(n), 0.5 + 1.0 / (4.0 * N), 1e-13) << n;
        EXPECT_LE(ex10_ratio_k2(n), 1.0);
        EXPECT_LE(ex10_remainder_ratio_k2(n), 2.0 / 3.0);
        EXPECT_LE(ex10_ratio_k3(n), 1.0);
        EXPECT_LE(ex10_remainder_ratio_k3(n), 0.75);
    }
    // Observed growth exponents 3, 2, 4, 3 (n -> 2n at n = 1000).
    const auto s2 = [](double n) { return static_cast<double>(sum_k2(static_cast<std::uint64_t>(n))); };
    const auto s3 = [](double n) { return static_cast<double>(sum_k3(static_cast<std::uint64_t>(n))); };
    const auto r2 = [&](double n) { return s2(n) - n * n * n / 3.0; };
    const auto r3 = [&](double n) { return s3(n) - n * n * n * n / 4.0; };
    EXPECT_NEAR(observed_power_at_infinity(s2, 1000.0), 3.0, 1e-2);
    EXPECT_NEAR(observed_power_at_infinity(r2, 1000.0), 2.0, 1e-2);
    EXPECT_NEAR(observed_power_at_infinity(s3, 1000.0), 4.0, 1e-2);
    EXPECT_NEAR(observed_power_at_infinity(r3, 1000.0), 3.0, 1e-2);
}

TEST(Ch10Ex11, BinomialIdentities) {
    for (unsigned n = 1; n <= 60; ++n) {
        EXPECT_EQ(ex11_sum_binomials(n), std::uint64_t{1} << n) << n;
        EXPECT_EQ(ex11_sum_even_binomials(n), std::uint64_t{1} << (n - 1)) << n;
        if (n <= 58) {
            EXPECT_EQ(ex11_sum_j_binomials(n), n * (std::uint64_t{1} << (n - 1))) << n;
        }
    }
    EXPECT_EQ(ex11_sum_even_binomials(0), 1u);  // n = 0: 1, not 2^(-1) (the identity needs n >= 1)
}

TEST(Ch10Ex12, InverseSquaresConverge) {
    const double pi2_6 = std::numbers::pi * std::numbers::pi / 6.0;
    for (unsigned n : {1u, 10u, 100u, 1000u, 100000u}) {
        const double s = ex12_sum_inv_squares(n);
        EXPECT_LT(s, 2.0 - 1.0 / n + 1e-15) << n;
        EXPECT_GT(pi2_6 - s, 1.0 / (n + 1.0)) << n;
        EXPECT_LT(pi2_6 - s, 1.0 / n) << n;
    }
}

TEST(Ch10Ex12, HarmonicAndKLogKDiverge) {
    EXPECT_EQ(ex12_harmonic(1), 1.0);  // equality at n = 1
    for (unsigned n : {2u, 3u, 10u, 1000u, 1000000u}) {
        const auto [lo, hi] = ex12_harmonic_bounds(n);
        const double h = ex12_harmonic(n);
        EXPECT_GT(h, lo) << n;
        EXPECT_LT(h, hi) << n;
    }
    for (unsigned n : {3u, 10u, 1000u, 1000000u}) {
        const auto [lo, hi] = ex12_k_log_k_bounds(n);
        const double s = ex12_sum_inv_k_log_k(n);
        EXPECT_GT(s, lo) << n;
        EXPECT_LT(s, hi) << n;
    }
    // The lower bound grows without limit: past any level for n large enough.
    EXPECT_GT(ex12_harmonic(1000000), 14.0);
    EXPECT_GT(ex12_sum_inv_k_log_k(1000000), 2.0);
}

TEST(Ch10Ex13, HarmonicMinusLogTendsToGamma) {
    double prev_x = ex13_x(1), prev_y = ex13_y(1);
    EXPECT_EQ(prev_x, 1.0);
    for (unsigned n = 2; n <= 2000; ++n) {
        const double x = ex13_x(n), y = ex13_y(n);
        EXPECT_LT(x, prev_x) << n;  // decreasing
        EXPECT_GT(y, prev_y) << n;  // increasing
        EXPECT_GT(x, ex13_euler_gamma);
        EXPECT_LT(y, ex13_euler_gamma);
        prev_x = x;
        prev_y = y;
    }
    EXPECT_NEAR(ex13_y(1), 1.0 - std::numbers::ln2, 1e-15);
    EXPECT_NEAR(ex13_x(2), 1.5 - std::numbers::ln2, 1e-15);
    for (unsigned n : {100u, 1000u, 10000u}) {
        const double N = n;
        EXPECT_NEAR(ex13_x(n) - ex13_euler_gamma, 1.0 / (2.0 * N) - 1.0 / (12.0 * N * N), 1e-3 / (N * N)) << n;
    }
}

TEST(Ch10Ex14, RadiusIsOne) {
    for (unsigned k : {10u, 1000u, 1000000u}) EXPECT_GT(ex14_ratio(k), 1.0);
    EXPECT_NEAR(ex14_ratio(1000000), 1.0, 1e-5);
    const Vector a = ex14_coefficients(100000);
    EXPECT_NEAR(radius_ratio_estimate(a), 1.0, 1e-4);
    EXPECT_NEAR(radius_root_estimate(a), 1.0, 2e-4);  // 1 + (ln k + ln ln k)/k
    // |x| < 1 converges: the tail beyond n is below |x|^(n+1) / ((1-|x|)(n+1) ln(n+1)).
    const double s = ex14_partial_sum(0.9, 400);
    EXPECT_NEAR(ex14_partial_sum(0.9, 200), s, std::pow(0.9, 201) / (0.1 * 201 * std::log(201.0)));
    // x = 1: partial sums grow like ln ln n (exercise 12).
    EXPECT_NEAR(ex14_partial_sum(1.0, 100000), ex12_sum_inv_k_log_k(100000), 1e-12);
    // x = -1: alternating, consecutive partial sums bracket the limit.
    const double s_even = ex14_partial_sum(-1.0, 100000), s_odd = ex14_partial_sum(-1.0, 100001);
    EXPECT_GT(s_even, s_odd);
    EXPECT_NEAR(s_even - s_odd, 1.0 / (100001.0 * std::log(100001.0)), 1e-12);
    for (unsigned n : {1000u, 10000u}) {
        EXPECT_GE(ex14_partial_sum(-1.0, 2 * n), s_even);
        EXPECT_LE(ex14_partial_sum(-1.0, 2 * n + 1), s_odd);
    }
}
