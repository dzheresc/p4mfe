#include <p4mfe/ch08/extrema.hpp>
#include <p4mfe/ch09/exercises/ex01_constrained_extrema.hpp>
#include <p4mfe/ch09/exercises/ex02_am_gm_hm.hpp>
#include <p4mfe/ch09/exercises/ex03_five_asset_portfolio.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

using namespace p4mfe;
using namespace p4mfe::ch09;
using namespace p4mfe::ch09::ex;

// ------------------------------------------------------------ Exercise 1 ---
TEST(Ch09Ex01, FourStationaryPointsInSymmetricPairs) {
    const auto pts = ex01_stationary_points();
    ASSERT_EQ(pts.size(), 4u);
    EXPECT_NEAR(pts[0].value, 12.761255, 1e-5);
    EXPECT_NEAR(pts[1].value, 3.434511, 1e-5);
    EXPECT_NEAR(pts[2].value, -3.434511, 1e-5);
    EXPECT_NEAR(pts[3].value, -12.761255, 1e-5);
    EXPECT_EQ(pts[0].type, CriticalPointType::Maximum);
    EXPECT_EQ(pts[1].type, CriticalPointType::Saddle);
    EXPECT_EQ(pts[2].type, CriticalPointType::Saddle);
    EXPECT_EQ(pts[3].type, CriticalPointType::Minimum);
    // x -> -x, lambda -> -lambda maps the maximum to the minimum.
    for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(pts[0].x[i], -pts[3].x[i], 1e-7);
    for (std::size_t j = 0; j < 2; ++j) EXPECT_NEAR(pts[0].lambda[j], -pts[3].lambda[j], 1e-7);
}

TEST(Ch09Ex01, MaximumPointAndMultipliers) {
    const auto mx = ex01_stationary_points().front();
    const Vector expected{2.137669, -0.683412, 0.406720, -2.009150};
    for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(mx.x[i], expected[i], 1e-5);
    EXPECT_NEAR(mx.lambda[0], 0.995446, 1e-5);
    EXPECT_NEAR(mx.lambda[1], -1.229346, 1e-5);
    // Feasible, and x agrees with the formulas in terms of lambda.
    for (double v : ex01_g(mx.x)) EXPECT_NEAR(v, 0.0, 1e-9);
    const Vector xl = ex01_x_from_lambda(mx.lambda);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(xl[i], mx.x[i], 1e-6);
    for (double v : lagrangian_gradient(ex01_f, ex01_g, mx.x, mx.lambda)) EXPECT_NEAR(v, 0.0, 1e-6);
}

TEST(Ch09Ex01, SweepOfTheFeasibleSetAgrees) {
    const auto pts = ex01_stationary_points();
    const auto s = ex01_sweep(600, 1200);
    // No sampled feasible point beats the Lagrange extrema, and the grid gets close to them.
    EXPECT_LE(s.max, pts.front().value + 1e-9);
    EXPECT_GE(s.min, pts.back().value - 1e-9);
    EXPECT_NEAR(s.max, pts.front().value, 1e-3);
    EXPECT_NEAR(s.min, pts.back().value, 1e-3);
}

// ------------------------------------------------------------ Exercise 2 ---
namespace {
// Deterministic positive test vectors (no RNG, so identical on every platform).
std::vector<std::vector<double>> ex02_vectors() {
    std::vector<std::vector<double>> out;
    for (std::size_t n = 1; n <= 12; ++n)
        for (int k = 0; k < 5; ++k) {
            std::vector<double> x(n);
            for (std::size_t i = 0; i < n; ++i)
                x[i] = std::exp(3.0 * std::sin(1.7 * static_cast<double>(i + 1) + 0.9 * k + 0.3 * static_cast<double>(n)));
            out.push_back(x);
        }
    return out;
}
}  // namespace

TEST(Ch09Ex02, MeansAreOrdered) {
    for (const auto& x : ex02_vectors()) {
        const double A = arithmetic_mean(x), G = geometric_mean(x), H = harmonic_mean(x);
        EXPECT_GE(A, G * (1.0 - 1e-14));
        EXPECT_GE(G, H * (1.0 - 1e-14));
        if (x.size() > 1) {  // the entries differ, so both inequalities are strict
            EXPECT_GT(A - G, 1e-6 * A);
            EXPECT_GT(G - H, 1e-6 * G);
        }
    }
}

TEST(Ch09Ex02, EqualityForEqualEntries) {
    const std::vector<double> x(7, 2.5);
    EXPECT_NEAR(arithmetic_mean(x), 2.5, 1e-14);
    EXPECT_NEAR(geometric_mean(x), 2.5, 1e-14);
    EXPECT_NEAR(harmonic_mean(x), 2.5, 1e-14);
    // Known small case: 1, 2, 4 has A = 7/3, G = 2, H = 12/7.
    EXPECT_NEAR(arithmetic_mean({1, 2, 4}), 7.0 / 3.0, 1e-14);
    EXPECT_NEAR(geometric_mean({1, 2, 4}), 2.0, 1e-14);
    EXPECT_NEAR(harmonic_mean({1, 2, 4}), 12.0 / 7.0, 1e-14);
    EXPECT_THROW((void)geometric_mean({1.0, 0.0}), std::invalid_argument);
    EXPECT_THROW((void)harmonic_mean({}), std::invalid_argument);
}

TEST(Ch09Ex02, LagrangeMaximizerIsTheEqualVector) {
    const double a = 1.5;
    const auto r = ex02_lagrange_max({1.0, 1.8, 2.2, 1.2, 1.3}, a);
    for (double v : r.x) EXPECT_NEAR(v, a, 1e-8);
    EXPECT_NEAR(r.lambda[0], -1.0 / a, 1e-8);
    EXPECT_EQ(ex02_classify(r, a), CriticalPointType::Maximum);
}

// ------------------------------------------------------------ Exercise 3 ---
TEST(Ch09Ex03, CorrelationIsPositiveDefinite) {
    const Ex03Data d;
    for (std::size_t k = 1; k <= 5; ++k) {
        ch01::Matrix M(k, ch01::Vector(k));
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j) M[i][j] = d.corr(i, j);
        EXPECT_GT(ch08::determinant(M), 0.0) << k;
    }
}

TEST(Ch09Ex03, MinimumVarianceFor15Percent) {
    const auto p = ex03_min_variance();
    const Vector expected{0.067354, 0.217299, -0.002244, 0.406893, 0.310697};
    double sum = 0.0;
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(p.w[i], expected[i], 1e-6);
        sum += p.w[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
    EXPECT_NEAR(p.mean, 0.15, 1e-12);
    EXPECT_NEAR(p.std_dev, 0.141214, 1e-6);
    const Ex03Data d;
    EXPECT_NEAR(p.std_dev * p.std_dev, frontier_variance(d.mu, d.covariance(), 0.15), 1e-12);
}

TEST(Ch09Ex03, MaximumReturnFor25PercentRisk) {
    const auto p = ex03_max_return();
    const Vector expected{-0.395503, 0.096062, 0.045839, 0.709880, 0.543722};
    double sum = 0.0;
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(p.w[i], expected[i], 1e-6);
        sum += p.w[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
    EXPECT_NEAR(p.std_dev, 0.25, 1e-12);
    EXPECT_NEAR(p.mean, 0.199151, 1e-6);
}

TEST(Ch09Ex03, GeneralLagrangeSolverAgrees) {
    const Ex03Data d;
    const Matrix S = d.covariance();
    const auto sum_minus_1 = [](const Vector& w) {
        double s = -1.0;
        for (double v : w) s += v;
        return s;
    };
    // (i) min w'Sw s.t. mu.w = 0.15, sum w = 1, from equal weights.
    {
        const auto f = [&](const Vector& w) { return portfolio_variance(w, S); };
        const auto g = [&](const Vector& w) { return Vector{dot(d.mu, w) - 0.15, sum_minus_1(w)}; };
        const auto r = lagrange_stationary_point(f, g, Vector(5, 0.2));
        const auto p = ex03_min_variance();
        for (std::size_t i = 0; i < 5; ++i) EXPECT_NEAR(r.x[i], p.w[i], 1e-6);
        EXPECT_EQ(classify_constrained(f, g, r.x, r.lambda), CriticalPointType::Minimum);
    }
    // (ii) max mu.w s.t. w'Sw = 0.25^2, sum w = 1, from the answer to (i).
    {
        const auto f = [&](const Vector& w) { return dot(d.mu, w); };
        const auto g = [&](const Vector& w) { return Vector{portfolio_variance(w, S) - 0.0625, sum_minus_1(w)}; };
        const auto r = lagrange_stationary_point(f, g, ex03_min_variance().w);
        const auto p = ex03_max_return();
        for (std::size_t i = 0; i < 5; ++i) EXPECT_NEAR(r.x[i], p.w[i], 1e-6);
        EXPECT_EQ(classify_constrained(f, g, r.x, r.lambda), CriticalPointType::Maximum);
    }
}
