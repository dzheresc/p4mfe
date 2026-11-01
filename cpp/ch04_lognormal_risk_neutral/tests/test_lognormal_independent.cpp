// Sections 4.2-4.3.2: lognormal variables and independent random variables.

#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <p4mfe/ch04/independent.hpp>
#include <p4mfe/ch04/lognormal.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

using namespace p4mfe::ch04;
namespace ch02 = p4mfe::ch02;
namespace ch03 = p4mfe::ch03;

TEST(Ch04Lognormal, MomentsFromMgf) {
    const double mu = 0.1, sigma = 0.3;
    EXPECT_NEAR(lognormal_moment(1, mu, sigma), lognormal_mean(mu, sigma), 1e-15);
    const double m1 = lognormal_moment(1, mu, sigma), m2 = lognormal_moment(2, mu, sigma);
    EXPECT_NEAR(m2 - m1 * m1, lognormal_variance(mu, sigma), 1e-14);
    const ch03::ContinuousDistribution Y{[=](double y) { return lognormal_pdf(y, mu, sigma); }, 0.0,
                                         ch02::infinity};
    for (double n : {3.0, 0.5, -1.0})
        EXPECT_NEAR(ch03::expected_value(Y, [n](double y) { return std::pow(y, n); }), lognormal_moment(n, mu, sigma),
                    1e-7)
            << n;
}

TEST(Ch04Lognormal, QuantilesMedianMode) {
    const double mu = 0.5, sigma = 0.8;
    for (double p : {0.01, 0.25, 0.5, 0.9, 0.999})
        EXPECT_NEAR(lognormal_cdf(lognormal_quantile(p, mu, sigma), mu, sigma), p, 1e-14) << p;
    EXPECT_NEAR(lognormal_quantile(0.5, mu, sigma), lognormal_median(mu), 1e-14);
    EXPECT_LT(lognormal_mode(mu, sigma), lognormal_median(mu));
    EXPECT_LT(lognormal_median(mu), lognormal_mean(mu, sigma));
    // The mode maximizes the density.
    const double m = lognormal_mode(mu, sigma);
    EXPECT_GT(lognormal_pdf(m, mu, sigma), lognormal_pdf(m * 1.001, mu, sigma));
    EXPECT_GT(lognormal_pdf(m, mu, sigma), lognormal_pdf(m * 0.999, mu, sigma));
    EXPECT_THROW((void)lognormal_quantile(1.0, mu, sigma), std::domain_error);
}

TEST(Ch04Lognormal, ParamsFromMoments) {
    const double mu = -0.2, sigma = 0.45;
    const auto p = lognormal_params_from_moments(lognormal_mean(mu, sigma), lognormal_variance(mu, sigma));
    EXPECT_NEAR(p.mean, mu, 1e-14);
    EXPECT_NEAR(p.stddev(), sigma, 1e-14);
}

TEST(Ch04Independent, ConvolutionOfNormalsIsNormal) {
    const NormalParams X{1.0, 0.25}, Y{-0.5, 1.44};
    const auto fX = [&](double x) { return ch03::normal_pdf(x, X.mean, X.stddev()); };
    const auto fY = [&](double y) { return ch03::normal_pdf(y, Y.mean, Y.stddev()); };
    const auto S = independent_normal_sum(std::array{X, Y});
    EXPECT_DOUBLE_EQ(S.mean, 0.5);
    EXPECT_DOUBLE_EQ(S.variance, 1.69);
    for (double z : {-3.0, 0.0, 0.5, 2.0}) EXPECT_NEAR(sum_pdf(fX, fY, z), ch03::normal_pdf(z, 0.5, 1.3), 1e-9) << z;
}

TEST(Ch04Independent, ConvolutionOfExponentialsIsGamma) {
    // Sum of two independent Exp(lambda): density lambda^2 z e^(-lambda z), z > 0.
    const double lambda = 2.0;
    const auto e = ch03::exponential_distribution(lambda);
    for (double z : {0.25, 1.0, 2.5})
        EXPECT_NEAR(sum_pdf(e.pdf, e.pdf, z, 0.0, z), lambda * lambda * z * std::exp(-lambda * z), 1e-9) << z;
}

TEST(Ch04Independent, JointDensityFactorsAndCovarianceVanishes) {
    const auto fX = [](double x) { return p4mfe::norm_pdf(x); };
    const auto e = ch03::exponential_distribution(1.0);
    const auto f = independent_joint_pdf(fX, e.pdf);
    EXPECT_DOUBLE_EQ(f(0.3, 2.0), p4mfe::norm_pdf(0.3) * std::exp(-2.0));
    // E[(X + 1) Y] = E[X + 1] E[Y] = 1 for independent X, Y: integrate the
    // factored density over a box that holds all but ~1e-14 of the mass.
    const auto inner = [&](double x) {
        return ch02::simpson_rule([&](double y) { return (x + 1.0) * y * f(x, y); }, 0.0, 40.0, 400);
    };
    EXPECT_NEAR(ch02::simpson_rule(inner, -9.0, 9.0, 200), 1.0, 1e-6);
}

TEST(Ch04Independent, LinearCombinationOfNormals) {
    const std::array X{NormalParams{1.0, 4.0}, NormalParams{-2.0, 9.0}, NormalParams{0.5, 1.0}};
    const std::array a{2.0, -1.0, 3.0};
    const auto C = independent_normal_combination(a, X);
    EXPECT_DOUBLE_EQ(C.mean, 2.0 + 2.0 + 1.5);
    EXPECT_DOUBLE_EQ(C.variance, 16.0 + 9.0 + 9.0);
    // Two-variable case agrees with chapter 3 at zero correlation.
    const auto D = ch03::linear_combination(2.0, X[0], -1.0, X[1], 0.0);
    const auto E = independent_normal_combination(std::array{2.0, -1.0}, std::array{X[0], X[1]});
    EXPECT_DOUBLE_EQ(D.mean, E.mean);
    EXPECT_DOUBLE_EQ(D.variance, E.variance);
    EXPECT_THROW((void)independent_normal_combination(std::array{1.0}, X), std::invalid_argument);
}

TEST(Ch04Independent, ProductOfLognormals) {
    const NormalParams L1{0.1, 0.04}, L2{-0.3, 0.09};
    const auto P = independent_lognormal_product(std::array{L1, L2});
    // E[Y1 Y2] = E[Y1] E[Y2] by independence.
    EXPECT_NEAR(lognormal_mean(P.mean, P.stddev()),
                lognormal_mean(L1.mean, L1.stddev()) * lognormal_mean(L2.mean, L2.stddev()), 1e-14);
    // Ratio Y1 / Y2: E[Y1 / Y2] = E[Y1] E[1 / Y2] = E[Y1] e^(-mu2 + s2^2/2).
    const auto R = independent_lognormal_product(std::array{1.0, -1.0}, std::array{L1, L2});
    EXPECT_NEAR(lognormal_mean(R.mean, R.stddev()),
                lognormal_mean(L1.mean, L1.stddev()) * lognormal_moment(-1, L2.mean, L2.stddev()), 1e-14);
}
