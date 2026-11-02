// Sections 4.2-4.3.2: lognormal variables and independent random variables.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <p4mfe/ch04/independent.hpp>
#include <p4mfe/ch04/lognormal.hpp>

using namespace p4mfe::ch04;
namespace ch02 = p4mfe::ch02;
namespace ch03 = p4mfe::ch03;

TEST(Ch04Lognormal, MomentsFromMgf) {
  const double mu = 0.1, sigma = 0.3;
  EXPECT_NEAR(LognormalMoment(1, mu, sigma), LognormalMean(mu, sigma), 1e-15);
  const double m1 = LognormalMoment(1, mu, sigma),
               m2 = LognormalMoment(2, mu, sigma);
  EXPECT_NEAR(m2 - m1 * m1, LognormalVariance(mu, sigma), 1e-14);
  const ch03::ContinuousDistribution y_dist{
      [=](double y) { return LognormalPdf(y, mu, sigma); }, 0.0,
      ch02::kInfinity};
  for (double n : {3.0, 0.5, -1.0})
    EXPECT_NEAR(
        ch03::ExpectedValue(y_dist, [n](double y) { return std::pow(y, n); }),
        LognormalMoment(n, mu, sigma), 1e-7)
        << n;
}

TEST(Ch04Lognormal, QuantilesMedianMode) {
  const double mu = 0.5, sigma = 0.8;
  for (double p : {0.01, 0.25, 0.5, 0.9, 0.999})
    EXPECT_NEAR(LognormalCdf(LognormalQuantile(p, mu, sigma), mu, sigma), p,
                1e-14)
        << p;
  EXPECT_NEAR(LognormalQuantile(0.5, mu, sigma), LognormalMedian(mu), 1e-14);
  EXPECT_LT(LognormalMode(mu, sigma), LognormalMedian(mu));
  EXPECT_LT(LognormalMedian(mu), LognormalMean(mu, sigma));
  // The mode maximizes the density.
  const double m = LognormalMode(mu, sigma);
  EXPECT_GT(LognormalPdf(m, mu, sigma), LognormalPdf(m * 1.001, mu, sigma));
  EXPECT_GT(LognormalPdf(m, mu, sigma), LognormalPdf(m * 0.999, mu, sigma));
  EXPECT_THROW((void)LognormalQuantile(1.0, mu, sigma), std::domain_error);
}

TEST(Ch04Lognormal, ParamsFromMoments) {
  const double mu = -0.2, sigma = 0.45;
  const auto p = LognormalParamsFromMoments(LognormalMean(mu, sigma),
                                            LognormalVariance(mu, sigma));
  EXPECT_NEAR(p.mean, mu, 1e-14);
  EXPECT_NEAR(p.Stddev(), sigma, 1e-14);
}

TEST(Ch04Independent, ConvolutionOfNormalsIsNormal) {
  const NormalParams x_dist{1.0, 0.25}, y_dist{-0.5, 1.44};
  const auto f_x = [&](double x) {
    return ch03::NormalPdf(x, x_dist.mean, x_dist.Stddev());
  };
  const auto f_y = [&](double y) {
    return ch03::NormalPdf(y, y_dist.mean, y_dist.Stddev());
  };
  const auto sum = IndependentNormalSum(std::array{x_dist, y_dist});
  EXPECT_DOUBLE_EQ(sum.mean, 0.5);
  EXPECT_DOUBLE_EQ(sum.variance, 1.69);
  for (double z : {-3.0, 0.0, 0.5, 2.0})
    EXPECT_NEAR(SumPdf(f_x, f_y, z), ch03::NormalPdf(z, 0.5, 1.3), 1e-9) << z;
}

TEST(Ch04Independent, ConvolutionOfExponentialsIsGamma) {
  // Sum of two independent Exp(lambda): density lambda^2 z e^(-lambda z), z >
  // 0.
  const double lambda = 2.0;
  const auto e = ch03::ExponentialDistribution(lambda);
  for (double z : {0.25, 1.0, 2.5})
    EXPECT_NEAR(SumPdf(e.pdf, e.pdf, z, 0.0, z),
                lambda * lambda * z * std::exp(-lambda * z), 1e-9)
        << z;
}

TEST(Ch04Independent, JointDensityFactorsAndCovarianceVanishes) {
  const auto f_x = [](double x) { return p4mfe::NormPdf(x); };
  const auto e = ch03::ExponentialDistribution(1.0);
  const auto f = IndependentJointPdf(f_x, e.pdf);
  EXPECT_DOUBLE_EQ(f(0.3, 2.0), p4mfe::NormPdf(0.3) * std::exp(-2.0));
  // E[(X + 1) Y] = E[X + 1] E[Y] = 1 for independent X, Y: integrate the
  // factored density over a box that holds all but ~1e-14 of the mass.
  const auto inner = [&](double x) {
    return ch02::SimpsonRule([&](double y) { return (x + 1.0) * y * f(x, y); },
                             0.0, 40.0, 400);
  };
  EXPECT_NEAR(ch02::SimpsonRule(inner, -9.0, 9.0, 200), 1.0, 1e-6);
}

TEST(Ch04Independent, LinearCombinationOfNormals) {
  const std::array x_dist{NormalParams{1.0, 4.0}, NormalParams{-2.0, 9.0},
                          NormalParams{0.5, 1.0}};
  const std::array a{2.0, -1.0, 3.0};
  const auto combination = IndependentNormalCombination(a, x_dist);
  EXPECT_DOUBLE_EQ(combination.mean, 2.0 + 2.0 + 1.5);
  EXPECT_DOUBLE_EQ(combination.variance, 16.0 + 9.0 + 9.0);
  // Two-variable case agrees with chapter 3 at zero correlation.
  const auto pair_ch03 =
      ch03::LinearCombination(2.0, x_dist[0], -1.0, x_dist[1], 0.0);
  const auto pair = IndependentNormalCombination(
      std::array{2.0, -1.0}, std::array{x_dist[0], x_dist[1]});
  EXPECT_DOUBLE_EQ(pair_ch03.mean, pair.mean);
  EXPECT_DOUBLE_EQ(pair_ch03.variance, pair.variance);
  EXPECT_THROW((void)IndependentNormalCombination(std::array{1.0}, x_dist),
               std::invalid_argument);
}

TEST(Ch04Independent, ProductOfLognormals) {
  const NormalParams log1{0.1, 0.04}, log2{-0.3, 0.09};
  const auto product = IndependentLognormalProduct(std::array{log1, log2});
  // E[Y1 Y2] = E[Y1] E[Y2] by independence.
  EXPECT_NEAR(LognormalMean(product.mean, product.Stddev()),
              LognormalMean(log1.mean, log1.Stddev()) *
                  LognormalMean(log2.mean, log2.Stddev()),
              1e-14);
  // Ratio Y1 / Y2: E[Y1 / Y2] = E[Y1] E[1 / Y2] = E[Y1] e^(-mu2 + s2^2/2).
  const auto ratio = IndependentLognormalProduct(std::array{1.0, -1.0},
                                                 std::array{log1, log2});
  EXPECT_NEAR(LognormalMean(ratio.mean, ratio.Stddev()),
              LognormalMean(log1.mean, log1.Stddev()) *
                  LognormalMoment(-1, log2.mean, log2.Stddev()),
              1e-14);
}
