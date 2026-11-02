#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <numbers>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <stdexcept>
#include <vector>

using namespace p4mfe::ch03;

TEST(Ch03Discrete, FairDie) {
  const DiscreteDistribution die({1, 2, 3, 4, 5, 6},
                                 std::vector<double>(6, 1.0 / 6.0));
  EXPECT_NEAR(mean(die), 3.5, 1e-14);
  EXPECT_NEAR(variance(die), 35.0 / 12.0, 1e-14);
  // E[X^2] - E[X]^2 equals var(X).
  EXPECT_NEAR(expected_value(die, [](double x) { return x * x; }) - 3.5 * 3.5,
              35.0 / 12.0, 1e-13);
}

TEST(Ch03Discrete, RejectsBadProbabilities) {
  EXPECT_THROW(DiscreteDistribution({1, 2}, {0.5, 0.6}), std::invalid_argument);
  EXPECT_THROW(DiscreteDistribution({1, 2}, {1.5, -0.5}),
               std::invalid_argument);
  EXPECT_THROW(DiscreteDistribution({1, 2}, {1.0}), std::invalid_argument);
}

TEST(Ch03Continuous, Exponential) {
  const double lambda = 2.0;
  const auto X = exponential_distribution(lambda);
  EXPECT_NEAR(expected_value(X, [](double) { return 1.0; }), 1.0, 1e-8);
  EXPECT_NEAR(mean(X), 1.0 / lambda, 1e-8);
  EXPECT_NEAR(variance(X), 1.0 / (lambda * lambda), 1e-8);
  EXPECT_NEAR(cdf(X, 1.0), 1.0 - std::exp(-lambda), 1e-8);
  EXPECT_EQ(cdf(X, -1.0), 0.0);
}

TEST(Ch03Continuous, StandardNormalMoments) {
  const ContinuousDistribution Z{[](double x) { return norm_pdf(x); },
                                 -p4mfe::ch02::infinity, p4mfe::ch02::infinity};
  EXPECT_NEAR(mean(Z), 0.0, 1e-9);
  EXPECT_NEAR(variance(Z), 1.0, 1e-8);
  EXPECT_NEAR(cdf(Z, 1.0), norm_cdf(1.0), 1e-8);
  // E[e^(tZ)] = e^(t^2/2).
  EXPECT_NEAR(expected_value(Z, [](double x) { return std::exp(0.5 * x); }),
              std_normal_mgf(0.5), 1e-8);
}

TEST(Ch03Covariance, JointDiscrete) {
  // Y = 2X + 1 with X uniform on {0, 1, 2}: correlation 1, cov = 2 var(X).
  const std::array<JointOutcome, 3> joint{
      {{0, 1, 1.0 / 3}, {1, 3, 1.0 / 3}, {2, 5, 1.0 / 3}}};
  EXPECT_NEAR(covariance(joint), 2.0 * (2.0 / 3.0), 1e-14);
  EXPECT_NEAR(correlation(joint), 1.0, 1e-14);
  // Independent fair coins: zero covariance.
  const std::array<JointOutcome, 4> coins{
      {{0, 0, .25}, {0, 1, .25}, {1, 0, .25}, {1, 1, .25}}};
  EXPECT_NEAR(covariance(coins), 0.0, 1e-15);
}

TEST(Ch03Covariance, VarianceOfSum) {
  EXPECT_DOUBLE_EQ(variance_of_sum(1, 4, 1, 9, 0), 13);
  EXPECT_DOUBLE_EQ(variance_of_sum(1, 4, -1, 9, 6),
                   1);  // perfectly correlated: (2 - 3)^2
}

TEST(Ch03Covariance, SampleEstimators) {
  const std::vector<double> x{1, 2, 3, 4, 5};
  const std::vector<double> y{2, 4, 5, 4, 5};
  EXPECT_DOUBLE_EQ(sample_mean(x), 3.0);
  EXPECT_DOUBLE_EQ(sample_variance(x), 2.5);
  EXPECT_DOUBLE_EQ(sample_covariance(x, y), 1.5);
  EXPECT_NEAR(sample_correlation(x, y), 1.5 / std::sqrt(2.5 * 1.5), 1e-15);
  const std::vector<double> neg{-1, -2, -3, -4, -5};
  EXPECT_NEAR(sample_correlation(x, neg), -1.0, 1e-15);
}
