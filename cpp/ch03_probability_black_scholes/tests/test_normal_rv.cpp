#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/cdf_approx.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>

using namespace p4mfe::ch03;

TEST(Ch03Normal, PdfCdfShiftScale) {
  EXPECT_NEAR(normal_pdf(3.0, 1.0, 2.0), norm_pdf(1.0) / 2.0, 1e-16);
  EXPECT_NEAR(normal_cdf(1.0, 1.0, 2.0), 0.5, 1e-16);
  EXPECT_NEAR(normal_cdf(1.0 + 1.96 * 2.0, 1.0, 2.0), 0.9750021048517795,
              1e-13);
}

TEST(Ch03Normal, LinearCombination) {
  const NormalParams X{1.0, 4.0}, Y{-2.0, 9.0};
  const auto Z = linear_combination(2.0, X, 1.0, Y, 0.5);
  EXPECT_DOUBLE_EQ(Z.mean, 0.0);
  EXPECT_DOUBLE_EQ(Z.variance, 4 * 4 + 9 + 2 * 2 * 0.5 * 2 * 3);
}

TEST(Ch03Lognormal, MomentsMatchIntegration) {
  const double mu = 0.1, sigma = 0.3;
  const ContinuousDistribution Y{
      [=](double y) { return lognormal_pdf(y, mu, sigma); }, 0.0,
      p4mfe::ch02::kInfinity};
  EXPECT_NEAR(mean(Y), lognormal_mean(mu, sigma), 1e-8);
  EXPECT_NEAR(variance(Y), lognormal_variance(mu, sigma), 1e-8);
  EXPECT_NEAR(lognormal_cdf(std::exp(mu), mu, sigma), 0.5, 1e-15);
}

TEST(Ch03Lognormal, PriceModel) {
  const double S0 = 50, T = 2, sigma = 0.25, mu = 0.08, q = 0.01;
  const auto ln_ST = log_price_distribution(S0, T, sigma, mu, q);
  EXPECT_NEAR(lognormal_mean(ln_ST.mean, ln_ST.stddev()),
              S0 * std::exp((mu - q) * T), 1e-12);
  // P(S_T > K) under mu = r is N(d2).
  const double K = 55, r = 0.04;
  const double d2 = (std::log(S0 / K) + (r - q - 0.5 * sigma * sigma) * T) /
                    (sigma * std::sqrt(T));
  EXPECT_NEAR(prob_price_above(K, S0, T, sigma, r, q), norm_cdf(d2), 1e-14);
}

// Section 3.9: both approximations of N(x) against the erfc-based reference.
TEST(Ch03CdfApprox, PolynomialWithinAbramowitzStegunBound) {
  for (double x = -8.0; x <= 8.0; x += 0.01)
    EXPECT_NEAR(norm_cdf_poly(x), norm_cdf(x), 7.5e-8) << x;
  EXPECT_NEAR(norm_cdf_poly(0.0), 0.5, 1e-9);
}

TEST(Ch03CdfApprox, SimpsonToTolerance) {
  for (double x : {-6.0, -2.5, -1.0, -0.1, 0.0, 0.3, 1.0, 1.96, 4.0}) {
    EXPECT_NEAR(norm_cdf_simpson(x), norm_cdf(x), 1e-12) << x;
  }
  EXPECT_NEAR(norm_cdf_simpson(1.0, 1e-6), norm_cdf(1.0), 1e-6);
}
