#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/cdf_approx.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>

using namespace p4mfe::ch03;

TEST(Ch03Normal, PdfCdfShiftScale) {
  EXPECT_NEAR(NormalPdf(3.0, 1.0, 2.0), NormPdf(1.0) / 2.0, 1e-16);
  EXPECT_NEAR(NormalCdf(1.0, 1.0, 2.0), 0.5, 1e-16);
  EXPECT_NEAR(NormalCdf(1.0 + 1.96 * 2.0, 1.0, 2.0), 0.9750021048517795, 1e-13);
}

TEST(Ch03Normal, LinearCombination) {
  const NormalParams x{1.0, 4.0}, y{-2.0, 9.0};
  const auto z = LinearCombination(2.0, x, 1.0, y, 0.5);
  EXPECT_DOUBLE_EQ(z.mean, 0.0);
  EXPECT_DOUBLE_EQ(z.variance, 4 * 4 + 9 + 2 * 2 * 0.5 * 2 * 3);
}

TEST(Ch03Lognormal, MomentsMatchIntegration) {
  const double mu = 0.1, sigma = 0.3;
  const ContinuousDistribution y{
      [=](double y) { return LognormalPdf(y, mu, sigma); }, 0.0,
      p4mfe::ch02::kInfinity};
  EXPECT_NEAR(Mean(y), LognormalMean(mu, sigma), 1e-8);
  EXPECT_NEAR(Variance(y), LognormalVariance(mu, sigma), 1e-8);
  EXPECT_NEAR(LognormalCdf(std::exp(mu), mu, sigma), 0.5, 1e-15);
}

TEST(Ch03Lognormal, PriceModel) {
  const double spot0 = 50, maturity = 2, sigma = 0.25, mu = 0.08, q = 0.01;
  const auto ln_spot_t = LogPriceDistribution(spot0, maturity, sigma, mu, q);
  EXPECT_NEAR(LognormalMean(ln_spot_t.mean, ln_spot_t.Stddev()),
              spot0 * std::exp((mu - q) * maturity), 1e-12);
  // P(S_T > K) under mu = r is N(d2).
  const double strike = 55, r = 0.04;
  const double d2 =
      (std::log(spot0 / strike) + (r - q - 0.5 * sigma * sigma) * maturity) /
      (sigma * std::sqrt(maturity));
  EXPECT_NEAR(ProbPriceAbove(strike, spot0, maturity, sigma, r, q), NormCdf(d2),
              1e-14);
}

// Section 3.9: both approximations of N(x) against the erfc-based reference.
TEST(Ch03CdfApprox, PolynomialWithinAbramowitzStegunBound) {
  for (double x = -8.0; x <= 8.0; x += 0.01)
    EXPECT_NEAR(NormCdfPoly(x), NormCdf(x), 7.5e-8) << x;
  EXPECT_NEAR(NormCdfPoly(0.0), 0.5, 1e-9);
}

TEST(Ch03CdfApprox, SimpsonToTolerance) {
  for (double x : {-6.0, -2.5, -1.0, -0.1, 0.0, 0.3, 1.0, 1.96, 4.0}) {
    EXPECT_NEAR(NormCdfSimpson(x), NormCdf(x), 1e-12) << x;
  }
  EXPECT_NEAR(NormCdfSimpson(1.0, 1e-6), NormCdf(1.0), 1e-6);
}
