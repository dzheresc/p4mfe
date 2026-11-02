// Section 4.1: densities of functions of random variables.

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <p4mfe/ch04/change_of_variables.hpp>

using namespace p4mfe::ch04;
using p4mfe::norm_pdf;
namespace ch02 = p4mfe::ch02;
namespace ch03 = p4mfe::ch03;

namespace {
const auto phi = [](double x) { return norm_pdf(x); };
}  // namespace

TEST(Ch04ChangeOfVariables, LinearOfStandardNormalIsNormal) {
  const double a = -2.0, b = 1.5;  // decreasing g: |h'| matters
  const auto f = linear_transform_pdf(phi, a, b);
  for (double y : {-4.0, -1.0, 0.0, 1.5, 3.0, 6.0})
    EXPECT_NEAR(f(y), ch03::NormalPdf(y, b, 2.0), 1e-16) << y;
}

TEST(Ch04ChangeOfVariables, GeneralFormulaMatchesLinear) {
  const double a = 3.0, b = -1.0;
  const auto f1 = transformed_pdf(
      phi, [=](double y) { return (y - b) / a; },
      [=](double) { return 1.0 / a; });
  const auto f2 = transformed_pdf_numeric(
      phi, [=](double y) { return (y - b) / a; }, [](double) { return true; });
  for (double y : {-5.0, -1.0, 0.0, 2.0}) {
    EXPECT_NEAR(f1(y), ch03::NormalPdf(y, b, a), 1e-16);
    EXPECT_NEAR(f2(y), ch03::NormalPdf(y, b, a), 1e-10);
  }
}

TEST(Ch04ChangeOfVariables, ExpOfNormalIsLognormal) {
  const double mu = 0.2, sigma = 0.4;
  const auto fX = [=](double x) { return ch03::NormalPdf(x, mu, sigma); };
  const auto fY = exp_transform_pdf(fX);
  const auto fY_general = transformed_pdf(
      fX, [](double y) { return std::log(y); },
      [](double y) { return 1.0 / y; }, [](double y) { return y > 0; });
  for (double y : {0.1, 0.5, 1.0, 1.2, 2.0, 5.0}) {
    EXPECT_NEAR(fY(y), ch03::LognormalPdf(y, mu, sigma), 1e-15) << y;
    EXPECT_NEAR(fY_general(y), ch03::LognormalPdf(y, mu, sigma), 1e-15) << y;
  }
  EXPECT_EQ(fY(0.0), 0.0);
  EXPECT_EQ(fY_general(-1.0), 0.0);
}

TEST(Ch04ChangeOfVariables, SqrtOfExponentialIsRayleigh) {
  // X ~ Exp(lambda), Y = sqrt(X): h(y) = y^2, f_Y(y) = 2 lambda y e^(-lambda
  // y^2), E[Y] = sqrt(pi / lambda) / 2. Only h is given; h' is numerical.
  const double lambda = 1.5;
  const auto X = ch03::ExponentialDistribution(lambda);
  const auto f = transformed_pdf_numeric(
      X.pdf, [](double y) { return y * y; }, [](double y) { return y >= 0; });
  for (double y : {0.1, 0.5, 1.0, 2.0})
    EXPECT_NEAR(f(y), 2 * lambda * y * std::exp(-lambda * y * y), 1e-9) << y;
  const ch03::ContinuousDistribution Y{f, 0.0, ch02::kInfinity};
  EXPECT_NEAR(ch03::Cdf(Y, 50.0), 1.0, 1e-8);
  EXPECT_NEAR(ch03::Mean(Y), 0.5 * std::sqrt(std::numbers::pi / lambda), 1e-8);
}

TEST(Ch04ChangeOfVariables, SquareOfStandardNormalIsChiSquared) {
  const auto f = square_transform_pdf(phi);
  for (double y : {0.01, 0.5, 1.0, 2.0, 7.0})
    EXPECT_NEAR(
        f(y), std::exp(-0.5 * y) / std::sqrt(2.0 * std::numbers::pi * y), 1e-15)
        << y;
  EXPECT_EQ(f(-1.0), 0.0);
  // E[Z^2] = 1, integrating away from the integrable singularity at 0.
  const ch03::ContinuousDistribution Y{f, 0.0, ch02::kInfinity};
  EXPECT_NEAR(ch03::Mean(Y, 1e-8), 1.0, 1e-6);
}
