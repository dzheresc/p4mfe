#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/parity.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>
#include <vector>

using namespace p4mfe;
using namespace p4mfe::ch10;

namespace {
const std::vector<double> kPts{0.1, 0.5, 1.0, 1.7, 2.5, 4.0};
}

// ----------------------------------------------------------------- 10.1 ---
TEST(Ch10Parity, ClassifiesClassicalFunctions) {
  const auto cosf = [](double x) { return std::cos(x); };
  const auto sinf = [](double x) { return std::sin(x); };
  const auto expf = [](double x) { return std::exp(x); };
  EXPECT_EQ(ClassifyParity(cosf, kPts), Parity::kEven);
  EXPECT_EQ(ClassifyParity(sinf, kPts), Parity::kOdd);
  EXPECT_EQ(ClassifyParity(expf, kPts), Parity::kNeither);
  EXPECT_EQ(ClassifyParity([](double x) { return norm_pdf(x); }, kPts),
            Parity::kEven);
  // N(x) - 1/2 is odd, i.e. N(-x) = 1 - N(x).
  EXPECT_TRUE(IsOdd([](double x) { return norm_cdf(x) - 0.5; }, kPts, 1e-14));
}

TEST(Ch10Parity, ProductsAndDerivatives) {
  const auto sinf = [](double x) { return std::sin(x); };
  const auto cube = [](double x) { return x * x * x; };
  // odd * odd = even, even * odd = odd.
  EXPECT_TRUE(IsEven([&](double x) { return sinf(x) * cube(x); }, kPts));
  EXPECT_TRUE(IsOdd([&](double x) { return std::cos(x) * cube(x); }, kPts));
  // The derivative of the even function x^4 + cos x is odd.
  EXPECT_TRUE(
      IsOdd([](double x) { return 4 * x * x * x - std::sin(x); }, kPts));
}

TEST(Ch10Parity, EvenOddDecomposition) {
  const auto f = [](double x) { return std::exp(x); };
  for (double x : kPts) {
    EXPECT_NEAR(EvenPart(f, x), std::cosh(x), 1e-13 * std::cosh(x));
    EXPECT_NEAR(OddPart(f, x), std::sinh(x), 1e-13 * std::cosh(x));
    EXPECT_DOUBLE_EQ(EvenPart(f, x) + OddPart(f, x), f(x));
  }
  const auto ep = [&](double x) { return EvenPart(f, x); };
  const auto op = [&](double x) { return OddPart(f, x); };
  EXPECT_TRUE(IsEven(ep, kPts));
  EXPECT_TRUE(IsOdd(op, kPts));
}

TEST(Ch10Parity, SymmetricIntegrals) {
  // int_{-a}^{a} e^x = 2 sinh a; odd part integrates to zero.
  const auto f = [](double x) { return std::exp(x) + x * x * x * std::cos(x); };
  EXPECT_NEAR(SymmetricIntegral(f, 1.5), 2.0 * std::sinh(1.5), 1e-10);
  EXPECT_NEAR(SymmetricIntegral(f, 1.5), ch02::Integrate(f, -1.5, 1.5, 400),
              1e-10);
  EXPECT_NEAR(SymmetricIntegral([](double x) { return std::sin(x); }, 3.0), 0.0,
              1e-15);
  // int_{-a}^{a} phi = 2 N(a) - 1.
  EXPECT_NEAR(SymmetricIntegral([](double x) { return norm_pdf(x); }, 2.0),
              2.0 * norm_cdf(2.0) - 1.0, 1e-10);
  EXPECT_THROW((void)SymmetricIntegral(f, -1.0), std::invalid_argument);
}

// ----------------------------------------------------------------- 10.2 ---
TEST(Ch10Interpolation, ReproducesPolynomialsExactly) {
  const std::vector<double> a{1.0, -2.0, 0.5, 3.0};  // 1 - 2x + x^2/2 + 3x^3
  const std::vector<double> xs{-1.0, 0.0, 0.7, 2.0};
  std::vector<double> ys;
  for (double x : xs) ys.push_back(PolynomialValue(a, x));
  const auto c = DividedDifferences(xs, ys);
  const auto coeffs = InterpolatingPolynomial(xs, ys);
  for (std::size_t k = 0; k < a.size(); ++k)
    EXPECT_NEAR(coeffs[k], a[k], 1e-12);
  EXPECT_NEAR(c.back(), 3.0,
              1e-12);  // leading divided difference = leading coefficient
  for (double x : {-2.0, 0.3, 1.1, 5.0}) {
    EXPECT_NEAR(LagrangeInterpolate(xs, ys, x), PolynomialValue(a, x), 1e-10);
    EXPECT_NEAR(NewtonInterpolate(xs, c, x), PolynomialValue(a, x), 1e-10);
  }
}

TEST(Ch10Interpolation, LagrangeBasisIsKroneckerDelta) {
  const std::vector<double> xs{0.0, 1.0, 3.0, 4.5};
  for (std::size_t i = 0; i < xs.size(); ++i)
    for (std::size_t j = 0; j < xs.size(); ++j)
      EXPECT_NEAR(LagrangeBasis(xs, i, xs[j]), i == j ? 1.0 : 0.0, 1e-15);
  // The basis sums to 1 (it interpolates the constant 1).
  double s = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i) s += LagrangeBasis(xs, i, 2.2);
  EXPECT_NEAR(s, 1.0, 1e-14);
}

TEST(Ch10Interpolation, ErrorWithinBound) {
  // sin on 5 equally spaced nodes in [0, pi/2]; |sin^(5)| <= 1.
  std::vector<double> xs, ys;
  for (int i = 0; i <= 4; ++i) {
    xs.push_back(i * std::numbers::pi / 8);
    ys.push_back(std::sin(xs.back()));
  }
  for (double x : {0.1, 0.5, 1.0, 1.5}) {
    const double err = std::abs(std::sin(x) - LagrangeInterpolate(xs, ys, x));
    EXPECT_LE(err, InterpolationErrorBound(1.0, xs, x));
  }
}

TEST(Ch10Interpolation, RejectsBadNodes) {
  const std::vector<double> xs{0.0, 1.0, 1.0}, ys{1.0, 2.0, 3.0};
  EXPECT_THROW((void)LagrangeInterpolate(xs, ys, 0.5), std::invalid_argument);
  EXPECT_THROW((void)DividedDifferences(std::vector<double>{0.0, 1.0}, ys),
               std::invalid_argument);
}
