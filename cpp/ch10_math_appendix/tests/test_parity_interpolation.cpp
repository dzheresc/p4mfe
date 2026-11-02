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
const std::vector<double> pts{0.1, 0.5, 1.0, 1.7, 2.5, 4.0};
}

// ----------------------------------------------------------------- 10.1 ---
TEST(Ch10Parity, ClassifiesClassicalFunctions) {
  const auto cosf = [](double x) { return std::cos(x); };
  const auto sinf = [](double x) { return std::sin(x); };
  const auto expf = [](double x) { return std::exp(x); };
  EXPECT_EQ(parity(cosf, pts), Parity::Even);
  EXPECT_EQ(parity(sinf, pts), Parity::Odd);
  EXPECT_EQ(parity(expf, pts), Parity::Neither);
  EXPECT_EQ(parity([](double x) { return norm_pdf(x); }, pts), Parity::Even);
  // N(x) - 1/2 is odd, i.e. N(-x) = 1 - N(x).
  EXPECT_TRUE(is_odd([](double x) { return norm_cdf(x) - 0.5; }, pts, 1e-14));
}

TEST(Ch10Parity, ProductsAndDerivatives) {
  const auto sinf = [](double x) { return std::sin(x); };
  const auto cube = [](double x) { return x * x * x; };
  // odd * odd = even, even * odd = odd.
  EXPECT_TRUE(is_even([&](double x) { return sinf(x) * cube(x); }, pts));
  EXPECT_TRUE(is_odd([&](double x) { return std::cos(x) * cube(x); }, pts));
  // The derivative of the even function x^4 + cos x is odd.
  EXPECT_TRUE(
      is_odd([](double x) { return 4 * x * x * x - std::sin(x); }, pts));
}

TEST(Ch10Parity, EvenOddDecomposition) {
  const auto f = [](double x) { return std::exp(x); };
  for (double x : pts) {
    EXPECT_NEAR(even_part(f, x), std::cosh(x), 1e-13 * std::cosh(x));
    EXPECT_NEAR(odd_part(f, x), std::sinh(x), 1e-13 * std::cosh(x));
    EXPECT_DOUBLE_EQ(even_part(f, x) + odd_part(f, x), f(x));
  }
  const auto ep = [&](double x) { return even_part(f, x); };
  const auto op = [&](double x) { return odd_part(f, x); };
  EXPECT_TRUE(is_even(ep, pts));
  EXPECT_TRUE(is_odd(op, pts));
}

TEST(Ch10Parity, SymmetricIntegrals) {
  // int_{-a}^{a} e^x = 2 sinh a; odd part integrates to zero.
  const auto f = [](double x) { return std::exp(x) + x * x * x * std::cos(x); };
  EXPECT_NEAR(symmetric_integral(f, 1.5), 2.0 * std::sinh(1.5), 1e-10);
  EXPECT_NEAR(symmetric_integral(f, 1.5), ch02::Integrate(f, -1.5, 1.5, 400),
              1e-10);
  EXPECT_NEAR(symmetric_integral([](double x) { return std::sin(x); }, 3.0),
              0.0, 1e-15);
  // int_{-a}^{a} phi = 2 N(a) - 1.
  EXPECT_NEAR(symmetric_integral([](double x) { return norm_pdf(x); }, 2.0),
              2.0 * norm_cdf(2.0) - 1.0, 1e-10);
  EXPECT_THROW((void)symmetric_integral(f, -1.0), std::invalid_argument);
}

// ----------------------------------------------------------------- 10.2 ---
TEST(Ch10Interpolation, ReproducesPolynomialsExactly) {
  const std::vector<double> a{1.0, -2.0, 0.5, 3.0};  // 1 - 2x + x^2/2 + 3x^3
  const std::vector<double> xs{-1.0, 0.0, 0.7, 2.0};
  std::vector<double> ys;
  for (double x : xs) ys.push_back(polynomial_value(a, x));
  const auto c = divided_differences(xs, ys);
  const auto coeffs = interpolating_polynomial(xs, ys);
  for (std::size_t k = 0; k < a.size(); ++k)
    EXPECT_NEAR(coeffs[k], a[k], 1e-12);
  EXPECT_NEAR(c.back(), 3.0,
              1e-12);  // leading divided difference = leading coefficient
  for (double x : {-2.0, 0.3, 1.1, 5.0}) {
    EXPECT_NEAR(lagrange_interpolate(xs, ys, x), polynomial_value(a, x), 1e-10);
    EXPECT_NEAR(newton_interpolate(xs, c, x), polynomial_value(a, x), 1e-10);
  }
}

TEST(Ch10Interpolation, LagrangeBasisIsKroneckerDelta) {
  const std::vector<double> xs{0.0, 1.0, 3.0, 4.5};
  for (std::size_t i = 0; i < xs.size(); ++i)
    for (std::size_t j = 0; j < xs.size(); ++j)
      EXPECT_NEAR(lagrange_basis(xs, i, xs[j]), i == j ? 1.0 : 0.0, 1e-15);
  // The basis sums to 1 (it interpolates the constant 1).
  double s = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i) s += lagrange_basis(xs, i, 2.2);
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
    const double err = std::abs(std::sin(x) - lagrange_interpolate(xs, ys, x));
    EXPECT_LE(err, interpolation_error_bound(1.0, xs, x));
  }
}

TEST(Ch10Interpolation, RejectsBadNodes) {
  const std::vector<double> xs{0.0, 1.0, 1.0}, ys{1.0, 2.0, 3.0};
  EXPECT_THROW((void)lagrange_interpolate(xs, ys, 0.5), std::invalid_argument);
  EXPECT_THROW((void)divided_differences(std::vector<double>{0.0, 1.0}, ys),
               std::invalid_argument);
}
