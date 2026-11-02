#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <p4mfe/ch10/exercises/ex01_parity.hpp>
#include <p4mfe/ch10/exercises/ex02_lhopital_sums.hpp>
#include <p4mfe/ch10/exercises/ex03_sum_k4.hpp>
#include <p4mfe/ch10/exercises/ex04_sum_squares_recursion.hpp>
#include <p4mfe/ch10/parity.hpp>
#include <p4mfe/ch10/sums.hpp>

using namespace p4mfe::ch10;
using namespace p4mfe::ch10::ex;

namespace {
const std::array<double, 7> kXs{0.1, 0.37, 0.8, 1.0, 1.3, 1.9, 2.4};
// Odd functions to test with.
double FSin(double x) { return std::sin(x); }
double FCubic(double x) { return x * x * x - 2.0 * x; }
double FAtan(double x) { return std::atan(3.0 * x); }
}  // namespace

TEST(Ch10Ex01, PartsOneAndTwo) {
  for (auto f : {FSin, FCubic, FAtan}) {
    EXPECT_TRUE(IsOdd(f, kXs));
    EXPECT_TRUE(IsEven([&](double x) { return x * f(x); }, kXs));
    EXPECT_TRUE(IsOdd([&](double x) { return x * x * f(x); }, kXs));
    EXPECT_TRUE(IsEven([&](double x) { return f(x * x); }, kXs));
    EXPECT_TRUE(IsOdd([&](double x) { return f(x * x * x); }, kXs));
  }
}

TEST(Ch10Ex01, GeneralParityRule) {
  for (auto f : {FSin, FCubic, FAtan})
    for (unsigned i = 1; i <= 4; ++i)
      for (unsigned j = 1; j <= 4; ++j) {
        const auto h = [&](double x) { return Ex01H(f, i, j, x); };
        EXPECT_EQ(ClassifyParity(h, kXs, 1e-10), Ex01PredictedParity(i, j))
            << i << ' ' << j;
      }
}

TEST(Ch10Ex02, NumeratorsMatchTheBook) {
  for (unsigned n = 3; n <= 30; ++n) {  // n >= 3 so that no exponents coincide
    EXPECT_EQ(Ex02Numerator(n, 2), Ex02BookN2(n)) << n;
    EXPECT_EQ(Ex02Numerator(n, 3), Ex02BookN3(n)) << n;
  }
}

TEST(Ch10Ex02, ClosedFormsAgreeWithDirectSums) {
  for (unsigned n : {1u, 2u, 5u, 12u})
    for (double x : {-0.7, 0.3, 0.9, 1.1, 2.0}) {
      for (unsigned j = 0; j <= 3; ++j) {
        const double direct = SumKjXk(n, j, x);
        EXPECT_NEAR(Ex02TClosedForm(n, j, x), direct,
                    1e-9 * std::max(1.0, std::abs(direct)))
            << n << ' ' << j << ' ' << x;
      }
      EXPECT_NEAR(Ex02TClosedForm(n, 2, x), SumK2Xk(n, x),
                  1e-9 * std::max(1.0, std::abs(SumK2Xk(n, x))));
    }
}

TEST(Ch10Ex02, LHopitalGivesTheSums) {
  for (unsigned n = 1; n <= 200; ++n) {
    // The numerators vanish to order j + 1 at x = 1, so l'Hopital applies.
    for (unsigned j = 2; j <= 3; ++j) {
      const IntPoly numer = Ex02Numerator(n, j);
      for (unsigned r = 0; r <= j; ++r)
        EXPECT_EQ(Ex02DerivativeAtOne(numer, r), 0)
            << n << ' ' << j << ' ' << r;
    }
    const std::int64_t nd = n;
    EXPECT_EQ(Ex02DerivativeAtOne(Ex02Numerator(n, 2), 3),
              -nd * (nd + 1) * (2 * nd + 1));
    EXPECT_EQ(Ex02SumByLhopital(n, 2), static_cast<std::int64_t>(SumK2(n)));
    EXPECT_EQ(Ex02SumByLhopital(n, 3), static_cast<std::int64_t>(SumK3(n)));
  }
  // The recipe also gives S(n, 1) and S(n, 4).
  for (unsigned n = 1; n <= 50; ++n) {
    EXPECT_EQ(Ex02SumByLhopital(n, 1), static_cast<std::int64_t>(SumK(n)));
    EXPECT_EQ(Ex02SumByLhopital(n, 4),
              static_cast<std::int64_t>(SumOfPowers(n, 4)));
  }
}

TEST(Ch10Ex03, SumOfFourthPowers) {
  for (std::uint64_t n = 0; n <= 4000; ++n) {
    const std::uint64_t direct =
        n <= 300 ? SumOfPowers(n, 4) : Ex03SumK4Recursive(n);
    ASSERT_EQ(Ex03SumK4(n), direct) << n;
  }
  for (std::uint64_t n = 0; n <= 300; ++n)
    ASSERT_EQ(Ex03SumK4Recursive(n), SumOfPowers(n, 4)) << n;
  EXPECT_EQ(Ex03SumK4(1), 1u);
  EXPECT_EQ(Ex03SumK4(2), 17u);
  EXPECT_EQ(Ex03SumK4(10), 25333u);
  // n^5/5 + n^4/2 + n^3/3 - n/30, from chapter 10's interpolation.
  const Vector c = SumOfPowersPolynomial(4);
  const std::array<double, 6> expected{0.0,       -1.0 / 30.0, 0.0,
                                       1.0 / 3.0, 0.5,         0.2};
  for (std::size_t m = 0; m < 6; ++m) EXPECT_NEAR(c[m], expected[m], 1e-9) << m;
}

TEST(Ch10Ex04, RecursionsReproduceSumOfSquares) {
  const auto x = Ex04Terms(60);
  const auto y = Ex04Terms1093(60);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::uint64_t n = i + 1;
    EXPECT_EQ(x[i], static_cast<double>(SumK2(n)))
        << n;  // integers below 2^53: exact
    EXPECT_EQ(y[i], static_cast<double>(SumK2(n))) << n;
  }
  // The cubic through (1, 1), (2, 5), (3, 14), (4, 30) is n(n+1)(2n+1)/6.
  const Vector c = Ex04Cubic();
  const std::array<double, 4> expected{0.0, 1.0 / 6.0, 0.5, 1.0 / 3.0};
  for (std::size_t m = 0; m < 4; ++m)
    EXPECT_NEAR(c[m], expected[m], 1e-12) << m;
  // Characteristic polynomial (z - 1)^4: coefficients 1, -4, 6, -4, 1.
  const std::array<double, 5> char_poly{kEx04Recursion[0], kEx04Recursion[1],
                                        kEx04Recursion[2], kEx04Recursion[3],
                                        -1.0};
  for (double z : {-1.5, 0.2, 2.0, 3.0})
    EXPECT_NEAR(-PolynomialValue(char_poly, z), std::pow(z - 1.0, 4), 1e-12)
        << z;
}
