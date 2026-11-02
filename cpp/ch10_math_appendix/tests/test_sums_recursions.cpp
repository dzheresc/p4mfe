#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch10/linear_recursions.hpp>
#include <p4mfe/ch10/sums.hpp>
#include <vector>

using namespace p4mfe::ch10;

// --------------------------------------------------------------- 10.3.1 ---
TEST(Ch10Sums, ClosedFormsOfSumsOfPowers) {
  static_assert(SumK(100) == 5050);
  for (std::uint64_t n : {1u, 2u, 7u, 50u, 1000u}) {
    EXPECT_EQ(SumK(n), SumOfPowers(n, 1));
    EXPECT_EQ(SumK2(n), SumOfPowers(n, 2));
    EXPECT_EQ(SumK3(n), SumOfPowers(n, 3));
  }
}

TEST(Ch10Sums, TelescopingRecursionMatchesDirectSums) {
  for (std::uint64_t n : {1u, 5u, 20u, 100u}) {
    const auto sums = SumsOfPowersRecursive(n, 6);
    for (unsigned i = 0; i <= 6; ++i)
      EXPECT_EQ(sums[i], SumOfPowers(n, i)) << n << ' ' << i;
  }
}

TEST(Ch10Sums, SumOfPowersPolynomialByInterpolation) {
  const auto p2 = SumOfPowersPolynomial(2);  // n/6 + n^2/2 + n^3/3
  ASSERT_EQ(p2.size(), 4u);
  EXPECT_NEAR(p2[0], 0.0, 1e-12);
  EXPECT_NEAR(p2[1], 1.0 / 6, 1e-12);
  EXPECT_NEAR(p2[2], 1.0 / 2, 1e-12);
  EXPECT_NEAR(p2[3], 1.0 / 3, 1e-12);
  // Leading coefficient 1/(i+1), next 1/2 (Faulhaber); value off the nodes.
  for (unsigned i = 1; i <= 6; ++i) {
    const auto p = SumOfPowersPolynomial(i);
    EXPECT_NEAR(p.back(), 1.0 / (i + 1), 1e-9) << i;
    EXPECT_NEAR(p[i], 0.5, 1e-9) << i;
    EXPECT_NEAR(PolynomialValue(p, 30.0),
                static_cast<double>(SumOfPowers(30, i)),
                1e-9 * static_cast<double>(SumOfPowers(30, i)));
  }
}

// --------------------------------------------------------------- 10.3.2 ---
TEST(Ch10Sums, FiniteSumsOfKjXk) {
  for (unsigned n : {1u, 4u, 25u})
    for (double x : {-0.8, 0.3, 0.95, 1.0, 1.7}) {
      const double tol = 1e-12 * std::max(1.0, std::abs(SumKjXk(n, 2, x)));
      EXPECT_NEAR(GeometricSum(n, x), SumKjXk(n, 0, x), tol) << n << ' ' << x;
      EXPECT_NEAR(SumKXk(n, x), SumKjXk(n, 1, x), tol) << n << ' ' << x;
      EXPECT_NEAR(SumK2Xk(n, x), SumKjXk(n, 2, x), tol * 10) << n << ' ' << x;
    }
}

TEST(Ch10Sums, InfiniteSumsOfKjXk) {
  EXPECT_EQ(KjXkNumerator(0), (Vector{0.0, 1.0}));
  EXPECT_EQ(KjXkNumerator(2), (Vector{0.0, 1.0, 1.0}));
  EXPECT_EQ(KjXkNumerator(3), (Vector{0.0, 1.0, 4.0, 1.0}));
  EXPECT_EQ(KjXkNumerator(4), (Vector{0.0, 1.0, 11.0, 11.0, 1.0}));
  for (double x : {-0.5, 0.2, 0.9}) {
    EXPECT_NEAR(SumKjXkInfinite(1, x), x / ((1 - x) * (1 - x)), 1e-13);
    EXPECT_NEAR(SumKjXkInfinite(2, x), x * (1 + x) / std::pow(1 - x, 3), 1e-12);
    for (unsigned j = 0; j <= 5; ++j) {
      const double v = SumKjXkInfinite(j, x);
      EXPECT_NEAR(SumKjXk(2000, j, x), v, 1e-10 * std::abs(v)) << j << ' ' << x;
    }
  }
  EXPECT_THROW((void)SumKjXkInfinite(1, 1.0), std::domain_error);
}

TEST(Ch10Sums, PerpetuityDurationFromSums) {
  // Cash flows of 1 at t = 1, 2, ...; discount x = e^(-y). Macaulay duration
  // sum k x^k / sum x^k = 1 / (1 - x).
  const double y = 0.05, x = std::exp(-y);
  EXPECT_NEAR(SumKjXkInfinite(1, x) / SumKjXkInfinite(0, x), 1.0 / (1.0 - x),
              1e-10);
}

// ----------------------------------------------------------------- 10.4 ---
TEST(Ch10Recursions, FibonacciClosedForm) {
  const auto fib = RecursionTerms(std::vector<double>{1.0, 1.0},
                                  std::vector<double>{0.0, 1.0}, 40);
  EXPECT_EQ(fib[10], 55.0);
  const auto r = SolveSecondOrder(1.0, 1.0, 0.0, 1.0);
  EXPECT_FALSE(r.double_root);
  EXPECT_NEAR(r.lambda1.real(), std::numbers::phi, 1e-15);
  for (unsigned n = 0; n < 40; ++n)
    EXPECT_NEAR(r(n), fib[n], 1e-6 * std::max(1.0, fib[n])) << n;
}

TEST(Ch10Recursions, DoubleAndComplexRoots) {
  // x_{n+2} = 4 x_{n+1} - 4 x_n: lambda = 2 double, x_n = (c1 + c2 n) 2^n.
  const auto d = SolveSecondOrder(4.0, -4.0, 1.0, 4.0);
  EXPECT_TRUE(d.double_root);
  const auto td = RecursionTerms(std::vector<double>{-4.0, 4.0},
                                 std::vector<double>{1.0, 4.0}, 20);
  for (unsigned n = 0; n < 20; ++n)
    EXPECT_NEAR(d(n), td[n], 1e-9 * std::max(1.0, std::abs(td[n])));
  // x_{n+2} = x_{n+1} - x_n: roots e^(+-i pi/3), period 6.
  const auto c = SolveSecondOrder(1.0, -1.0, 1.0, 2.0);
  const auto tc = RecursionTerms(std::vector<double>{-1.0, 1.0},
                                 std::vector<double>{1.0, 2.0}, 30);
  for (unsigned n = 0; n < 30; ++n) EXPECT_NEAR(c(n), tc[n], 1e-12);
  EXPECT_NEAR(tc[6], tc[0], 0.0);
  EXPECT_NEAR(std::abs(c.lambda1), 1.0, 1e-15);
}

TEST(Ch10Recursions, ThirdOrderAgainstCharacteristicRoots) {
  // x_{n+3} = 6 x_{n+2} - 11 x_{n+1} + 6 x_n: roots 1, 2, 3, so
  // x_n = 1 + 2^n + 3^n with x_0 = 3, x_1 = 6, x_2 = 14.
  const auto t = RecursionTerms(std::vector<double>{6.0, -11.0, 6.0},
                                std::vector<double>{3.0, 6.0, 14.0}, 15);
  for (unsigned n = 0; n < 15; ++n)
    EXPECT_EQ(t[n], 1.0 + std::pow(2.0, n) + std::pow(3.0, n));
}

TEST(Ch10Recursions, AffineRecursionAndLoanAmortization) {
  double x = 5.0;
  for (unsigned n = 0; n < 12; ++n, x = 0.7 * x + 3.0)
    EXPECT_NEAR(FirstOrderAffine(0.7, 3.0, 5.0, n), x, 1e-12);
  EXPECT_EQ(FirstOrderAffine(1.0, 2.0, 1.0, 5), 11.0);
  // A 30-year monthly mortgage pays off exactly.
  const double principal = 300000.0, r = 0.06 / 12;
  const unsigned num_periods = 360;
  const double payment = LoanPayment(principal, r, num_periods);
  EXPECT_NEAR(payment, 1798.65, 0.01);
  EXPECT_NEAR(FirstOrderAffine(1.0 + r, -payment, principal, num_periods), 0.0,
              1e-6);
}
