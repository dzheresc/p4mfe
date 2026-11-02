#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <p4mfe/ch05/exercises/ex05_central_difference_newton.hpp>
#include <p4mfe/ch05/exercises/ex06_atm_delta_dividend.hpp>
#include <p4mfe/ch05/exercises/ex07_treasury_bootstrap.hpp>
#include <p4mfe/ch05/exercises/ex08_bootstrap_overnight.hpp>
#include <p4mfe/ch05/exercises/ex09_bootstrap_pricing_error.hpp>

using namespace p4mfe;
using namespace p4mfe::ch05;
using namespace p4mfe::ch05::ex;

TEST(Ch05Ex05, RootsFromBothStarts) {
  const Vector root1{-1.690550760, 1.983107243, -0.884558078};
  const Vector root2{-1.0, 3.0, 1.0};
  EXPECT_EQ(Norm2(Ex05F(root2)), 0.0);  // exact root
  for (auto scheme : {Difference::kCentral, Difference::kForward}) {
    const auto a = Ex05Solve({1, 2, 3}, scheme);
    const auto b = Ex05Solve({2, 2, 2}, scheme);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(a.x[i], root1[i], 1e-8);
    // From (2, 2, 2) the iterates go through a chaotic phase, and which
    // root catches them depends on rounding (compiler, libm, BLAS): this
    // machine may land on either root.
    double d1 = 0.0, d2 = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
      d1 = std::max(d1, std::abs(b.x[i] - root1[i]));
      d2 = std::max(d2, std::abs(b.x[i] - root2[i]));
    }
    EXPECT_LT(std::min(d1, d2), 1e-8);
    EXPECT_LE(Norm2(Ex05F(a.x)), 1e-9);
    EXPECT_LE(Norm2(Ex05F(b.x)), 1e-9);
    EXPECT_EQ(a.iterations, 9u);
    EXPECT_GT(b.iterations, 20u);  // the long excursion from (2, 2, 2)
  }
}

TEST(Ch05Ex05, CentralJacobianIsSecondOrder) {
  const Vector x{1.0, 2.0, 3.0};
  const Matrix jacobian = Ex05Jacobian(x);
  double err_c = 0.0, err_f = 0.0;
  const Matrix jacobian_central =
      FiniteDifferenceJacobian(Ex05F, x, 1e-4, Difference::kCentral);
  const Matrix jacobian_forward =
      FiniteDifferenceJacobian(Ex05F, x, 1e-4, Difference::kForward);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      err_c =
          std::max(err_c, std::abs(jacobian_central(i, j) - jacobian(i, j)));
      err_f =
          std::max(err_f, std::abs(jacobian_forward(i, j) - jacobian(i, j)));
    }
  EXPECT_LT(err_c, 1e-6);  // O(h^2)
  EXPECT_GT(err_f, 1e-4);  // O(h)
  EXPECT_LT(err_f, 1e-2);
}

TEST(Ch05Ex06, UniqueDividendYield) {
  const auto res = Ex06Solve();
  EXPECT_NEAR(res.x, 0.066906, 5e-7);
  EXPECT_LE(res.iterations, 6u);
  EXPECT_NEAR(Ex06Delta(res.x), 0.5, 1e-12);
  // Delta is strictly decreasing in q, above 1/2 at q = 0, below at r +
  // sigma^2/2.
  EXPECT_GT(Ex06Delta(0.0), 0.5);
  EXPECT_LT(Ex06Delta(Ex06UpperBound()), 0.5);
  EXPECT_LT(res.x, Ex06UpperBound());
  for (double q = 0.0; q < 1.0; q += 0.01) EXPECT_LT(Ex06DeltaDq(q), 0.0);
  // Derivative by central differences.
  const double h = 1e-6;
  EXPECT_NEAR(Ex06DeltaDq(0.05),
              (Ex06Delta(0.05 + h) - Ex06Delta(0.05 - h)) / (2 * h), 1e-8);
  // Same root from another start.
  EXPECT_NEAR(Ex06Solve(0.2).x, res.x, 1e-12);
}

TEST(Ch05Ex07, ContinuousCurveReprices) {
  const Ex07Quotes q;
  const auto c = Ex07CurveContinuous(q);
  const double expected[] = {0.05234096, 0.05063562, 0.04728926,
                             0.04758188, 0.04630336, 0.04677207};
  ASSERT_EQ(c.rates.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(c.rates[i], expected[i], 5e-9);
  for (const auto& bq : q.quotes) {
    const auto cf = ch02::BondCashFlows(bq.maturity, bq.coupon_rate, 2);
    EXPECT_NEAR(ch02::BondPriceFromZeroRate(
                    cf, [&](double t) { return ZeroRateLinear(c, t); }),
                bq.price, 1e-9);
  }
  // The generic compounded bootstrap with m = 0 is the chapter 5 one.
  const auto c0 = Ex07BootstrapCompounded(q.quotes, 0);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(c0.rates[i], c.rates[i], 1e-12);
}

TEST(Ch05Ex07, SemiannualCurveIsHigher) {
  const Ex07Quotes q;
  const auto c = Ex07CurveContinuous(q);
  const auto s = Ex07CurveSemiannual(q);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_GT(s.rates[i], c.rates[i]);
    EXPECT_NEAR(s.rates[i] - c.rates[i], 0.25 * c.rates[i] * c.rates[i],
                2e-5);  // ~ r^2/4
    // Nearly the converted continuous curve (exact at the bills).
    EXPECT_NEAR(s.rates[i], Ex07ToCompounded(c.rates[i], 2), 1e-7);
  }
  EXPECT_NEAR(s.rates[0], Ex07ToCompounded(c.rates[0], 2), 1e-14);
  EXPECT_NEAR(s.rates[1], Ex07ToCompounded(c.rates[1], 2), 1e-14);
  // The semiannual curve reprices every bond with semiannual discounting.
  for (const auto& bq : q.quotes) {
    const auto cf = ch02::BondCashFlows(bq.maturity, bq.coupon_rate, 2);
    double price = 0.0;
    for (std::size_t i = 0; i < cf.times.size(); ++i)
      price += cf.amounts[i] *
               Ex07Discount(ZeroRateLinear(s, cf.times[i]), cf.times[i], 2);
    EXPECT_NEAR(price, bq.price, 1e-9);
  }
}

TEST(Ch05Ex08, OvernightAndHint) {
  const auto c = Ex08Curve();
  const double expected[] = {0.05,       0.05063562, 0.04936960,
                             0.05298265, 0.04532568, 0.03211897};
  ASSERT_EQ(c.rates.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(c.rates[i], expected[i], 5e-9);
  EXPECT_NEAR(Ex08HintResidual(c), 0.0, 1e-9);
  // r(0, 7/6) is the hint's (3 r(0, 1) + r(0, 5/3)) / 4, not the misprinted 4
  // r(0, 5/3).
  EXPECT_NEAR(ZeroRateLinear(c, 7.0 / 6.0), (3 * c.rates[2] + c.rates[3]) / 4,
              1e-15);
  for (const auto& bq : Ex08Quotes()) {
    const auto cf = ch02::BondCashFlows(bq.maturity, bq.coupon_rate, 2);
    EXPECT_NEAR(ch02::BondPriceFromZeroRate(
                    cf, [&](double t) { return ZeroRateLinear(c, t); }),
                bq.price, 1e-9);
  }
}

TEST(Ch05Ex09, ThreeYearPricingError) {
  const auto c = Ex09Curve();
  const double expected[] = {0.01089965, 0.01390016, 0.01209855, 0.02682389,
                             0.03737166};
  ASSERT_EQ(c.rates.size(), 5u);
  for (std::size_t i = 0; i < 5; ++i)
    EXPECT_NEAR(c.rates[i], expected[i], 5e-9);
  // Bills: r = -ln(P/100)/T exactly.
  EXPECT_NEAR(c.rates[0], -std::log(0.994565) / 0.5, 1e-14);
  EXPECT_NEAR(c.rates[1], -std::log(0.986196), 1e-14);
  EXPECT_NEAR(Ex09ModelPrice(c), 108.193042, 5e-7);
  EXPECT_NEAR(Ex09RelativeError(c), 0.00586210, 5e-9);
  // Hand computation of the 3-year price from the curve.
  double price = 0.0;
  for (int k = 1; k <= 6; ++k) {
    const double t = 0.5 * k;
    price += (k == 6 ? 102.25 : 2.25) * std::exp(-ZeroRateLinear(c, t) * t);
  }
  EXPECT_NEAR(Ex09ModelPrice(c), price, 1e-12);
}
