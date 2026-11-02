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
  EXPECT_EQ(norm2(ex05_F(root2)), 0.0);  // exact root
  for (auto scheme : {Difference::Central, Difference::Forward}) {
    const auto a = ex05_solve({1, 2, 3}, scheme);
    const auto b = ex05_solve({2, 2, 2}, scheme);
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
    EXPECT_LE(norm2(ex05_F(a.x)), 1e-9);
    EXPECT_LE(norm2(ex05_F(b.x)), 1e-9);
    EXPECT_EQ(a.iterations, 9u);
    EXPECT_GT(b.iterations, 20u);  // the long excursion from (2, 2, 2)
  }
}

TEST(Ch05Ex05, CentralJacobianIsSecondOrder) {
  const Vector x{1.0, 2.0, 3.0};
  const Matrix J = ex05_jacobian(x);
  double err_c = 0.0, err_f = 0.0;
  const Matrix Jc =
      finite_difference_jacobian(ex05_F, x, 1e-4, Difference::Central);
  const Matrix Jf =
      finite_difference_jacobian(ex05_F, x, 1e-4, Difference::Forward);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      err_c = std::max(err_c, std::abs(Jc(i, j) - J(i, j)));
      err_f = std::max(err_f, std::abs(Jf(i, j) - J(i, j)));
    }
  EXPECT_LT(err_c, 1e-6);  // O(h^2)
  EXPECT_GT(err_f, 1e-4);  // O(h)
  EXPECT_LT(err_f, 1e-2);
}

TEST(Ch05Ex06, UniqueDividendYield) {
  const auto res = ex06_solve();
  EXPECT_NEAR(res.x, 0.066906, 5e-7);
  EXPECT_LE(res.iterations, 6u);
  EXPECT_NEAR(ex06_delta(res.x), 0.5, 1e-12);
  // Delta is strictly decreasing in q, above 1/2 at q = 0, below at r +
  // sigma^2/2.
  EXPECT_GT(ex06_delta(0.0), 0.5);
  EXPECT_LT(ex06_delta(ex06_upper_bound()), 0.5);
  EXPECT_LT(res.x, ex06_upper_bound());
  for (double q = 0.0; q < 1.0; q += 0.01) EXPECT_LT(ex06_delta_dq(q), 0.0);
  // Derivative by central differences.
  const double h = 1e-6;
  EXPECT_NEAR(ex06_delta_dq(0.05),
              (ex06_delta(0.05 + h) - ex06_delta(0.05 - h)) / (2 * h), 1e-8);
  // Same root from another start.
  EXPECT_NEAR(ex06_solve(0.2).x, res.x, 1e-12);
}

TEST(Ch05Ex07, ContinuousCurveReprices) {
  const Ex07Quotes q;
  const auto c = ex07_curve_continuous(q);
  const double expected[] = {0.05234096, 0.05063562, 0.04728926,
                             0.04758188, 0.04630336, 0.04677207};
  ASSERT_EQ(c.rates.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(c.rates[i], expected[i], 5e-9);
  for (const auto& bq : q.quotes) {
    const auto cf = ch02::bond_cash_flows(bq.maturity, bq.coupon_rate, 2);
    EXPECT_NEAR(ch02::bond_price_from_zero_rate(
                    cf, [&](double t) { return zero_rate_linear(c, t); }),
                bq.price, 1e-9);
  }
  // The generic compounded bootstrap with m = 0 is the chapter 5 one.
  const auto c0 = ex07_bootstrap_compounded(q.quotes, 0);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(c0.rates[i], c.rates[i], 1e-12);
}

TEST(Ch05Ex07, SemiannualCurveIsHigher) {
  const Ex07Quotes q;
  const auto c = ex07_curve_continuous(q);
  const auto s = ex07_curve_semiannual(q);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_GT(s.rates[i], c.rates[i]);
    EXPECT_NEAR(s.rates[i] - c.rates[i], 0.25 * c.rates[i] * c.rates[i],
                2e-5);  // ~ r^2/4
    // Nearly the converted continuous curve (exact at the bills).
    EXPECT_NEAR(s.rates[i], ex07_to_compounded(c.rates[i], 2), 1e-7);
  }
  EXPECT_NEAR(s.rates[0], ex07_to_compounded(c.rates[0], 2), 1e-14);
  EXPECT_NEAR(s.rates[1], ex07_to_compounded(c.rates[1], 2), 1e-14);
  // The semiannual curve reprices every bond with semiannual discounting.
  for (const auto& bq : q.quotes) {
    const auto cf = ch02::bond_cash_flows(bq.maturity, bq.coupon_rate, 2);
    double B = 0.0;
    for (std::size_t i = 0; i < cf.times.size(); ++i)
      B += cf.amounts[i] *
           ex07_discount(zero_rate_linear(s, cf.times[i]), cf.times[i], 2);
    EXPECT_NEAR(B, bq.price, 1e-9);
  }
}

TEST(Ch05Ex08, OvernightAndHint) {
  const auto c = ex08_curve();
  const double expected[] = {0.05,       0.05063562, 0.04936960,
                             0.05298265, 0.04532568, 0.03211897};
  ASSERT_EQ(c.rates.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_NEAR(c.rates[i], expected[i], 5e-9);
  EXPECT_NEAR(ex08_hint_residual(c), 0.0, 1e-9);
  // r(0, 7/6) is the hint's (3 r(0, 1) + r(0, 5/3)) / 4, not the misprinted 4
  // r(0, 5/3).
  EXPECT_NEAR(zero_rate_linear(c, 7.0 / 6.0), (3 * c.rates[2] + c.rates[3]) / 4,
              1e-15);
  for (const auto& bq : ex08_quotes()) {
    const auto cf = ch02::bond_cash_flows(bq.maturity, bq.coupon_rate, 2);
    EXPECT_NEAR(ch02::bond_price_from_zero_rate(
                    cf, [&](double t) { return zero_rate_linear(c, t); }),
                bq.price, 1e-9);
  }
}

TEST(Ch05Ex09, ThreeYearPricingError) {
  const auto c = ex09_curve();
  const double expected[] = {0.01089965, 0.01390016, 0.01209855, 0.02682389,
                             0.03737166};
  ASSERT_EQ(c.rates.size(), 5u);
  for (std::size_t i = 0; i < 5; ++i)
    EXPECT_NEAR(c.rates[i], expected[i], 5e-9);
  // Bills: r = -ln(P/100)/T exactly.
  EXPECT_NEAR(c.rates[0], -std::log(0.994565) / 0.5, 1e-14);
  EXPECT_NEAR(c.rates[1], -std::log(0.986196), 1e-14);
  EXPECT_NEAR(ex09_model_price(c), 108.193042, 5e-7);
  EXPECT_NEAR(ex09_relative_error(c), 0.00586210, 5e-9);
  // Hand computation of the 3-year price from the curve.
  double B = 0.0;
  for (int k = 1; k <= 6; ++k) {
    const double t = 0.5 * k;
    B += (k == 6 ? 102.25 : 2.25) * std::exp(-zero_rate_linear(c, t) * t);
  }
  EXPECT_NEAR(ex09_model_price(c), B, 1e-12);
}
