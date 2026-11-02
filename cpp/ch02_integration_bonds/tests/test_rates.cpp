#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/rates.hpp>

using namespace p4mfe::ch02;

namespace {
// An increasing instantaneous rate curve with a closed-form integral:
//   r(t) = 0.0525 + ln(1 + 2t) / 200
//   integral_0^t r = 0.0525 t + ((1 + 2t) ln(1 + 2t) - 2t) / 400
double r_inst(double t) { return 0.0525 + std::log1p(2 * t) / 200; }
double r_int(double t) {
  return 0.0525 * t + ((1 + 2 * t) * std::log1p(2 * t) - 2 * t) / 400;
}
double r_zero(double t) { return r_int(t) / t; }
}  // namespace

TEST(Ch02Rates, ConstantRate) {
  EXPECT_DOUBLE_EQ(discount_factor(0.05, 2.0), std::exp(-0.1));
  EXPECT_NEAR(zero_rate_from_discount(discount_factor(0.05, 2.0), 2.0), 0.05,
              1e-15);
  EXPECT_NEAR(discount_factor_from_inst_rate([](double) { return 0.05; }, 2.0),
              std::exp(-0.1), 1e-14);
}

TEST(Ch02Rates, CurveRelations) {
  for (double t : {0.25, 1.0, 3.5}) {
    EXPECT_NEAR(discount_factor_from_inst_rate(r_inst, t), std::exp(-r_int(t)),
                1e-12)
        << t;
    EXPECT_NEAR(zero_rate_from_inst_rate(r_inst, t), r_zero(t), 1e-12) << t;
    EXPECT_NEAR(inst_rate_from_zero_rate(r_zero, t), r_inst(t), 1e-8) << t;
  }
  EXPECT_DOUBLE_EQ(discount_factor_from_inst_rate(r_inst, 0.0), 1.0);
}

TEST(Ch02Rates, ForwardRatesAgree) {
  const double t1 = 0.5, t2 = 2.0;
  const double f = forward_rate(r_zero(t1), t1, r_zero(t2), t2);
  EXPECT_NEAR(forward_rate_from_discounts(std::exp(-r_int(t1)), t1,
                                          std::exp(-r_int(t2)), t2),
              f, 1e-13);
  EXPECT_NEAR(forward_rate_from_inst_rate(r_inst, t1, t2), f, 1e-12);
  // Increasing r(t): the forward rate lies above the zero rate to t2.
  EXPECT_GT(f, r_zero(t2));
  // No arbitrage: invest to t1 then roll at the forward rate = invest to t2.
  EXPECT_NEAR(std::exp(r_zero(t1) * t1) * std::exp(f * (t2 - t1)),
              std::exp(r_zero(t2) * t2), 1e-13);
  // Flat curve: forward rate equals the rate.
  EXPECT_NEAR(forward_rate(0.04, 1, 0.04, 3), 0.04, 1e-15);
  EXPECT_THROW((void)forward_rate(0.04, 3, 0.04, 1), std::invalid_argument);
}

TEST(Ch02Discrete, CompoundingEquivalence) {
  const double r2 = 0.06;  // semiannual
  const double r = continuous_from_discrete(r2, 2);
  EXPECT_NEAR(r, 2 * std::log(1.03), 1e-15);
  EXPECT_NEAR(discrete_from_continuous(r, 2), r2, 1e-15);
  for (double t : {0.5, 1.0, 7.25})
    EXPECT_NEAR(discount_factor_discrete(r2, t, 2), discount_factor(r, t),
                1e-15);
  // More frequent compounding needs a lower quoted rate for the same growth.
  EXPECT_GT(discrete_from_continuous(0.05, 1),
            discrete_from_continuous(0.05, 12));
  EXPECT_GT(discrete_from_continuous(0.05, 12), 0.05);
}
