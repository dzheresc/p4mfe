#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/exercises/ex08_09_rate_curves.hpp>
#include <p4mfe/ch02/exercises/ex10_13_bond_pricing.hpp>

using namespace p4mfe::ch02;
using namespace p4mfe::ch02::ex;

TEST(Ch02Ex08, ParallelShift) {
  const auto zero1 = [](double t) {
    return 0.03 + 0.02 * (1 - std::exp(-t)) / (1 + t) + 0.004 * std::sin(t);
  };
  const double dr = 0.0075;
  const auto zero2 = Ex08Shift(zero1, dr);
  for (double t : {0.25, 1.0, 3.0, 10.0}) {
    // zero shift => instantaneous shift
    EXPECT_NEAR(InstRateFromZeroRate(zero2, t) - InstRateFromZeroRate(zero1, t),
                dr, 1e-9)
        << t;
  }
  // instantaneous shift => zero shift
  const auto r1 = [](double t) {
    return 0.04 + 0.01 * std::log1p(t) - 0.002 * t * std::exp(-t);
  };
  const auto r2 = [&](double t) { return r1(t) + dr; };
  for (double t : {0.25, 1.0, 3.0, 10.0})
    EXPECT_NEAR(ZeroRateFromInstRate(r2, t) - ZeroRateFromInstRate(r1, t), dr,
                1e-12)
        << t;
}

TEST(Ch02Ex09, InstantaneousAndDiscreteCurves) {
  for (double t : {0.5, 1.0, 2.0, 5.0, 30.0}) {
    EXPECT_NEAR(Ex09InstRate(t), InstRateFromZeroRate(Ex09ZeroRate, t), 1e-9)
        << t;
    EXPECT_NEAR(Ex09ZeroRate(t), ZeroRateFromInstRate(Ex09InstRate, t), 1e-12)
        << t;
    // Same discount factor under all three compounding conventions.
    const double discount = std::exp(-Ex09ZeroRate(t) * t);
    EXPECT_NEAR(DiscountFactorDiscrete(Ex09AnnualZeroRate(t), t, 1), discount,
                1e-14)
        << t;
    EXPECT_NEAR(DiscountFactorDiscrete(Ex09SemiannualZeroRate(t), t, 2),
                discount, 1e-14)
        << t;
    EXPECT_NEAR(Ex09AnnualZeroRate(t),
                DiscreteFromContinuous(Ex09ZeroRate(t), 1), 1e-15);
    EXPECT_NEAR(Ex09SemiannualZeroRate(t),
                DiscreteFromContinuous(Ex09ZeroRate(t), 2), 1e-15);
  }
  EXPECT_NEAR(Ex09AnnualZeroRate(0), std::exp(0.05) - 1, 1e-15);
  EXPECT_NEAR(Ex09SemiannualZeroRate(0), 2 * (std::exp(0.025) - 1), 1e-15);
}

TEST(Ch02Ex10, PriceFromZeroRates) {
  const auto d = Ex10DiscountFactors();
  EXPECT_NEAR(d[0], 0.975310, 1e-6);
  EXPECT_NEAR(d[1], 0.948854, 1e-6);
  EXPECT_NEAR(d[2], 0.922886, 1e-6);
  EXPECT_NEAR(d[3], 0.895834, 1e-6);
  EXPECT_NEAR(Ex10Price(), 98.9406, 1e-4);
  EXPECT_NEAR(Ex10Price(), 2.5 * (d[0] + d[1] + d[2] + d[3]) + 100 * d[3],
              1e-12);
}

TEST(Ch02Ex11, ParYield) {
  const double c = Ex11ParYield();
  EXPECT_NEAR(c, 0.055661, 1e-6);
  EXPECT_NEAR(Ex10Price(c), 100.0, 1e-10);
}

TEST(Ch02Ex12, ZeroCurveDiscountsAndPrice) {
  for (double t : {0.5, 1.0, 1.5, 2.0, 7.0}) {
    EXPECT_NEAR(Ex12ZeroRate(t), ZeroRateFromInstRate(Ex12InstRate, t), 1e-12)
        << t;
    EXPECT_NEAR(Ex12DiscountFactor(t),
                DiscountFactorFromInstRate(Ex12InstRate, t), 1e-12)
        << t;
  }
  EXPECT_NEAR(Ex12ZeroRate(1e-9), 0.05, 1e-11);
  EXPECT_NEAR(Ex12DiscountFactor(0.5), 0.974782, 1e-6);
  EXPECT_NEAR(Ex12DiscountFactor(1.0), 0.949394, 1e-6);
  EXPECT_NEAR(Ex12DiscountFactor(1.5), 0.924083, 1e-6);
  EXPECT_NEAR(Ex12DiscountFactor(2.0), 0.898994, 1e-6);
  EXPECT_NEAR(Ex12Price(), 99.2675, 1e-4);
  EXPECT_NEAR(Ex12Price(),
              BondPriceFromInstRate(BondCashFlows(2.0, 0.05, 2), Ex12InstRate),
              1e-10);
}

TEST(Ch02Ex13, SimpsonDiscountFactors) {
  EXPECT_EQ(Ex13Integral(1.0, 1e-6).intervals, 8u);
  EXPECT_EQ(Ex13Integral(2.0, 1e-6).intervals, 16u);
  EXPECT_EQ(Ex13Integral(3.0, 1e-8).intervals, 64u);
  for (double t : {1.0, 2.0, 3.0}) {
    const double exact = Ex13DiscountFactor(t, 1e-14);
    EXPECT_NEAR(Ex13DiscountFactor(t, Ex13Tol(t)), exact, Ex13Tol(t)) << t;
  }
  EXPECT_NEAR(Ex13DiscountFactor(1.0, 1e-6), 0.956595, 5e-7);
  EXPECT_NEAR(Ex13DiscountFactor(2.0, 1e-6), 0.910128, 5e-7);
  EXPECT_NEAR(Ex13DiscountFactor(3.0, 1e-8), 0.86574103, 5e-9);
  EXPECT_NEAR(Ex13Price(), 100.2364, 1e-4);
  EXPECT_NEAR(Ex13Price(),
              BondPriceFromInstRate(BondCashFlows(3.0, 0.05, 1), Ex13InstRate),
              1e-6);
}
