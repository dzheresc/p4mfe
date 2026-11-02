#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>
#include <p4mfe/ch05/bond_yield.hpp>
#include <p4mfe/ch05/bootstrapping.hpp>
#include <p4mfe/ch05/implied_vol.hpp>
#include <stdexcept>

using namespace p4mfe;
using namespace p4mfe::ch05;

// ------------------------------------------------------------------ 5.3 ---
TEST(Ch05BondYield, ThreeMethodsAgreeWithChapter2) {
  const auto cf = ch02::BondCashFlows(3.0, 0.06, 2);
  for (double price : {90.0, 101.0, 105.0}) {
    const double y2 = ch02::BondYield(cf, price);
    const auto n = BondYieldNewton(cf, price);
    const auto s = BondYieldSecant(cf, price);
    const auto b = BondYieldBisection(cf, price);
    EXPECT_NEAR(n.x, y2, 1e-12);
    EXPECT_NEAR(s.x, y2, 1e-12);
    EXPECT_NEAR(b.x, y2, 1e-11);
    EXPECT_NEAR(ch02::BondPrice(cf, n.x), price, 1e-10);
    EXPECT_LE(n.iterations, 6u);
    EXPECT_LT(n.iterations, b.iterations);
  }
}

TEST(Ch05BondYield, RoundTripLongBond) {
  const auto cf = ch02::BondCashFlows(29.75, 0.045, 2);  // between coupon dates
  for (double y : {-0.005, 0.02, 0.12}) {
    const double price = ch02::BondPrice(cf, y);
    EXPECT_NEAR(BondYieldNewton(cf, price).x, y, 1e-12);
  }
}

// ------------------------------------------------------------------ 5.4 ---
TEST(Ch05ImpliedVol, RecoversSigmaAndAgreesWithChapter3) {
  const double spot = 30, maturity = 0.5, r = 0.03, q = 0.01;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    for (double strike : {25.0, 30.0, 36.0}) {
      for (double sigma : {0.15, 0.3, 0.8}) {
        const double price =
            ch03::BsPrice(type, spot, strike, maturity, sigma, r, q);
        const auto n =
            ImpliedVolNewton(type, price, spot, strike, maturity, r, q);
        const auto s =
            ImpliedVolSecant(type, price, spot, strike, maturity, r, q);
        const auto b =
            ImpliedVolBisection(type, price, spot, strike, maturity, r, q);
        EXPECT_NEAR(n.x, sigma, 1e-9) << strike << ' ' << sigma;
        EXPECT_NEAR(s.x, sigma, 1e-9) << strike << ' ' << sigma;
        EXPECT_NEAR(b.x, sigma, 1e-8) << strike << ' ' << sigma;
        EXPECT_NEAR(
            n.x,
            ch03::ImpliedVolNewton(type, price, spot, strike, maturity, r, q)
                .sigma,
            1e-9);
      }
    }
  }
}

TEST(Ch05ImpliedVol, NewtonFromInflectionPointIsMonotone) {
  // From sigma_0 the Newton iterates approach the root from one side.
  const double spot = 40, strike = 44, maturity = 0.75, r = 0.04, q = 0.0,
               sigma = 0.27;
  const double price = ch03::BsCall(spot, strike, maturity, sigma, r, q);
  const auto res = ImpliedVolNewton(OptionType::kCall, price, spot, strike,
                                    maturity, r, q, {1e-14, 1e-14, 100});
  const double side = res.iterates.front() - sigma;
  for (double x : res.iterates) {
    if (std::abs(x - sigma) > 1e-14) {
      EXPECT_GT((x - sigma) * side, 0.0);
    }
  }
}

TEST(Ch05ImpliedVol, RejectsArbitrageablePrices) {
  EXPECT_THROW(
      (void)ImpliedVolNewton(OptionType::kCall, 31.0, 30, 30, 0.5, 0.03),
      std::domain_error);
  EXPECT_THROW((void)ImpliedVolSecant(OptionType::kPut, 0.0, 30, 30, 0.5, 0.03),
               std::domain_error);
}

// ------------------------------------------------------------------ 5.5 ---
TEST(Ch05Bootstrapping, LinearInterpolation) {
  const ZeroCurve c{{0.0, 1.0, 3.0}, {0.05, 0.06, 0.07}};
  EXPECT_DOUBLE_EQ(ZeroRateLinear(c, 0.5), 0.055);
  EXPECT_DOUBLE_EQ(ZeroRateLinear(c, 2.0), 0.065);
  EXPECT_DOUBLE_EQ(ZeroRateLinear(c, 3.0), 0.07);
  EXPECT_DOUBLE_EQ(ZeroRateLinear(c, 4.0), 0.07);   // flat beyond
  EXPECT_DOUBLE_EQ(ZeroRateLinear(c, -1.0), 0.05);  // and before
}

TEST(Ch05Bootstrapping, RecoversPiecewiseLinearCurve) {
  // A curve linear between 0, 0.5, 1, 3, 5 years; bonds at 0.5, 1, 3, 5 years
  // leave coupon dates 1.5, 2, 2.5, 3.5, 4, 4.5 to interpolation.
  const ZeroCurve truth{{0.0, 0.5, 1.0, 3.0, 5.0},
                        {0.05, 0.051, 0.0525, 0.058, 0.061}};
  std::vector<BondQuote> quotes;
  const double coupons[] = {0.0, 0.05, 0.055, 0.06};
  for (int j = 0; j < 4; ++j) {
    const double maturity = truth.times[static_cast<std::size_t>(j) + 1];
    const auto cf = ch02::BondCashFlows(maturity, coupons[j], 2);
    quotes.push_back(
        {maturity, coupons[j], ch02::BondPriceFromZeroRate(cf, [&](double t) {
           return ZeroRateLinear(truth, t);
         })});
  }
  const ZeroCurve overnight{{0.0}, {0.05}};
  const auto curve = BootstrapZeroCurve(quotes, 2, overnight);
  ASSERT_EQ(curve.times.size(), truth.times.size());
  for (std::size_t j = 0; j < truth.times.size(); ++j) {
    EXPECT_DOUBLE_EQ(curve.times[j], truth.times[j]);
    EXPECT_NEAR(curve.rates[j], truth.rates[j], 1e-12);
  }
}

TEST(Ch05Bootstrapping, AgreesWithChapter2WhenNoInterpolationIsNeeded) {
  const std::vector<BondQuote> quotes{{0.5, 0.0, 97.5},
                                      {1.0, 0.05, 100.0},
                                      {1.5, 0.055, 100.3},
                                      {2.0, 0.06, 101.0}};
  const auto c2 = ch02::BootstrapZeroRates(quotes, 2);
  const auto c5 = BootstrapZeroCurve(quotes, 2);
  ASSERT_EQ(c5.times.size(), c2.times.size());
  for (std::size_t j = 0; j < c2.times.size(); ++j)
    EXPECT_NEAR(c5.rates[j], c2.rates[j], 1e-12);
}

TEST(Ch05Bootstrapping, FirstBondWithoutCurveGetsItsYield) {
  // With no starting curve, the first bond is priced at a flat rate: its yield.
  const auto cf = ch02::BondCashFlows(2.0, 0.04, 2);
  const double price = 99.0;
  EXPECT_NEAR(BootstrapNextRate({}, cf, price).x, ch02::BondYield(cf, price),
              1e-12);
}

TEST(Ch05Bootstrapping, MaturityMustExtendTheCurve) {
  const ZeroCurve c{{0.0, 1.0}, {0.05, 0.06}};
  EXPECT_THROW(
      (void)BootstrapNextRate(c, ch02::BondCashFlows(1.0, 0.05, 2), 100.0),
      std::invalid_argument);
}
