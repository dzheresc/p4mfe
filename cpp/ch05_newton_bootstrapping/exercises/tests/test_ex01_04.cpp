#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch05/exercises/ex01_bond_yield.hpp>
#include <p4mfe/ch05/exercises/ex02_zero_curve_bond.hpp>
#include <p4mfe/ch05/exercises/ex03_implied_vol_limits.hpp>
#include <p4mfe/ch05/exercises/ex04_implied_vol_methods.hpp>
#include <p4mfe/ch05/implied_vol.hpp>

using namespace p4mfe;
using namespace p4mfe::ch05;
using namespace p4mfe::ch05::ex;

TEST(Ch05Ex01, YieldDurationConvexity) {
  const auto r = Ex01Solve();
  EXPECT_NEAR(r.yield, 0.033401, 5e-7);
  EXPECT_NEAR(r.duration, 4.642735, 5e-7);
  EXPECT_NEAR(r.convexity, 22.573118, 5e-7);
  // Independent checks: the yield reprices the bond; D and C are the
  // first two (scaled) derivatives of B(y), by finite differences.
  const auto cf = ch02::BondCashFlows(5.0, 0.03375, 2);
  const double price = 100.0 + 1.0 / 32.0;
  EXPECT_NEAR(ch02::BondPrice(cf, r.yield), price, 1e-10);
  const double h = 1e-4;
  const double price_up = ch02::BondPrice(cf, r.yield + h),
               price_down = ch02::BondPrice(cf, r.yield - h);
  EXPECT_NEAR(-(price_up - price_down) / (2 * h) / price, r.duration, 1e-6);
  EXPECT_NEAR((price_up - 2 * price + price_down) / (h * h) / price,
              r.convexity, 1e-4);
  // Agrees with the bisection and secant yields and with chapter 2.
  EXPECT_NEAR(BondYieldBisection(cf, price).x, r.yield, 1e-10);
  EXPECT_NEAR(ch02::BondYield(cf, price), r.yield, 1e-10);
}

TEST(Ch05Ex02, BondOffZeroCurve) {
  const auto r = Ex02Solve();
  EXPECT_NEAR(r.price, 104.173911, 5e-7);
  EXPECT_NEAR(r.yield, 0.056792, 5e-7);
  EXPECT_NEAR(r.duration, 1.890108, 5e-7);
  EXPECT_NEAR(r.convexity, 3.689501, 5e-7);
  EXPECT_NEAR(r.yield_semiannual, 0.057606, 5e-7);
  EXPECT_NEAR(r.modified_duration_semiannual, 1.837191, 5e-7);
  // Direct sum of the four discounted cash flows.
  double price = 0.0;
  for (double t : {0.5, 1.0, 1.5})
    price += 4.0 * std::exp(-Ex02ZeroRate(t) * t);
  price += 104.0 * std::exp(-Ex02ZeroRate(2.0) * 2.0);
  EXPECT_NEAR(r.price, price, 1e-12);
  // Both yields reprice the bond; modified duration is -(1/B) dB/dy_2.
  const auto cf = Ex02CashFlows();
  EXPECT_NEAR(ch02::BondPrice(cf, r.yield), price, 1e-10);
  EXPECT_NEAR(ch02::BondPriceDiscrete(cf, r.yield_semiannual, 2), price, 1e-10);
  const double h = 1e-5;
  const double d_price =
      (ch02::BondPriceDiscrete(cf, r.yield_semiannual + h, 2) -
       ch02::BondPriceDiscrete(cf, r.yield_semiannual - h, 2)) /
      (2 * h);
  EXPECT_NEAR(-d_price / price, r.modified_duration_semiannual, 1e-7);
}

TEST(Ch05Ex03, LimitsAndMonotonicity) {
  // One market for each case F < K, F = K, F > K.
  const double spot = 50.0, maturity = 0.75, r = 0.05, q = 0.01;
  const double forward = spot * std::exp((r - q) * maturity);
  for (double strike : {60.0, forward, 40.0}) {
    const auto rg = Ex03PriceRange(spot, strike, maturity, r, q);
    const double call = 0.5 * (rg.lower + rg.upper);
    const Ex03Market m{spot, strike, maturity, r, q, call};
    // (i)
    EXPECT_NEAR(Ex03F(200.0, m), Ex03LimitInfinity(m), 1e-9);
    const auto d_big = ch03::BsD1D2(spot, strike, maturity, 200.0, r, q);
    EXPECT_GT(d_big.d1, 50.0);
    EXPECT_LT(d_big.d2, -50.0);
    // (ii)
    EXPECT_NEAR(Ex03F(1e-12, m), Ex03LimitZero(m),
                1e-9);  // f - lim = O(x) when F = K
    const auto d_small = ch03::BsD1D2(spot, strike, maturity, 1e-6, r, q);
    const double lim = Ex03DLimitZero(m);
    if (std::isinf(lim)) {
      EXPECT_GT(d_small.d1 * lim, 0.0);
      EXPECT_GT(std::abs(d_small.d1), 1e3);
      EXPECT_GT(std::abs(d_small.d2), 1e3);
    } else {
      EXPECT_NEAR(d_small.d1, 0.0, 1e-5);
      EXPECT_NEAR(d_small.d2, 0.0, 1e-5);
    }
    // (iii) f' = vega > 0, so f increases and stays between the limits. (For
    // small x, deep out of the money, f equals its limit to double precision,
    // so the computed values are only non-decreasing.)
    double prev = Ex03LimitZero(m);
    for (double x = 0.01; x < 5.0; x += 0.01) {
      const double fx = Ex03F(x, m);
      EXPECT_GT(ch03::BsGreeks(ch01::OptionType::kCall, spot, strike, maturity,
                               x, r, q)
                    .vega,
                0.0);
      EXPECT_GE(fx, prev);
      EXPECT_LT(fx, Ex03LimitInfinity(m));
      prev = fx;
    }
    EXPECT_GT(Ex03F(1.0, m), Ex03F(0.5, m));
    // (iv) the range is the no-arbitrage range (3.99)...
    const auto nb = ch01::CallBounds(spot, strike, maturity, r, q);
    EXPECT_NEAR(rg.lower, nb.lower, 1e-12);
    EXPECT_NEAR(rg.upper, nb.upper, 1e-12);
    // ...inside it there is a root, outside it none.
    EXPECT_TRUE(Ex03HasPositiveRoot(m));
    const double sigma = ImpliedVolNewton(ch01::OptionType::kCall, call, spot,
                                          strike, maturity, r, q)
                             .x;
    EXPECT_NEAR(Ex03F(sigma, m), 0.0, 1e-8);
    EXPECT_FALSE(Ex03HasPositiveRoot({spot, strike, maturity, r, q, rg.upper}));
    EXPECT_FALSE(Ex03HasPositiveRoot({spot, strike, maturity, r, q, rg.lower}));
    EXPECT_FALSE(
        Ex03HasPositiveRoot({spot, strike, maturity, r, q, rg.upper + 1.0}));
  }
}

TEST(Ch05Ex04, ImpliedVolThreeMethods) {
  const auto b = Ex04Bisection(), s = Ex04Secant(), n = Ex04Newton();
  EXPECT_NEAR(n.x, 0.397049, 5e-7);
  EXPECT_NEAR(b.x, n.x, 1e-6);
  EXPECT_NEAR(s.x, n.x, 1e-6);
  EXPECT_EQ(n.iterations, 3u);
  EXPECT_EQ(s.iterations, 3u);
  EXPECT_GE(b.iterations, 20u);  // at least log2(1 / 1e-6) halvings
  // The implied vol reprices the call; matches chapter 3's solver.
  const Ex04Market m;
  EXPECT_NEAR(ch03::BsCall(m.spot, m.strike, m.maturity, n.x, m.r, m.q), m.call,
              1e-9);
  EXPECT_NEAR(ch03::ImpliedVolNewton(ch01::OptionType::kCall, m.call, m.spot,
                                     m.strike, m.maturity, m.r, m.q)
                  .sigma,
              n.x, 1e-8);
  // (ii)
  EXPECT_NEAR(Ex04SigmaApprox(), 0.396672, 5e-7);
  EXPECT_NEAR(Ex04RelativeError(), 9.49e-4, 5e-6);
}
