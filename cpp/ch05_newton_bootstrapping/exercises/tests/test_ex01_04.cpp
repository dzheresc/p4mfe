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
  const auto r = ex01_solve();
  EXPECT_NEAR(r.yield, 0.033401, 5e-7);
  EXPECT_NEAR(r.duration, 4.642735, 5e-7);
  EXPECT_NEAR(r.convexity, 22.573118, 5e-7);
  // Independent checks: the yield reprices the bond; D and C are the
  // first two (scaled) derivatives of B(y), by finite differences.
  const auto cf = ch02::bond_cash_flows(5.0, 0.03375, 2);
  const double B = 100.0 + 1.0 / 32.0;
  EXPECT_NEAR(ch02::bond_price(cf, r.yield), B, 1e-10);
  const double h = 1e-4;
  const double Bp = ch02::bond_price(cf, r.yield + h),
               Bm = ch02::bond_price(cf, r.yield - h);
  EXPECT_NEAR(-(Bp - Bm) / (2 * h) / B, r.duration, 1e-6);
  EXPECT_NEAR((Bp - 2 * B + Bm) / (h * h) / B, r.convexity, 1e-4);
  // Agrees with the bisection and secant yields and with chapter 2.
  EXPECT_NEAR(bond_yield_bisection(cf, B).x, r.yield, 1e-10);
  EXPECT_NEAR(ch02::bond_yield(cf, B), r.yield, 1e-10);
}

TEST(Ch05Ex02, BondOffZeroCurve) {
  const auto r = ex02_solve();
  EXPECT_NEAR(r.price, 104.173911, 5e-7);
  EXPECT_NEAR(r.yield, 0.056792, 5e-7);
  EXPECT_NEAR(r.duration, 1.890108, 5e-7);
  EXPECT_NEAR(r.convexity, 3.689501, 5e-7);
  EXPECT_NEAR(r.yield_semiannual, 0.057606, 5e-7);
  EXPECT_NEAR(r.modified_duration_semiannual, 1.837191, 5e-7);
  // Direct sum of the four discounted cash flows.
  double B = 0.0;
  for (double t : {0.5, 1.0, 1.5}) B += 4.0 * std::exp(-ex02_zero_rate(t) * t);
  B += 104.0 * std::exp(-ex02_zero_rate(2.0) * 2.0);
  EXPECT_NEAR(r.price, B, 1e-12);
  // Both yields reprice the bond; modified duration is -(1/B) dB/dy_2.
  const auto cf = ex02_cash_flows();
  EXPECT_NEAR(ch02::bond_price(cf, r.yield), B, 1e-10);
  EXPECT_NEAR(ch02::bond_price_discrete(cf, r.yield_semiannual, 2), B, 1e-10);
  const double h = 1e-5;
  const double dB = (ch02::bond_price_discrete(cf, r.yield_semiannual + h, 2) -
                     ch02::bond_price_discrete(cf, r.yield_semiannual - h, 2)) /
                    (2 * h);
  EXPECT_NEAR(-dB / B, r.modified_duration_semiannual, 1e-7);
}

TEST(Ch05Ex03, LimitsAndMonotonicity) {
  // One market for each case F < K, F = K, F > K.
  const double S = 50.0, T = 0.75, r = 0.05, q = 0.01;
  const double F = S * std::exp((r - q) * T);
  for (double K : {60.0, F, 40.0}) {
    const auto rg = ex03_price_range(S, K, T, r, q);
    const double C = 0.5 * (rg.lower + rg.upper);
    const Ex03Market m{S, K, T, r, q, C};
    // (i)
    EXPECT_NEAR(ex03_f(200.0, m), ex03_limit_infinity(m), 1e-9);
    const auto d_big = ch03::bs_d1_d2(S, K, T, 200.0, r, q);
    EXPECT_GT(d_big.d1, 50.0);
    EXPECT_LT(d_big.d2, -50.0);
    // (ii)
    EXPECT_NEAR(ex03_f(1e-12, m), ex03_limit_zero(m),
                1e-9);  // f - lim = O(x) when F = K
    const auto d_small = ch03::bs_d1_d2(S, K, T, 1e-6, r, q);
    const double lim = ex03_d_limit_zero(m);
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
    double prev = ex03_limit_zero(m);
    for (double x = 0.01; x < 5.0; x += 0.01) {
      const double fx = ex03_f(x, m);
      EXPECT_GT(ch03::bs_greeks(ch01::OptionType::kCall, S, K, T, x, r, q).vega,
                0.0);
      EXPECT_GE(fx, prev);
      EXPECT_LT(fx, ex03_limit_infinity(m));
      prev = fx;
    }
    EXPECT_GT(ex03_f(1.0, m), ex03_f(0.5, m));
    // (iv) the range is the no-arbitrage range (3.99)...
    const auto nb = ch01::CallBounds(S, K, T, r, q);
    EXPECT_NEAR(rg.lower, nb.lower, 1e-12);
    EXPECT_NEAR(rg.upper, nb.upper, 1e-12);
    // ...inside it there is a root, outside it none.
    EXPECT_TRUE(ex03_has_positive_root(m));
    const double sigma =
        implied_vol_newton(ch01::OptionType::kCall, C, S, K, T, r, q).x;
    EXPECT_NEAR(ex03_f(sigma, m), 0.0, 1e-8);
    EXPECT_FALSE(ex03_has_positive_root({S, K, T, r, q, rg.upper}));
    EXPECT_FALSE(ex03_has_positive_root({S, K, T, r, q, rg.lower}));
    EXPECT_FALSE(ex03_has_positive_root({S, K, T, r, q, rg.upper + 1.0}));
  }
}

TEST(Ch05Ex04, ImpliedVolThreeMethods) {
  const auto b = ex04_bisection(), s = ex04_secant(), n = ex04_newton();
  EXPECT_NEAR(n.x, 0.397049, 5e-7);
  EXPECT_NEAR(b.x, n.x, 1e-6);
  EXPECT_NEAR(s.x, n.x, 1e-6);
  EXPECT_EQ(n.iterations, 3u);
  EXPECT_EQ(s.iterations, 3u);
  EXPECT_GE(b.iterations, 20u);  // at least log2(1 / 1e-6) halvings
  // The implied vol reprices the call; matches chapter 3's solver.
  const Ex04Market m;
  EXPECT_NEAR(ch03::bs_call(m.S, m.K, m.T, n.x, m.r, m.q), m.C, 1e-9);
  EXPECT_NEAR(ch03::implied_vol_newton(ch01::OptionType::kCall, m.C, m.S, m.K,
                                       m.T, m.r, m.q)
                  .sigma,
              n.x, 1e-8);
  // (ii)
  EXPECT_NEAR(ex04_sigma_approx(), 0.396672, 5e-7);
  EXPECT_NEAR(ex04_relative_error(), 9.49e-4, 5e-6);
}
