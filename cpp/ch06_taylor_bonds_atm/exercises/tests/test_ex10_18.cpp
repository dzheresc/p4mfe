#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <p4mfe/ch06/exercises/ex10_log_product_integral.hpp>
#include <p4mfe/ch06/exercises/ex11_crr_expansion.hpp>
#include <p4mfe/ch06/exercises/ex12_atm_put_error.hpp>
#include <p4mfe/ch06/exercises/ex13_atm_parity.hpp>
#include <p4mfe/ch06/exercises/ex14_atm_put_maturity.hpp>
#include <p4mfe/ch06/exercises/ex15_atm_put_dividends.hpp>
#include <p4mfe/ch06/exercises/ex16_bond_price_change.hpp>
#include <p4mfe/ch06/exercises/ex17_portfolio_immunization.hpp>
#include <p4mfe/ch06/exercises/ex18_curve_shift_yield.hpp>

using namespace p4mfe;
using namespace p4mfe::ch06;
using namespace p4mfe::ch06::ex;

TEST(Ch06Ex10, LogProductIntegral) {
  EXPECT_NEAR(kEx10Exact, 0.355065933151774, 1e-14);
  // (i) the integrand vanishes at both ends.
  EXPECT_LT(std::abs(Ex10Integrand(1e-10)), 1e-8);
  EXPECT_LT(std::abs(Ex10Integrand(1.0 - 1e-10)), 1e-8);
  // The integral numerically.
  EXPECT_NEAR(Ex10Numerical(), kEx10Exact, 1e-8);
  // (iii) the moments and the series.
  for (unsigned n : {0u, 1u, 3u, 7u})
    EXPECT_NEAR(Ex10MomentNumerical(n), Ex10Moment(n), 1e-8);
  for (unsigned num_terms : {10u, 100u, 1000u}) {
    const double tail = kEx10Exact - Ex10PartialSum(num_terms);
    EXPECT_GT(tail, 0.0);
    EXPECT_LT(tail, 0.5 / (static_cast<double>(num_terms) * num_terms));
  }
  // (iv) the partial fractions.
  for (double n = 1; n <= 50; ++n)
    EXPECT_NEAR(1.0 / (n * (n + 1) * (n + 1)),
                1.0 / n - 1.0 / (n + 1) - 1.0 / ((n + 1) * (n + 1)), 1e-15);
}

TEST(Ch06Ex11, CrrSmallStep) {
  for (double r : {0.0, 0.05}) {
    for (double sigma : {0.2, 0.4}) {
      for (double dt : {1e-2, 1e-3, 1e-4, 1e-5}) {
        const auto c = Ex11Crr(r, sigma, dt);
        const auto a = Ex11Approx(r, sigma, dt);
        // The tree is valid and matches the first two moments (u d = 1 by
        // construction).
        EXPECT_NEAR(c.u * c.d, 1.0, 1e-13);
        EXPECT_NEAR(c.p * c.u + (1 - c.p) * c.d, std::exp(r * dt), 1e-13);
        // Same expansions up to O(dt^(3/2)) -- in particular after dropping
        // O(dt).
        const double s = std::pow(dt, 1.5);
        const double k = Ex11UCoefficient(r, sigma);
        EXPECT_NEAR((c.u - a.u) / s, k, 0.2 * k + 1e-3)
            << r << " " << sigma << " " << dt;
        EXPECT_NEAR((c.d - a.d) / s, -k, 0.2 * k + 1e-3);
        EXPECT_LT(std::abs(c.p - a.p) / s, 1.0);
        EXPECT_LT(std::abs(c.u - a.u), dt);
        EXPECT_LT(std::abs(c.p - a.p), dt);
      }
    }
  }
}

TEST(Ch06Ex12, AtmPutApproximation) {
  const auto z = Ex12ZeroRates();
  EXPECT_NEAR(z.approx, 5.984134, 5e-7);
  EXPECT_NEAR(z.black_scholes, 5.968587, 5e-7);
  EXPECT_NEAR(z.relative_error, 0.002605, 5e-7);
  // r = q = 0: P_BS = S (2 N(sigma sqrt(T)/2) - 1).
  EXPECT_NEAR(z.black_scholes, AtmPriceExact(60.0, 1.0, 0.25), 1e-12);
  const auto w = Ex12WithRates();
  EXPECT_NEAR(w.approx, 4.814848, 5e-7);
  EXPECT_NEAR(w.black_scholes, 4.886987, 5e-7);
  EXPECT_NEAR(w.relative_error, 0.014761, 5e-7);
}

TEST(Ch06Ex13, ParityConsistentApproximation) {
  for (double r : {0.0, 0.06, 0.1})
    for (double q : {0.0, 0.03})
      for (double maturity : {0.25, 1.0, 5.0}) {
        EXPECT_NEAR(Ex13ParityGap(60.0, maturity, 0.25, r, q), 0.0, 1e-12);
        if (r != q) {
          EXPECT_GT(std::abs(Ex13ParityGap6103(60.0, maturity, 0.25, r, q)),
                    1e-6);
        }
      }
  const auto res = Ex13Solve();
  EXPECT_NEAR(res.approx, 4.861031, 5e-7);
  EXPECT_NEAR(res.relative_error, 0.005311, 5e-7);
  EXPECT_LT(res.relative_error, Ex12WithRates().relative_error);
  // For r = q = 0 both reduce to sigma S sqrt(T/(2 pi)).
  EXPECT_NEAR(Ex13Put(60.0, 1.0, 0.25, 0.0, 0.0),
              AtmPriceApprox(60.0, 1.0, 0.25), 1e-14);
}

TEST(Ch06Ex14, LongMaturities) {
  const auto t = Ex14Table();
  const double expected[] = {0.011262, 0.018784, 0.037632, 0.075497};
  for (std::size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t[i].relative_error, expected[i], 5e-7);
    EXPECT_GT(t[i].approx, t[i].black_scholes);  // always overestimates
  }
  EXPECT_NEAR(t[3].approx, 21.409489, 5e-7);
  EXPECT_NEAR(t[3].black_scholes, 19.906602, 5e-7);
}

TEST(Ch06Ex15, LongMaturitiesWithRates) {
  const auto t = Ex15Table();
  const double expected[] = {0.004199, 0.028303, 0.072609, 0.288986, 1.413676};
  for (std::size_t i = 0; i < t.size(); ++i)
    EXPECT_NEAR(t[i].relative_error, expected[i], 5e-7);
  EXPECT_NEAR(t[0].approx, 4.131720, 5e-7);
  EXPECT_NEAR(t[0].black_scholes, 4.149140, 5e-7);
  EXPECT_LT(t[4].approx, 0.0);  // the 20-year "price" is negative
}

TEST(Ch06Ex16, DurationAndConvexityPriceChange) {
  const auto t = Ex16Table();
  const double dur[] = {100.8485, 100.2425, 99.485, 97.97};
  const double conv[] = {100.84862625, 100.24565625, 99.497625, 98.0205};
  for (std::size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t[i].duration_only, dur[i], 1e-10);
    EXPECT_NEAR(t[i].duration_convexity, conv[i], 1e-10);
    EXPECT_NEAR(t[i].duration_convexity - t[i].duration_only,
                101.0 * 2.5 * t[i].dy * t[i].dy / 2.0, 1e-10);
  }
}

TEST(Ch06Ex17, PortfolioImmunization) {
  EXPECT_NEAR(Ex17NewValue(), 49'505'000.0, 1e-6);
  const auto h = Ex17Immunize();
  EXPECT_NEAR(h.units1, -20202.0202, 1e-4);
  EXPECT_NEAR(h.units2, -139762.4039, 1e-4);
  EXPECT_NEAR(h.cash, -12'962'962.963, 1e-3);
  // The hedged position has zero value, D$ and C$.
  const auto p = Ex17Portfolio(), b1 = Ex17Bond1(), b2 = Ex17Bond2();
  EXPECT_NEAR(p.dollar_duration + h.units1 * b1.dollar_duration +
                  h.units2 * b2.dollar_duration,
              0.0, 1e-4);
  EXPECT_NEAR(p.dollar_convexity + h.units1 * b1.dollar_convexity +
                  h.units2 * b2.dollar_convexity,
              0.0, 1e-4);
  EXPECT_NEAR(p.value + h.units1 * b1.value + h.units2 * b2.value + h.cash, 0.0,
              1e-4);
}

TEST(Ch06Ex18, YieldUnderParallelShifts) {
  const auto r = Ex18Solve();
  EXPECT_NEAR(r.price, 106.199524, 5e-7);
  EXPECT_NEAR(r.yield, 0.042511, 5e-7);
  const auto cf = Ex18CashFlows();
  EXPECT_NEAR(ch02::BondPrice(cf, r.yield), r.price, 1e-9);
  const auto dy = Ex18YieldChanges();
  const double expected[] = {0.00099979, 0.00199957, 0.00499893, 0.00999784,
                             0.01999562};
  for (std::size_t i = 0; i < dy.size(); ++i) {
    EXPECT_NEAR(dy[i], expected[i], 5e-9);
    EXPECT_LT(dy[i], kEx18Shifts[i]);
    EXPECT_GT(dy[i], 0.999 * kEx18Shifts[i]);
    // The new yield reprices the shifted bond.
    const double price = ShiftedCurvePrice(cf, Ex18ZeroRate, kEx18Shifts[i]);
    EXPECT_NEAR(ch02::BondPrice(cf, r.yield + dy[i]), price, 1e-9);
  }
}
