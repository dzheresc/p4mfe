#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch06/dollar_duration.hpp>
#include <p4mfe/ch06/duration_convexity.hpp>
#include <p4mfe/ch06/yield_curve_shifts.hpp>
#include <stdexcept>
#include <vector>

using namespace p4mfe;
using namespace p4mfe::ch06;

namespace {
// Upward sloping zero curve r(0, t) = 0.0125 + ln(1 + 2t) / 200.
double zero_curve(double t) { return 0.0125 + std::log1p(2 * t) / 200; }
}  // namespace

// ------------------------------------------------------------------ 6.5 ---
TEST(Ch06YieldShifts, FlatCurveMatchesYieldMetrics) {
  const auto cf = ch02::BondCashFlows(5.0, 0.06, 2);
  const double y = 0.045;
  const auto flat = [y](double) { return y; };
  const auto m = curve_shift_metrics(cf, flat);
  const auto my = ch02::ComputeBondMetrics(cf, y);
  EXPECT_NEAR(m.price, my.price, 1e-12);
  EXPECT_NEAR(m.duration, my.duration, 1e-12);
  EXPECT_NEAR(m.convexity, my.convexity, 1e-12);
  for (double dr : {-0.01, 0.0005, 0.02}) {
    EXPECT_NEAR(shifted_curve_price(cf, flat, dr), ch02::BondPrice(cf, y + dr),
                1e-12);
    EXPECT_NEAR(yield_change_from_shift(cf, flat, dr), dr, 1e-12);
  }
}

TEST(Ch06YieldShifts, TaylorApproximationOfShiftedPrice) {
  const auto cf = ch02::BondCashFlows(10.0, 0.05, 2);
  const auto m = curve_shift_metrics(cf, zero_curve);
  // D and C are the derivatives of B(dr) at 0.
  const double h = 1e-5;
  const double Bp = shifted_curve_price(cf, zero_curve, h),
               Bm = shifted_curve_price(cf, zero_curve, -h);
  EXPECT_NEAR(-(Bp - Bm) / (2 * h) / m.price, m.duration, 1e-6);
  EXPECT_NEAR((Bp - 2 * m.price + Bm) / (h * h) / m.price, m.convexity, 1e-3);
  for (double dr : {-0.005, 0.001, 0.01}) {
    const double exact = shifted_curve_price(cf, zero_curve, dr);
    const double err2 =
        std::abs(shifted_curve_price_approx(cf, zero_curve, dr) - exact);
    const double err1 =
        std::abs(ch02::PriceApproxDuration(m.price, m.duration, dr) - exact);
    EXPECT_LT(err2, err1 / 10);
    // Third-order error: about B D3 |dr|^3 / 6 with D3 <= T^3.
    EXPECT_LT(err2, m.price * std::pow(10.0 * std::abs(dr), 3) / 6);
    // Not flat: the yield moves by about, not exactly, dr.
    const double dy = yield_change_from_shift(cf, zero_curve, dr);
    EXPECT_NEAR(dy / dr, 1.0, 5e-3);
  }
}

// ------------------------------------------------------------------ 6.6 ---
TEST(Ch06DurationConvexity, Connections) {
  const auto cf = ch02::BondCashFlows(7.5, 0.04, 2);
  const double y = 0.05;
  const auto m = ch02::ComputeBondMetrics(cf, y);
  EXPECT_GT(cash_flow_dispersion(cf, y), 0.0);
  EXPECT_NEAR(cash_flow_dispersion(cf, y),
              m.convexity - m.duration * m.duration, 1e-12);
  // dD/dy = D^2 - C (< 0).
  const double h = 1e-6;
  const double dD =
      (ch02::BondDuration(cf, y + h) - ch02::BondDuration(cf, y - h)) / (2 * h);
  EXPECT_NEAR(dD, duration_derivative(m.duration, m.convexity), 1e-6);
  EXPECT_LT(dD, 0.0);
  // Zero coupon bond: D = T, C = T^2, no dispersion.
  const auto z = ch02::BondCashFlows(4.0, 0.0, 2);
  EXPECT_NEAR(cash_flow_dispersion(z, y), 0.0, 1e-12);
}

TEST(Ch06DurationConvexity, DiscreteConvexity) {
  const auto cf = ch02::BondCashFlows(3.0, 0.05, 2);
  const double y = 0.06;
  for (int m : {1, 2, 12}) {
    const double h = 1e-5;
    const double Bp = ch02::BondPriceDiscrete(cf, y + h, m),
                 Bm = ch02::BondPriceDiscrete(cf, y - h, m);
    const double B = ch02::BondPriceDiscrete(cf, y, m);
    EXPECT_NEAR(convexity_discrete(cf, y, m), (Bp - 2 * B + Bm) / (h * h) / B,
                1e-3);
  }
  EXPECT_NEAR(convexity_discrete(cf, y, 1000000), ch02::BondConvexity(cf, y),
              1e-4);
  EXPECT_THROW((void)convexity_discrete(cf, y, 0), std::invalid_argument);
}

TEST(Ch06DurationConvexity, PortfolioAverages) {
  const double y = 0.05;
  const auto b1 = ch02::BondCashFlows(2.0, 0.04, 2),
             b2 = ch02::BondCashFlows(9.0, 0.06, 2);
  const double n1 = 30, n2 = 70;
  ch02::CashFlows both;  // the portfolio as one set of cash flows
  for (const auto* p : {&b1, &b2}) {
    const double n = p == &b1 ? n1 : n2;
    for (std::size_t i = 0; i < p->times.size(); ++i) {
      both.times.push_back(p->times[i]);
      both.amounts.push_back(n * p->amounts[i]);
    }
  }
  const auto m1 = ch02::ComputeBondMetrics(b1, y),
             m2 = ch02::ComputeBondMetrics(b2, y),
             mp = ch02::ComputeBondMetrics(both, y);
  const std::vector<double> V{n1 * m1.price, n2 * m2.price};
  EXPECT_NEAR(
      portfolio_duration(V, std::vector<double>{m1.duration, m2.duration}),
      mp.duration, 1e-12);
  EXPECT_NEAR(
      portfolio_convexity(V, std::vector<double>{m1.convexity, m2.convexity}),
      mp.convexity, 1e-12);
  EXPECT_THROW((void)portfolio_duration(std::vector<double>{1, -1},
                                        std::vector<double>{2, 3}),
               std::invalid_argument);
}

// ------------------------------------------------------------- 6.7, 6.7.1 ---
TEST(Ch06DollarDuration, DollarRiskAndDv01) {
  const auto cf = ch02::BondCashFlows(5.0, 0.05, 2);
  const double y = 0.04;
  const auto r = bond_dollar_risk(cf, y);
  EXPECT_NEAR(r.dollar_duration, ch02::DollarDuration(cf, y), 1e-12);
  EXPECT_NEAR(r.dv01(), ch02::Dv01(cf, y), 1e-15);
  EXPECT_NEAR(r.dollar_convexity, dollar_convexity(cf, y), 1e-12);
  EXPECT_NEAR(duration_from_dollar(r), ch02::BondDuration(cf, y), 1e-12);
  EXPECT_NEAR(convexity_from_dollar(r), ch02::BondConvexity(cf, y), 1e-12);
  // DV01 by repricing: one-sided is off by C$ bp^2 / 2 (~5e-6 here), central
  // by B E[t^3] bp^3 / 6 (~2e-9).
  EXPECT_NEAR(dv01_reprice(cf, y), r.dv01() + 0.5 * r.dollar_convexity * 1e-8,
              1e-8);
  EXPECT_GT(std::abs(dv01_reprice(cf, y) - r.dv01()), 1e-6);
  EXPECT_NEAR(dv01_central(cf, y), r.dv01(), 1e-8);
  for (double dy : {-0.01, 0.002}) {
    const double dV = ch02::BondPrice(cf, y + dy) - r.value;
    EXPECT_LT(std::abs(value_change_approx_convexity(r, dy) - dV),
              std::abs(value_change_approx(r, dy) - dV) / 10);
  }
}

TEST(Ch06DollarDuration, ZeroValuePortfolio) {
  // Long one bond, short another of equal value: V = 0, D undefined, D$ fine.
  const double y = 0.05;
  const auto r1 = bond_dollar_risk(ch02::BondCashFlows(2.0, 0.05, 2), y);
  const auto r2 = bond_dollar_risk(ch02::BondCashFlows(10.0, 0.05, 2), y);
  const std::vector<DollarRisk> risks{r1, r2};
  const std::vector<double> units{1.0, -r1.value / r2.value};
  const auto p = portfolio_dollar_risk(units, risks);
  EXPECT_NEAR(p.value, 0.0, 1e-12);
  EXPECT_NEAR(p.dollar_duration,
              r1.dollar_duration + units[1] * r2.dollar_duration, 1e-12);
  EXPECT_LT(p.dollar_duration,
            0.0);  // net short duration: gains if yields rise
  EXPECT_THROW((void)duration_from_dollar(p), std::domain_error);
}

// ---------------------------------------------------------------- 6.7.2 ---
TEST(Ch06Immunization, ZeroValueDurationConvexity) {
  const double y = 0.045;
  // Portfolio: 7-year bonds. Hedges: 2-year and 20-year bonds.
  const auto port_cf = ch02::BondCashFlows(7.0, 0.05, 2);
  const auto h1_cf = ch02::BondCashFlows(2.0, 0.03, 2),
             h2_cf = ch02::BondCashFlows(20.0, 0.06, 2);
  const auto P = scale(bond_dollar_risk(port_cf, y), 1000.0);
  const auto H1 = bond_dollar_risk(h1_cf, y), H2 = bond_dollar_risk(h2_cf, y);
  const auto im = immunize(P, H1, H2);
  const auto total =
      portfolio_dollar_risk(std::vector<double>{1.0, im.units1, im.units2},
                            std::vector<DollarRisk>{P, H1, H2});
  EXPECT_NEAR(total.value + im.cash, 0.0, 1e-8);
  EXPECT_NEAR(total.dollar_duration, 0.0, 1e-7);
  EXPECT_NEAR(total.dollar_convexity, 0.0, 1e-5);
  // Exact repricing after parallel shifts: P&L is O(dy^3), far below the
  // unhedged and the duration-only hedged P&L.
  const double q = duration_hedge_units(P, H1);
  const auto value = [&](double dy, double u1, double u2, double cash) {
    return 1000.0 * ch02::BondPrice(port_cf, y + dy) +
           u1 * ch02::BondPrice(h1_cf, y + dy) +
           u2 * ch02::BondPrice(h2_cf, y + dy) + cash;
  };
  for (double dy : {-0.01, -0.001, 0.005}) {
    const double immunized = value(dy, im.units1, im.units2, im.cash);
    const double duration_only = value(dy, q, 0.0, -(P.value + q * H1.value));
    const double unhedged = value(dy, 0.0, 0.0, -P.value);
    EXPECT_LT(std::abs(immunized), std::abs(duration_only) / 5) << dy;
    EXPECT_LT(std::abs(duration_only), std::abs(unhedged) / 5) << dy;
  }
  EXPECT_THROW((void)immunize(P, H1, scale(H1, 2.0)), std::invalid_argument);
}
