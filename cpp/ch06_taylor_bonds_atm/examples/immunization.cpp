// Sections 6.5-6.7: a parallel shift of the zero curve, the duration and
// duration-convexity estimates of the new price, dollar duration and DV01,
// and immunizing a bond position with two other bonds and cash.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch06/dollar_duration.hpp>
#include <p4mfe/ch06/yield_curve_shifts.hpp>
#include <vector>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch06;

  const auto zero = [](double t) { return 0.0125 + std::log1p(2 * t) / 200; };
  const auto cf = ch02::BondCashFlows(10.0, 0.05, 2);
  const auto m = CurveShiftMetrics(cf, zero);
  std::printf(
      "10-year 5%% semiannual bond off the zero curve: B = %.6f, D = %.6f, C = "
      "%.6f\n",
      m.price, m.duration, m.convexity);
  std::printf("%8s  %12s  %12s  %12s  %12s\n", "dr (bp)", "exact", "duration",
              "dur+conv", "yield chg");
  for (double bp : {-100.0, -25.0, 10.0, 50.0, 200.0}) {
    const double dr = bp / 1e4;
    std::printf("%8.0f  %12.6f  %12.6f  %12.6f  %10.4f bp\n", bp,
                ShiftedCurvePrice(cf, zero, dr),
                ch02::PriceApproxDuration(m.price, m.duration, dr),
                ShiftedCurvePriceApprox(cf, zero, dr),
                1e4 * YieldChangeFromShift(cf, zero, dr));
  }

  const double y = 0.045;
  const auto port_cf = ch02::BondCashFlows(7.0, 0.05, 2);
  const auto h1_cf = ch02::BondCashFlows(2.0, 0.03, 2),
             h2_cf = ch02::BondCashFlows(20.0, 0.06, 2);
  const auto portfolio = Scale(BondDollarRisk(port_cf, y), 1000.0);
  const auto hedge1 = BondDollarRisk(h1_cf, y),
             hedge2 = BondDollarRisk(h2_cf, y);
  std::printf("\nAt a flat yield of 4.5%%:\n");
  std::printf(
      "  1000 x 7y 5%%:  V = %12.4f  D$ = %12.4f  C$ = %14.4f  DV01 = %.4f\n",
      portfolio.value, portfolio.dollar_duration, portfolio.dollar_convexity,
      portfolio.Dv01());
  std::printf(
      "  2y 3%%:         B = %12.4f  D$ = %12.4f  C$ = %14.4f  DV01 = %.4f\n",
      hedge1.value, hedge1.dollar_duration, hedge1.dollar_convexity,
      hedge1.Dv01());
  std::printf(
      "  20y 6%%:        B = %12.4f  D$ = %12.4f  C$ = %14.4f  DV01 = %.4f\n",
      hedge2.value, hedge2.dollar_duration, hedge2.dollar_convexity,
      hedge2.Dv01());

  const auto im = Immunize(portfolio, hedge1, hedge2);
  const double q = DurationHedgeUnits(portfolio, hedge1);
  std::printf("\nImmunized: %.4f x 2y, %.4f x 20y, cash %.4f\n", im.units1,
              im.units2, im.cash);
  std::printf("Duration hedge only: %.4f x 2y, cash %.4f\n", q,
              -(portfolio.value + q * hedge1.value));
  const auto value = [&](double dy, double u1, double u2, double cash) {
    return 1000.0 * ch02::BondPrice(port_cf, y + dy) +
           u1 * ch02::BondPrice(h1_cf, y + dy) +
           u2 * ch02::BondPrice(h2_cf, y + dy) + cash;
  };
  std::printf("\n%8s  %14s  %14s  %14s\n", "dy (bp)", "unhedged P&L",
              "duration P&L", "immunized P&L");
  for (double bp : {-200.0, -50.0, -10.0, 10.0, 50.0, 200.0}) {
    const double dy = bp / 1e4;
    std::printf("%8.0f  %14.6f  %14.6f  %14.6f\n", bp,
                value(dy, 0, 0, -portfolio.value),
                value(dy, q, 0, -(portfolio.value + q * hedge1.value)),
                value(dy, im.units1, im.units2, im.cash));
  }
}
