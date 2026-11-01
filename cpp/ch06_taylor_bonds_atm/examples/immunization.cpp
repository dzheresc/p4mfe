// Sections 6.5-6.7: a parallel shift of the zero curve, the duration and
// duration-convexity estimates of the new price, dollar duration and DV01,
// and immunizing a bond position with two other bonds and cash.

#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch06/dollar_duration.hpp>
#include <p4mfe/ch06/yield_curve_shifts.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace p4mfe;
    using namespace p4mfe::ch06;

    const auto zero = [](double t) { return 0.0125 + std::log1p(2 * t) / 200; };
    const auto cf = ch02::bond_cash_flows(10.0, 0.05, 2);
    const auto m = curve_shift_metrics(cf, zero);
    std::printf("10-year 5%% semiannual bond off the zero curve: B = %.6f, D = %.6f, C = %.6f\n", m.price,
                m.duration, m.convexity);
    std::printf("%8s  %12s  %12s  %12s  %12s\n", "dr (bp)", "exact", "duration", "dur+conv", "yield chg");
    for (double bp : {-100.0, -25.0, 10.0, 50.0, 200.0}) {
        const double dr = bp / 1e4;
        std::printf("%8.0f  %12.6f  %12.6f  %12.6f  %10.4f bp\n", bp, shifted_curve_price(cf, zero, dr),
                    ch02::price_approx_duration(m.price, m.duration, dr), shifted_curve_price_approx(cf, zero, dr),
                    1e4 * yield_change_from_shift(cf, zero, dr));
    }

    const double y = 0.045;
    const auto port_cf = ch02::bond_cash_flows(7.0, 0.05, 2);
    const auto h1_cf = ch02::bond_cash_flows(2.0, 0.03, 2), h2_cf = ch02::bond_cash_flows(20.0, 0.06, 2);
    const auto P = scale(bond_dollar_risk(port_cf, y), 1000.0);
    const auto H1 = bond_dollar_risk(h1_cf, y), H2 = bond_dollar_risk(h2_cf, y);
    std::printf("\nAt a flat yield of 4.5%%:\n");
    std::printf("  1000 x 7y 5%%:  V = %12.4f  D$ = %12.4f  C$ = %14.4f  DV01 = %.4f\n", P.value, P.dollar_duration,
                P.dollar_convexity, P.dv01());
    std::printf("  2y 3%%:         B = %12.4f  D$ = %12.4f  C$ = %14.4f  DV01 = %.4f\n", H1.value, H1.dollar_duration,
                H1.dollar_convexity, H1.dv01());
    std::printf("  20y 6%%:        B = %12.4f  D$ = %12.4f  C$ = %14.4f  DV01 = %.4f\n", H2.value, H2.dollar_duration,
                H2.dollar_convexity, H2.dv01());

    const auto im = immunize(P, H1, H2);
    const double q = duration_hedge_units(P, H1);
    std::printf("\nImmunized: %.4f x 2y, %.4f x 20y, cash %.4f\n", im.units1, im.units2, im.cash);
    std::printf("Duration hedge only: %.4f x 2y, cash %.4f\n", q, -(P.value + q * H1.value));
    const auto value = [&](double dy, double u1, double u2, double cash) {
        return 1000.0 * ch02::bond_price(port_cf, y + dy) + u1 * ch02::bond_price(h1_cf, y + dy) +
               u2 * ch02::bond_price(h2_cf, y + dy) + cash;
    };
    std::printf("\n%8s  %14s  %14s  %14s\n", "dy (bp)", "unhedged P&L", "duration P&L", "immunized P&L");
    for (double bp : {-200.0, -50.0, -10.0, 10.0, 50.0, 200.0}) {
        const double dy = bp / 1e4;
        std::printf("%8.0f  %14.6f  %14.6f  %14.6f\n", bp, value(dy, 0, 0, -P.value),
                    value(dy, q, 0, -(P.value + q * H1.value)), value(dy, im.units1, im.units2, im.cash));
    }
}
