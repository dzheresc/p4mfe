#pragma once
// Exercise 18 (section 6.11): yield of a bond priced off a zero curve, and
// how the yield moves under parallel shifts of the curve.

#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch06/yield_curve_shifts.hpp>

#include <array>
#include <cmath>
#include <cstddef>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 18 ---
// (i) The zero rate curve is
//   r1(0, t) = 0.025 + exp(-t/100)/100 + t/(100 (t + 1)).
// Find the yield of a four year semiannual coupon bond with coupon rate 6%,
// face value 100, interest compounded continuously.
// (ii) The zero curve shifts up in parallel by dr = 10, 20, 50, 100, 200bp.
// By how much does the yield of the bond increase in each case?
//
// (i) Cash flows: 3 at t = 0.5, 1, ..., 3.5 and 103 at t = 4. Pricing off the
// curve, B = sum c_i exp(-r1(0, t_i) t_i) = 106.199524, and the yield, from
// Newton's method on sum c_i e^(-y t_i) = B, is
//   y = 0.042511 (4.2511%).
// (ii) After the shift every discount factor gets an extra e^(-dr t_i); a new
// yield is solved from the new price:
//   dr      yield change     change - dr
//   0.001   0.00099979     -0.0021bp
//   0.002   0.00199957     -0.0043bp
//   0.005   0.00499893     -0.0107bp
//   0.01    0.00999784     -0.0216bp
//   0.02    0.01999562     -0.0438bp
// The yield moves by almost exactly dr (99.98% of it): the curve is nearly
// flat over the bond's life (r1 runs from 3.83% at 6 months to 4.26% at 4
// years), and the shortfall grows linearly in dr. A flat curve would move the
// yield by exactly dr (section 6.5).
[[nodiscard]] inline double ex18_zero_rate(double t) {
    return 0.025 + std::exp(-t / 100.0) / 100.0 + t / (100.0 * (t + 1.0));
}
[[nodiscard]] inline ch02::CashFlows ex18_cash_flows() { return ch02::bond_cash_flows(4.0, 0.06, 2); }

struct Ex18Result {
    double price, yield;
};
[[nodiscard]] inline Ex18Result ex18_solve() {
    const auto cf = ex18_cash_flows();
    const double B = ch02::bond_price_from_zero_rate(cf, ex18_zero_rate);
    return {B, ch02::bond_yield(cf, B)};
}

inline constexpr std::array<double, 5> ex18_shifts{0.001, 0.002, 0.005, 0.01, 0.02};

[[nodiscard]] inline std::array<double, 5> ex18_yield_changes() {
    const auto cf = ex18_cash_flows();
    std::array<double, 5> out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = yield_change_from_shift(cf, ex18_zero_rate, ex18_shifts[i]);
    return out;
}

}  // namespace p4mfe::ch06::ex
