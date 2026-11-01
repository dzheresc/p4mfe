#pragma once
// Exercise 1 (section 5.7): yield, duration and convexity of a coupon bond.

#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/bond_yield.hpp>
#include <p4mfe/ch05/solvers.hpp>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Use Newton's method to find the yield of a five year semiannual coupon bond
// with 3.375% coupon rate and price 100 1/32. What are the duration and
// convexity of the bond?
//
// Face value 100, so the cash flows are 1.6875 at t = 0.5, 1, ..., 4.5 and
// 101.6875 at t = 5. The (continuously compounded) yield solves
//   f(y) = sum_i c_i e^(-y t_i) - 100.03125 = 0,  f'(y) = -sum_i t_i c_i e^(-y t_i),
// and Newton's method from y0 = 0.1 (Table 5.6) converges in 5 steps to
//   y = 0.033401 (3.3401%).
// At that yield
//   D = sum t_i c_i e^(-y t_i) / B = 4.642735,
//   C = sum t_i^2 c_i e^(-y t_i) / B = 22.573118.
// Sanity check: the price is just above par, so y is just below the coupon
// rate expressed continuously, 2 ln(1 + 0.03375/2) = 0.033469.
struct Ex01Bond {
    double maturity = 5.0, coupon_rate = 0.03375, price = 100.0 + 1.0 / 32.0;
    int freq = 2;
};
struct Ex01Result {
    double yield;
    std::size_t iterations;
    double duration, convexity;
};
[[nodiscard]] inline Ex01Result ex01_solve(const Ex01Bond& b = {}) {
    const auto cf = ch02::bond_cash_flows(b.maturity, b.coupon_rate, b.freq);
    const auto res = bond_yield_newton(cf, b.price);
    const auto m = ch02::bond_metrics(cf, res.x);
    return {res.x, res.iterations, m.duration, m.convexity};
}

}  // namespace p4mfe::ch05::ex
