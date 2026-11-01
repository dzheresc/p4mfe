#pragma once
// Exercises 14-19 (section 2.12): price, duration and convexity of coupon
// bonds, how duration moves with the coupon, first and second order price
// changes for a yield move, and the identity C = D^2 - dD/dy.
//
// As in the book, yields are continuously compounded and
//   B(y) = sum_i c_i e^(-y t_i),  D = -B'/B = (1/B) sum_i t_i c_i e^(-y t_i),
//   C = B''/B = (1/B) sum_i t_i^2 c_i e^(-y t_i).

#include <p4mfe/ch02/bonds.hpp>

#include <cmath>

namespace p4mfe::ch02::ex {

// ----------------------------------------------------------- Exercise 14 ---
// The yield of a semiannual coupon bond with 6% coupon rate and 30 months to
// maturity is 9%. What are the price, duration and convexity of the bond?
//
// Cash flows: 3 at t = 0.5, 1, 1.5, 2 and 103 at t = 2.5. With y = 0.09:
//   B = 3 (e^(-0.045) + e^(-0.09) + e^(-0.135) + e^(-0.18)) + 103 e^(-0.225) = 92.9839,
//   D = (1/B) sum t_i c_i e^(-0.09 t_i) = 2.352418,
//   C = (1/B) sum t_i^2 c_i e^(-0.09 t_i) = 5.736739.
// The bond trades below par since its 6% coupon is under the 9% yield.
[[nodiscard]] inline BondMetrics ex14_metrics() { return bond_metrics(bond_cash_flows(2.5, 0.06, 2), 0.09); }

// ----------------------------------------------------------- Exercise 15 ---
// The yield of a 14 month quarterly coupon bond with 8% coupon rate is 7%.
// Compute the price, duration and convexity of the bond.
//
// Coupons of 2 every 3 months, counted back from maturity at 14/12: at
// 2, 5, 8, 11 and 14 months, i.e. t = 1/6, 5/12, 2/3, 11/12, 7/6, with the
// face value 100 paid at 7/6. With y = 0.07:
//   B = 101.7049,  D = 1.118911,  C = 1.285705.
// (B is the full price, which includes the interest accrued since the last
// coupon date one month ago.)
[[nodiscard]] inline BondMetrics ex15_metrics() {
    return bond_metrics(bond_cash_flows(14.0 / 12.0, 0.08, 4), 0.07);
}

// ----------------------------------------------------------- Exercise 16 ---
// If the coupon rate of a bond goes up, what can be said about the value of
// the bond and its duration? Give a financial argument.
//
// Value: it goes up. Every cash flow is at least as large and each is
// discounted by the same positive factor, B = (c/m) A + F e^(-y T) with
// A = sum_i e^(-y t_i) > 0, so dB/dc = A/m > 0.
//
// Duration: it goes down. Duration is the average of the cash flow times
// weighted by present value. A higher coupon moves weight from the final
// payment of principal at T to the earlier coupon dates, so the average time
// falls. Formally, with A1 = sum_i t_i e^(-y t_i),
//   D(c) = ((c/m) A1 + F T e^(-yT)) / ((c/m) A + F e^(-yT)),
//   dD/dc = (1/m) F e^(-yT) (A1 - T A) / ((c/m) A + F e^(-yT))^2 < 0,
// because t_i <= T for every i, with strict inequality for the earlier
// coupons, so A1 < T A. A zero coupon bond (c = 0) has the largest
// duration, T; as c -> inf the duration falls to A1 / A, the duration of the
// coupons alone. Financially: a bond paying more of its value sooner is less
// exposed to the yield, since less of its value sits at long maturities where
// a yield change compounds over more years.
[[nodiscard]] inline BondMetrics ex16_metrics(double T, double coupon_rate, int freq, double y) {
    return bond_metrics(bond_cash_flows(T, coupon_rate, freq), y);
}

// ----------------------------------------------------------- Exercise 17 ---
// By how much would the price of a ten year zero coupon bond change if the
// yield increases by ten basis points (0.001)?
//
// B(y) = 100 e^(-10 y), so D = 10 and C = 100. To first order
//   dB/B ~ -D dy = -10 * 0.001 = -1%,
// and with convexity -D dy + C dy^2/2 = -1% + 0.005% = -0.995%. The exact
// relative change is e^(-10 * 0.001) - 1 = -0.9950%. So the price drops by
// about 1% (0.995% exactly), whatever the current yield.
inline constexpr double ex17_duration_estimate = -0.01;
[[nodiscard]] inline double ex17_exact_relative_change(double T = 10.0, double dy = 0.001) {
    return std::expm1(-T * dy);
}
[[nodiscard]] inline double ex17_convexity_estimate(double T = 10.0, double dy = 0.001) {
    return -T * dy + 0.5 * T * T * dy * dy;
}

// ----------------------------------------------------------- Exercise 18 ---
// A five year bond with 3.5 years duration is worth 102. What is the value of
// the bond if the yield decreases by fifty basis points?
//
// Only price and duration are known, so use the first order approximation
//   B(y + dy) ~ B (1 - D dy) = 102 (1 + 3.5 * 0.005) = 103.785.
// The true value is a little higher, since convexity adds B C dy^2 / 2 > 0
// (for example C ~ 15 would add about 0.019).
[[nodiscard]] inline double ex18_new_price(double B = 102.0, double D = 3.5, double dy = -0.005) {
    return price_approx_duration(B, D, dy);
}

// ----------------------------------------------------------- Exercise 19 ---
// Establish the relationship between duration and convexity
//   C = D^2 - dD/dy.
//
// D = -B'/B, so by the quotient rule
//   dD/dy = -(B'' B - (B')^2) / B^2 = -B''/B + (B'/B)^2 = -C + D^2,
// hence C = D^2 - dD/dy. (So dD/dy = D^2 - C = -(1/B) sum (t_i - D)^2 c_i e^(-y t_i) <= 0:
// C - D^2 is the variance of the cash flow times under the present value
// weights, and duration falls as the yield rises.)

// dD/dy by a central difference, to check the identity numerically.
[[nodiscard]] inline double ex19_duration_slope(const CashFlows& cf, double y, double h = 1e-5) {
    return (bond_duration(cf, y + h) - bond_duration(cf, y - h)) / (2.0 * h);
}

}  // namespace p4mfe::ch02::ex
