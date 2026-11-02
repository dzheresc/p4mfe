#pragma once
// Exercise 2 (section 5.7): price, yield, duration and convexity of a bond
// priced off a given zero rate curve.

#include <cmath>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/bond_yield.hpp>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 2 ---
// Compute the price, yield, modified duration, and convexity of a two year
// semiannual coupon bond with face value 100 and coupon rate 8%, if the zero
// rate curve is given by r(0, t) = 0.05 + 0.01 ln(1 + t/2).
//
// Cash flows: 4 at t = 0.5, 1, 1.5 and 104 at t = 2. Discounting each at its
// own zero rate,
//   B = sum_i c_i e^(-r(0, t_i) t_i) = 104.173911.
// The yield is the single rate that reproduces B, found with Newton's method:
//   y = 0.056792.
// Duration and convexity at that yield:
//   D = 1.890108,  C = 3.689501.
// "Modified duration": with a continuously compounded yield, -(1/B) dB/dy is
// both the duration of section 2.5 and the modified duration, so the answer
// is D = 1.890108. If the yield is quoted semiannually instead,
// y_2 = 2 (e^(y/2) - 1) = 0.057606, the modified duration of section 2.8,
// D_mac / (1 + y_2/2), is 1.837191; both are given.
[[nodiscard]] inline double Ex02ZeroRate(double t) {
  return 0.05 + 0.01 * std::log(1.0 + t / 2.0);
}

struct Ex02Result {
  double price, yield, duration, convexity;
  double yield_semiannual, modified_duration_semiannual;
};
[[nodiscard]] inline ch02::CashFlows Ex02CashFlows() {
  return ch02::BondCashFlows(2.0, 0.08, 2);
}

[[nodiscard]] inline Ex02Result Ex02Solve() {
  const auto cf = Ex02CashFlows();
  const double price = ch02::BondPriceFromZeroRate(cf, Ex02ZeroRate);
  const double y = BondYieldNewton(cf, price).x;
  const auto m = ch02::ComputeBondMetrics(cf, y);
  const double y2 = 2.0 * (std::exp(y / 2.0) - 1.0);
  return {price,       y,  m.duration,
          m.convexity, y2, ch02::ModifiedDuration(cf, y2, 2)};
}

}  // namespace p4mfe::ch05::ex
