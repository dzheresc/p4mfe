#pragma once
// Exercise 8 (section 5.7): bootstrapping with a known overnight rate and a
// zero rate curve linear on given intervals.

#include <cmath>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/bootstrapping.hpp>
#include <vector>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 8 ---
// Bootstrap a continuously compounded zero rate curve from these semiannual
// coupon bonds:
//   maturity    coupon   price
//   6 months    0        97.5
//   1 year      5        100
//   20 months   6        103
//   40 months   5        102
//   5 years     4        103
// The overnight rate is 5% and the curve is linear on [0, 0.5], [0.5, 1],
// [1, 5/3], [5/3, 10/3], [10/3, 5].
//
// The intervals end exactly at the maturities, so this is
// bootstrap_zero_curve started from the one-point curve r(0, 0) = 0.05:
//   6 months: 97.5 = 100 e^(-0.5 r(0, 0.5))                 r(0, 0.5) = 5.0636%
//   1 year:   100 = 2.5 e^(-0.5 r(0, 0.5)) + 102.5 e^(-r(0, 1))   r(0, 1)
//   = 4.9370% 20 months, coupons at 2, 8, 14, 20 months (the hint, (5.43)):
//     103 = 3 e^(-r(1/6)/6) + 3 e^(-2 r(2/3)/3) + 3 e^(-7 r(7/6)/6) + 103 e^(-5
//     r(5/3)/3), r(1/6), r(2/3) from [0, 0.5] and [0.5, 1]; with x = r(0, 5/3),
//     r(0, 7/6) = (3 r(0, 1) + x)/4; Newton:                  r(0, 5/3)
//     = 5.2983%
//   40 months, coupons every 6 months from 4 months:          r(0, 10/3)
//   = 4.5326% 5 years:                                                  r(0, 5)
//   = 3.2119%
// The hint in the book prints r(0, 7/6) = (3 r(0, 1) + 4 r(0, 5/3))/4; the
// linear interpolation it states, (3/2)(5/3 - t) r(0, 1) + (3/2)(t - 1) r(0,
// 5/3) at t = 7/6, gives weights 3/4 and 1/4, i.e. (3 r(0, 1) + r(0, 5/3))/4.
// The code uses the interpolation, and the tests check (5.43) holds. The steep
// drop at 5 years is what the data say: a 4% coupon bond at 103 has a yield of
// only about 3.3%.
[[nodiscard]] inline std::vector<ch02::BondQuote> Ex08Quotes() {
  return {{0.5, 0.0, 97.5},
          {1.0, 0.05, 100.0},
          {5.0 / 3.0, 0.06, 103.0},
          {10.0 / 3.0, 0.05, 102.0},
          {5.0, 0.04, 103.0}};
}
inline constexpr double kEx08Overnight = 0.05;

[[nodiscard]] inline ch02::ZeroCurve Ex08Curve() {
  return BootstrapZeroCurve(Ex08Quotes(), 2,
                            ch02::ZeroCurve{{0.0}, {kEx08Overnight}});
}

// Right side of (5.43) minus 103, using the hint's formula for r(0, 7/6).
[[nodiscard]] inline double Ex08HintResidual(const ch02::ZeroCurve& c) {
  const auto r = [&](double t) { return ZeroRateLinear(c, t); };
  const double x = r(5.0 / 3.0);
  const double r76 = (3.0 * r(1.0) + x) / 4.0;
  return 3.0 * std::exp(-r(1.0 / 6.0) / 6.0) +
         3.0 * std::exp(-2.0 * r(2.0 / 3.0) / 3.0) +
         3.0 * std::exp(-7.0 * r76 / 6.0) + 103.0 * std::exp(-5.0 * x / 3.0) -
         103.0;
}

}  // namespace p4mfe::ch05::ex
