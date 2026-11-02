#pragma once
// Exercise 9 (section 5.7): bootstrap from five Treasury instruments and
// price the sixth with the resulting curve.

#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/bootstrapping.hpp>
#include <vector>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 9 ---
//   6-month T-bill    0       99.4565
//   12-month T-bill   0       98.6196
//   2-year T-bond     2       101 17.5/32
//   3-year T-bond     4.5     107 18/32
//   5-year T-bond     3.125   102 8/32
//   10-year T-bond    4       103 8.5/32
// Semiannual coupons, continuously compounded interest.
// (i) Bootstrap a zero rate curve from the bills and the 2, 5 and 10-year
//     bonds.
// (ii) Price the 3-year 4.5% bond off that curve and find the relative error
//      to its market price 107 18/32.
//
// (i) Linear interpolation between maturities (section 5.5); every cash flow
// is at 6 months or later, so nothing before the first bill is needed:
//   t      0.5       1         2         5         10
//   r    1.0900%   1.3900%   1.2099%   2.6824%   3.7372%
// (ii) The 3-year bond's coupons at 2.5 and 3 years use rates interpolated
// on [2, 5]:
//   B = 108.193042, market 107.5625, relative error (B - P)/P = +0.586%.
// The curve overprices the bond by about 63 cents: the linear interpolation
// between 2 and 5 years puts the 3-year rate at about 1.70%, while the
// 3-year bond's own price implies higher rates there (its yield is ~1.9%).
// The error measures how far "linear between 2 and 5 years" is from the
// market in that part of the curve.
[[nodiscard]] inline std::vector<ch02::BondQuote> Ex09Quotes() {
  return {{0.5, 0.0, 99.4565},
          {1.0, 0.0, 98.6196},
          {2.0, 0.02, 101.0 + 17.5 / 32.0},
          {5.0, 0.03125, 102.0 + 8.0 / 32.0},
          {10.0, 0.04, 103.0 + 8.5 / 32.0}};
}
inline constexpr ch02::BondQuote kEx09ThreeYear{3.0, 0.045,
                                                107.0 + 18.0 / 32.0};

[[nodiscard]] inline ch02::ZeroCurve Ex09Curve() {
  return BootstrapZeroCurve(Ex09Quotes(), 2);
}

[[nodiscard]] inline double Ex09ModelPrice(const ch02::ZeroCurve& c) {
  const auto cf = ch02::BondCashFlows(kEx09ThreeYear.maturity,
                                      kEx09ThreeYear.coupon_rate, 2);
  return ch02::BondPriceFromZeroRate(
      cf, [&](double t) { return ZeroRateLinear(c, t); });
}
[[nodiscard]] inline double Ex09RelativeError(const ch02::ZeroCurve& c) {
  return (Ex09ModelPrice(c) - kEx09ThreeYear.price) / kEx09ThreeYear.price;
}

}  // namespace p4mfe::ch05::ex
