#pragma once
// Exercise 17 (section 6.11): value change of a bond portfolio from DV01 and
// dollar convexity, and its immunization with two bonds.

#include <p4mfe/ch06/dollar_duration.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 17 ---
// A bond portfolio worth $50mil has DV01 = $10,000 and dollar convexity
// $400mil. (i) The yield curve moves up by 50bp. What is the new portfolio
// value? (ii) Bonds available for trading:
//            Principal  Value  Duration  Convexity
//   Bond 1     1000      1100     2.5        12
//   Bond 2      100       106     3           9
// How do you immunize the portfolio?
//
// (i) DV01 = D$ / 10000, so D$ = 10,000 * 10,000 = $100mil, and
//   dV ~ -D$ dy + C$ dy^2 / 2 = -100mil * 0.005 + 400mil * 0.005^2 / 2
//      = -500,000 + 5,000 = -$495,000,
// so the portfolio is worth about $49,505,000 (duration alone: $49,500,000).
//
// (ii) Per bond, D$ = B D and C$ = B C:
//   Bond 1: D$ = 1100 * 2.5 = 2750,  C$ = 1100 * 12 = 13200;
//   Bond 2: D$ =  106 * 3   =  318,  C$ =  106 * 9  =   954.
// Immunizing means choosing q1, q2 units so the hedged position has zero D$
// and zero C$ (6.7.2):
//   2750 q1 +   318 q2 = -100,000,000
//   13200 q1 +  954 q2 = -400,000,000
// whose solution is
//   q1 = -20,202.02,  q2 = -139,762.40:
// sell 20,202 units of Bond 1 (worth $22.22mil) and 139,762 units of Bond 2
// (worth $14.81mil). The hedged position is then insensitive to small parallel
// shifts up to O(dy^3). The remaining $12.96mil of value (50 - 22.22 - 14.81)
// is cash-like: the chapter's `immunize` also returns the cash leg,
// -$12.96mil, that makes the total value zero; cash has no duration or
// convexity, so it does not change the hedge.
struct Ex17Data {
  double value = 50e6, dv01 = 1e4, dollar_convexity = 400e6;
  double price1 = 1100.0, duration1 = 2.5, convexity1 = 12.0;
  double price2 = 106.0, duration2 = 3.0, convexity2 = 9.0;
};

[[nodiscard]] constexpr DollarRisk Ex17Portfolio(
    const Ex17Data& x = {}) noexcept {
  return {x.value, x.dv01 * 10000.0, x.dollar_convexity};
}
[[nodiscard]] constexpr DollarRisk Ex17Bond1(const Ex17Data& x = {}) noexcept {
  return {x.price1, x.price1 * x.duration1, x.price1 * x.convexity1};
}
[[nodiscard]] constexpr DollarRisk Ex17Bond2(const Ex17Data& x = {}) noexcept {
  return {x.price2, x.price2 * x.duration2, x.price2 * x.convexity2};
}
// (i)
[[nodiscard]] constexpr double Ex17NewValue(double dy = 0.005,
                                            const Ex17Data& x = {}) noexcept {
  const auto p = Ex17Portfolio(x);
  return p.value + ValueChangeApproxConvexity(p, dy);
}
// (ii)
[[nodiscard]] inline Immunization Ex17Immunize(const Ex17Data& x = {}) {
  return Immunize(Ex17Portfolio(x), Ex17Bond1(x), Ex17Bond2(x));
}

}  // namespace p4mfe::ch06::ex
