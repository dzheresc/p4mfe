#pragma once
// Chapter 6, section 6.5: parallel shifts in the yield curve.
//
// A bond priced off a zero rate curve r(0, t),
//   B = sum_i c_i e^(-r(0, t_i) t_i),
// after a parallel shift r(0, t) -> r(0, t) + dr is worth
//   B(dr) = sum_i c_i e^(-r(0, t_i) t_i) e^(-dr t_i).
// Taylor's formula around dr = 0 gives
//   B(dr) ~ B (1 - D dr + C dr^2 / 2),
// with the duration and convexity with respect to the shift
//   D = (1/B) sum_i t_i c_i e^(-r(0, t_i) t_i),
//   C = (1/B) sum_i t_i^2 c_i e^(-r(0, t_i) t_i).
// For a flat curve r(0, t) = y these are the yield duration and convexity of
// chapter 2 (section 2.7), and the bond's yield moves by exactly dr. For a
// curve that is not flat the yield change is close to, but not exactly, dr.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <p4mfe/ch02/bonds.hpp>

namespace p4mfe::ch06 {

using ch02::BondMetrics;
using ch02::CashFlows;

// B(dr) = sum_i c_i e^(-(r(0, t_i) + dr) t_i).
template <std::invocable<double> Z>
[[nodiscard]] double ShiftedCurvePrice(const CashFlows& cf, Z&& zero,
                                       double dr) {
  return ch02::BondPriceFromZeroRate(cf,
                                     [&](double t) { return zero(t) + dr; });
}

// Price, duration and convexity with respect to a parallel shift of the curve.
template <std::invocable<double> Z>
[[nodiscard]] BondMetrics CurveShiftMetrics(const CashFlows& cf, Z&& zero) {
  double price = 0.0, price_t = 0.0, price_tt = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    const double t = cf.times[i];
    const double v = cf.amounts[i] * std::exp(-zero(t) * t);
    price += v;
    price_t += t * v;
    price_tt += t * t * v;
  }
  return {price, price_t / price, price_tt / price};
}

// Second-order Taylor estimate of B(dr) (chapter 2's duration-convexity
// formula).
template <std::invocable<double> Z>
[[nodiscard]] double ShiftedCurvePriceApprox(const CashFlows& cf, Z&& zero,
                                             double dr) {
  const auto m = CurveShiftMetrics(cf, zero);
  return ch02::PriceApproxDurationConvexity(m.price, m.duration, m.convexity,
                                            dr);
}

// Change in the bond's yield caused by a parallel shift dr of the zero curve.
template <std::invocable<double> Z>
[[nodiscard]] double YieldChangeFromShift(const CashFlows& cf, Z&& zero,
                                          double dr) {
  const double y0 = ch02::BondYield(cf, ch02::BondPriceFromZeroRate(cf, zero));
  const double y1 = ch02::BondYield(cf, ShiftedCurvePrice(cf, zero, dr));
  return y1 - y0;
}

}  // namespace p4mfe::ch06
