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
[[nodiscard]] double shifted_curve_price(const CashFlows& cf, Z&& zero,
                                         double dr) {
  return ch02::bond_price_from_zero_rate(
      cf, [&](double t) { return zero(t) + dr; });
}

// Price, duration and convexity with respect to a parallel shift of the curve.
template <std::invocable<double> Z>
[[nodiscard]] BondMetrics curve_shift_metrics(const CashFlows& cf, Z&& zero) {
  double B = 0.0, Bt = 0.0, Btt = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    const double t = cf.times[i];
    const double v = cf.amounts[i] * std::exp(-zero(t) * t);
    B += v;
    Bt += t * v;
    Btt += t * t * v;
  }
  return {B, Bt / B, Btt / B};
}

// Second-order Taylor estimate of B(dr) (chapter 2's duration-convexity
// formula).
template <std::invocable<double> Z>
[[nodiscard]] double shifted_curve_price_approx(const CashFlows& cf, Z&& zero,
                                                double dr) {
  const auto m = curve_shift_metrics(cf, zero);
  return ch02::price_approx_duration_convexity(m.price, m.duration, m.convexity,
                                               dr);
}

// Change in the bond's yield caused by a parallel shift dr of the zero curve.
template <std::invocable<double> Z>
[[nodiscard]] double yield_change_from_shift(const CashFlows& cf, Z&& zero,
                                             double dr) {
  const double y0 =
      ch02::bond_yield(cf, ch02::bond_price_from_zero_rate(cf, zero));
  const double y1 = ch02::bond_yield(cf, shifted_curve_price(cf, zero, dr));
  return y1 - y0;
}

}  // namespace p4mfe::ch06
