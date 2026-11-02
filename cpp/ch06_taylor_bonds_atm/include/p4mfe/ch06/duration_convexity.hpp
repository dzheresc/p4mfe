#pragma once
// Chapter 6, section 6.6: connections between duration and convexity.
//
// With B(y) = sum_i c_i e^(-y t_i), duration and convexity are the scaled
// first two derivatives of the price,
//   D = -B'(y) / B(y),   C = B''(y) / B(y),
// so they are the first two moments of the cash-flow times t_i under the
// weights w_i = c_i e^(-y t_i) / B (positive, summing to 1):
//   D = sum_i w_i t_i,   C = sum_i w_i t_i^2.
// Hence:
//   * C - D^2 = sum_i w_i (t_i - D)^2 >= 0, the dispersion of the cash flows
//     around D (zero only for a zero coupon bond, where D = T, C = T^2);
//   * differentiating D(y) = -B'/B,  dD/dy = (B'/B)^2 - B''/B = D^2 - C <= 0,
//     so duration decreases as the yield increases;
//   * the value-weighted average of durations (convexities) of bonds is the
//     duration (convexity) of the portfolio, as long as all are computed at
//     one common yield.
//
// For a yield compounded m times a year, B(y) = sum c_i (1 + y/m)^(-m t_i):
//   modified duration  D_mod = -B'/B = D_mac / (1 + y/m)        (chapter 2)
//   convexity          C_m   =  B''/B = (1/B) sum c_i t_i (t_i + 1/m) (1 +
//   y/m)^(-m t_i - 2),
// and D_mod -> D, C_m -> C as m -> infinity.

#include <cstddef>
#include <p4mfe/ch02/bonds.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch06 {

using ch02::CashFlows;

// dD/dy = D^2 - C.
[[nodiscard]] constexpr double duration_derivative(double D,
                                                   double C) noexcept {
  return D * D - C;
}

// C - D^2 = sum_i w_i (t_i - D)^2.
[[nodiscard]] inline double cash_flow_dispersion(const CashFlows& cf,
                                                 double y) {
  const auto m = ch02::bond_metrics(cf, y);
  return m.convexity - m.duration * m.duration;
}

// Convexity for a yield compounded m times a year.
[[nodiscard]] inline double convexity_discrete(const CashFlows& cf, double y,
                                               int m) {
  if (m <= 0)
    throw std::invalid_argument("convexity_discrete: m must be positive");
  const double g = 1.0 + y / m;
  double B = 0.0, B2 = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    const double t = cf.times[i];
    const double v = cf.amounts[i] * ch02::discount_factor_discrete(y, t, m);
    B += v;
    B2 += v * t * (t + 1.0 / m) / (g * g);
  }
  return B2 / B;
}

// Value-weighted averages: D_P = sum V_i D_i / sum V_i (likewise convexity).
[[nodiscard]] inline double portfolio_average(std::span<const double> values,
                                              std::span<const double> x) {
  if (values.size() != x.size() || values.empty())
    throw std::invalid_argument(
        "portfolio_average: non-empty, same-size inputs required");
  double V = 0.0, s = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    V += values[i];
    s += values[i] * x[i];
  }
  if (V == 0.0)
    throw std::invalid_argument(
        "portfolio_average: zero total value (use dollar duration)");
  return s / V;
}
[[nodiscard]] inline double portfolio_duration(
    std::span<const double> values, std::span<const double> durations) {
  return portfolio_average(values, durations);
}
[[nodiscard]] inline double portfolio_convexity(
    std::span<const double> values, std::span<const double> convexities) {
  return portfolio_average(values, convexities);
}

}  // namespace p4mfe::ch06
