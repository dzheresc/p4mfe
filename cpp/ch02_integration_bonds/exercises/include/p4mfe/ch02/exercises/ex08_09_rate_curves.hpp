#pragma once
// Exercises 8-9 (section 2.12): zero rate and instantaneous rate curves,
// parallel shifts, and converting a zero curve to other compounding
// frequencies.

#include <cmath>
#include <concepts>
#include <p4mfe/ch02/rates.hpp>

namespace p4mfe::ch02::ex {

// ------------------------------------------------------------ Exercise 8 ---
// Show that a parallel shift in the zero rate curve generates an identical
// parallel shift in the instantaneous rate curve, and vice versa: if r1(0,t),
// r2(0,t) are zero rate curves with instantaneous curves r1(t), r2(t), and
// dr > 0, then
//   r2(0, t) = r1(0, t) + dr   for all t   <=>   r2(t) = r1(t) + dr   for all
//   t.
//
// Solution. The curves are linked by (section 2.5)
//   t r(0, t) = integral_0^t r(s) ds,  i.e.  r(t) = d/dt (t r(0, t)).
// (=>) If r2(0,t) = r1(0,t) + dr then t r2(0,t) = t r1(0,t) + t dr, and
//      differentiating, r2(t) = r1(t) + dr.
// (<=) If r2(s) = r1(s) + dr then integral_0^t r2(s) ds = integral_0^t r1(s) ds
// + t dr;
//      divide by t: r2(0,t) = r1(0,t) + dr.
// (Nothing uses dr > 0; the same holds for any constant shift.)

// A zero curve shifted by dr, and its instantaneous curve computed
// numerically from r(t) = d/dt (t r(0,t)); the test checks the difference
// is dr for an arbitrary curve.
template <std::invocable<double> Z>
[[nodiscard]] auto ex08_shift(Z zero, double dr) {
  return [zero, dr](double t) { return zero(t) + dr; };
}

// ------------------------------------------------------------ Exercise 9 ---
// The continuously compounded zero rate curve is
//   r_c(0, t) = 0.05 + 0.01 ln(1 + t/2).
// (i)   Find the instantaneous interest rate curve.
// (ii)  Compute the corresponding annually compounded zero rate curve.
// (iii) Compute the corresponding semiannually compounded zero rate curve.
//
// (i) r(t) = d/dt (t r_c(0,t)) = r_c(0,t) + t d/dt r_c(0,t)
//          = 0.05 + 0.01 ln(1 + t/2) + 0.01 t / (2 + t).
// (ii) A rate r_1 compounded annually gives the same discount factor iff
//   (1 + r_1(0,t))^(-t) = e^(-r_c(0,t) t)  <=>  r_1(0,t) = e^(r_c(0,t)) - 1
//                                                   = e^0.05 (1 + t/2)^0.01
//                                                   - 1.
// (iii) Semiannually: (1 + r_2/2)^(-2t) = e^(-r_c t), so
//   r_2(0,t) = 2 (e^(r_c(0,t)/2) - 1) = 2 (e^0.025 (1 + t/2)^0.005 - 1).
// At t = 0 all three curves start at their equivalent of 5%: r(0) = 0.05,
// r_1(0,0) = e^0.05 - 1 = 5.127%, r_2(0,0) = 2(e^0.025 - 1) = 5.063%.
[[nodiscard]] inline double ex09_zero_rate(double t) {
  return 0.05 + 0.01 * std::log1p(0.5 * t);
}
[[nodiscard]] inline double ex09_inst_rate(double t) {
  return ex09_zero_rate(t) + 0.01 * t / (2.0 + t);
}
[[nodiscard]] inline double ex09_annual_zero_rate(double t) {
  return std::exp(0.05) * std::pow(1.0 + 0.5 * t, 0.01) - 1.0;
}
[[nodiscard]] inline double ex09_semiannual_zero_rate(double t) {
  return 2.0 * (std::exp(0.025) * std::pow(1.0 + 0.5 * t, 0.005) - 1.0);
}

}  // namespace p4mfe::ch02::ex
