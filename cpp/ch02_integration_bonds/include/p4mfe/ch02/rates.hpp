#pragma once
// Chapter 2, sections 2.5 and 2.8: interest rate curves, forward rates, and
// discretely compounded interest.
//
// Conventions: time in years from today (t = 0). Unless a function says
// otherwise, rates are continuously compounded.
//   r(t)       instantaneous (short) rate curve
//   r(0, t)    zero rate curve: the constant rate that grows 1 to the same
//              amount over [0, t]
//   D(0, t)    discount factor: today's value of 1 paid at t
// They are linked by
//   D(0, t) = exp(-integral_0^t r(s) ds) = exp(-r(0, t) t),
//   r(0, t) = (1/t) integral_0^t r(s) ds,   r(t) = d/dt (t r(0, t)).

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>

#include "integration.hpp"

namespace p4mfe::ch02 {

// -------------------------------------------------------------- 2.5.1 ------
// Constant rate r: D(0, t) = e^(-r t).
[[nodiscard]] inline double DiscountFactor(double r, double t) noexcept {
  return std::exp(-r * t);
}

// The zero rate implied by a discount factor: r(0, t) = -ln(D) / t.
[[nodiscard]] inline double ZeroRateFromDiscount(double discount, double t) {
  if (t <= 0.0)
    throw std::domain_error("zero_rate_from_discount: t must be positive");
  if (discount <= 0.0)
    throw std::domain_error("zero_rate_from_discount: D must be positive");
  return -std::log(discount) / t;
}

// ---------------------------------------------------------------- 2.5 ------
// D(0, t) = exp(-integral_0^t r(s) ds) for an instantaneous rate curve r(s).
template <std::invocable<double> R>
[[nodiscard]] double DiscountFactorFromInstRate(R&& r, double t,
                                                double tol = 1e-12) {
  if (t == 0.0) return 1.0;
  return std::exp(-IntegrateToTolerance(r, 0.0, t, tol, Rule::kSimpson).value);
}

// r(0, t) = (1/t) integral_0^t r(s) ds.
template <std::invocable<double> R>
[[nodiscard]] double ZeroRateFromInstRate(R&& r, double t, double tol = 1e-12) {
  if (t <= 0.0)
    throw std::domain_error("zero_rate_from_inst_rate: t must be positive");
  return IntegrateToTolerance(r, 0.0, t, tol, Rule::kSimpson).value / t;
}

// r(t) = d/dt (t r(0, t)), by a central difference that stays inside t > 0.
template <std::invocable<double> Z>
[[nodiscard]] double InstRateFromZeroRate(Z&& zero, double t) {
  if (t <= 0.0)
    throw std::domain_error("inst_rate_from_zero_rate: t must be positive");
  const double h = std::min(
      std::cbrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, t),
      0.5 * t);
  return ((t + h) * zero(t + h) - (t - h) * zero(t - h)) / (2.0 * h);
}

// -------------------------------------------------------------- 2.5.2 ------
// Forward rate r(t1, t2): the rate agreed today for borrowing over [t1, t2].
// No-arbitrage requires D(0, t2) = D(0, t1) exp(-r(t1, t2) (t2 - t1)), so
//   r(t1, t2) = (r(0, t2) t2 - r(0, t1) t1) / (t2 - t1)
//             = ln(D(0, t1) / D(0, t2)) / (t2 - t1)
//             = (1 / (t2 - t1)) integral_{t1}^{t2} r(s) ds.
[[nodiscard]] inline double ForwardRate(double r1, double t1, double r2,
                                        double t2) {
  if (!(t2 > t1)) throw std::invalid_argument("forward_rate: need t2 > t1");
  return (r2 * t2 - r1 * t1) / (t2 - t1);
}

[[nodiscard]] inline double ForwardRateFromDiscounts(double discount1,
                                                     double t1,
                                                     double discount2,
                                                     double t2) {
  if (!(t2 > t1))
    throw std::invalid_argument("forward_rate_from_discounts: need t2 > t1");
  return std::log(discount1 / discount2) / (t2 - t1);
}

template <std::invocable<double> R>
[[nodiscard]] double ForwardRateFromInstRate(R&& r, double t1, double t2,
                                             double tol = 1e-12) {
  if (!(t2 > t1))
    throw std::invalid_argument("forward_rate_from_inst_rate: need t2 > t1");
  return IntegrateToTolerance(r, t1, t2, tol, Rule::kSimpson).value / (t2 - t1);
}

// ---------------------------------------------------------------- 2.8 ------
// Rate r compounded n times a year: D(0, t) = (1 + r/n)^(-n t).
[[nodiscard]] inline double DiscountFactorDiscrete(double r, double t,
                                                   double n) {
  if (n <= 0.0)
    throw std::invalid_argument("discount_factor_discrete: n must be positive");
  return std::pow(1.0 + r / n, -n * t);
}

// Equivalent rates (same growth over any horizon):
//   e^r = (1 + r_n/n)^n  <=>  r = n ln(1 + r_n/n),  r_n = n (e^(r/n) - 1).
[[nodiscard]] inline double ContinuousFromDiscrete(double r_n, double n) {
  if (n <= 0.0)
    throw std::invalid_argument("continuous_from_discrete: n must be positive");
  return n * std::log1p(r_n / n);
}
[[nodiscard]] inline double DiscreteFromContinuous(double r, double n) {
  if (n <= 0.0)
    throw std::invalid_argument("discrete_from_continuous: n must be positive");
  return n * std::expm1(r / n);
}

}  // namespace p4mfe::ch02
