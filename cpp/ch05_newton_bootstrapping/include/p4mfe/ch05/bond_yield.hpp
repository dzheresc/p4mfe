#pragma once
// Chapter 5, section 5.3: computing bond yields.
//
// The yield y of a bond with cash flows c_i at t_i and price B solves
//   f(y) = sum_i c_i e^(-y t_i) - B = 0,
//   f'(y) = -sum_i t_i c_i e^(-y t_i)  (minus the dollar duration).
// f is decreasing and convex, so Newton's method from y_0 = 0.1 converges for
// any realistic price (the book's choice of initial guess). The secant and
// bisection methods of section 5.1 are given for comparison.
//
// Pricing comes from chapter 2 (bond_price, dollar_duration); chapter 2's
// own bond_yield is a hand-rolled Newton loop and must agree with these.

#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/solvers.hpp>

namespace p4mfe::ch05 {

using ch02::CashFlows;

inline constexpr Tolerances kBondYieldTolerances{1e-10, 1e-12, 100};

[[nodiscard]] inline RootResult BondYieldNewton(
    const CashFlows& cf, double price, double y0 = 0.1,
    const Tolerances& tol = kBondYieldTolerances) {
  return Newton([&](double y) { return ch02::BondPrice(cf, y) - price; },
                [&](double y) { return -ch02::DollarDuration(cf, y); }, y0,
                tol);
}

[[nodiscard]] inline RootResult BondYieldSecant(
    const CashFlows& cf, double price, double y_prev = 0.11, double y0 = 0.1,
    const Tolerances& tol = kBondYieldTolerances) {
  return Secant([&](double y) { return ch02::BondPrice(cf, y) - price; },
                y_prev, y0, tol);
}

// Bisection on [lo, hi]; the default bracket covers yields from -50% to 100%.
[[nodiscard]] inline RootResult BondYieldBisection(
    const CashFlows& cf, double price, double lo = -0.5, double hi = 1.0,
    const Tolerances& tol = kBondYieldTolerances) {
  return Bisection([&](double y) { return ch02::BondPrice(cf, y) - price; }, lo,
                   hi, tol);
}

}  // namespace p4mfe::ch05
