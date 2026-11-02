#pragma once
// Exercise 14 (section 1.12): derivatives of bond price with respect to yield.
//
// Let c_i and t_i, i = 1:n, be positive constants.
// (i)  f(y) = sum_i c_i e^(-y t_i).  Compute f'(y) and f''(y).
// (ii) g(y) = sum_i c_i (1 + y/m)^(-m t_i).  Compute g'(y) and g''(y).
// (Note: f and g are the price of a bond with cash flows c_i at times t_i as
//  a function of its yield, compounded continuously and m times a year.)
//
// Solution. Term by term:
//   f'(y)  = -sum_i c_i t_i e^(-y t_i),
//   f''(y) =  sum_i c_i t_i^2 e^(-y t_i);
// and since d/dy (1 + y/m)^(-m t) = -m t (1 + y/m)^(-m t - 1) * (1/m),
//   g'(y)  = -sum_i c_i t_i (1 + y/m)^(-m t_i - 1),
//   g''(y) =  sum_i c_i t_i (t_i + 1/m) (1 + y/m)^(-m t_i - 2).
// So -f'/f is the duration and f''/f the convexity (section 2.7); -g'/g is the
// modified duration (section 2.8).

#include <cmath>
#include <span>
#include <stdexcept>

namespace p4mfe::ch01::ex {

struct CashFlow {
  double time;    // t_i > 0
  double amount;  // c_i > 0
};

// Value and first two derivatives of a price-yield function.
struct PriceDerivs {
  double value;
  double first;
  double second;
};

// (i) continuously compounded yield.
[[nodiscard]] inline PriceDerivs ex14_continuous(std::span<const CashFlow> cfs,
                                                 double y) {
  PriceDerivs d{0.0, 0.0, 0.0};
  for (const auto& [t, c] : cfs) {
    const double df = c * std::exp(-y * t);
    d.value += df;
    d.first -= t * df;
    d.second += t * t * df;
  }
  return d;
}

// (ii) yield compounded m times a year.
[[nodiscard]] inline PriceDerivs ex14_discrete(std::span<const CashFlow> cfs,
                                               double y, double m) {
  if (m <= 0.0)
    throw std::invalid_argument("ex14_discrete: m must be positive");
  const double base = 1.0 + y / m;
  PriceDerivs d{0.0, 0.0, 0.0};
  for (const auto& [t, c] : cfs) {
    d.value += c * std::pow(base, -m * t);
    d.first -= c * t * std::pow(base, -m * t - 1.0);
    d.second += c * t * (t + 1.0 / m) * std::pow(base, -m * t - 2.0);
  }
  return d;
}

}  // namespace p4mfe::ch01::ex
