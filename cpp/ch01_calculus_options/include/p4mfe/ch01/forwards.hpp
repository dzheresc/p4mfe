#pragma once
// Chapter 1, section 1.10: forward and futures contracts.
//
// With a constant interest rate, forward and futures prices coincide, so the
// functions below serve for both.

#include <cmath>
#include <p4mfe/ch01/options.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch01 {

// Forward price for delivery at T of an asset with continuous dividend yield q:
//   F = S e^((r - q) T)
[[nodiscard]] inline double ForwardPrice(double spot, double maturity, double r,
                                         double q = 0.0) noexcept {
  return spot * std::exp((r - q) * maturity);
}

struct Dividend {
  double time;    // payment date, 0 < time <= T
  double amount;  // cash amount
};

// Forward price of an asset paying known discrete dividends before T:
//   F = (S - sum_i D_i e^(-r t_i)) e^(r T)
[[nodiscard]] inline double ForwardPrice(double spot, double maturity, double r,
                                         std::span<const Dividend> dividends) {
  double pv = 0.0;
  for (const auto& d : dividends) {
    if (d.time < 0.0 || d.time > maturity)
      throw std::invalid_argument("forward_price: dividend outside [0, T]");
    pv += d.amount * DiscountFactor(r, d.time);
  }
  return (spot - pv) * std::exp(r * maturity);
}

// Value today of a long forward contract with delivery price K and time to
// maturity T: S e^(-qT) - K e^(-rT) = (F - K) e^(-rT). Zero when K = F.
[[nodiscard]] inline double LongForwardValue(double spot, double strike,
                                             double maturity, double r,
                                             double q = 0.0) noexcept {
  return spot * std::exp(-q * maturity) - strike * DiscountFactor(r, maturity);
}

// ------------------------------------------------------------ 1.10.1 -------
// Long a call and short a put with the same strike K and maturity T is a
// synthetic long forward with delivery price K: C - P = long_forward_value.
// Put-Call parity is that identity, and setting C = P recovers K = F.
[[nodiscard]] inline double SyntheticForwardValue(double call,
                                                  double put) noexcept {
  return call - put;
}

}  // namespace p4mfe::ch01
