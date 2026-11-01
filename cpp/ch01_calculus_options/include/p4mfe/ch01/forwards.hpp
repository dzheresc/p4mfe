#pragma once
// Chapter 1, section 1.10: forward and futures contracts.
//
// With a constant interest rate, forward and futures prices coincide, so the
// functions below serve for both.

#include <p4mfe/ch01/options.hpp>

#include <cmath>
#include <span>
#include <stdexcept>

namespace p4mfe::ch01 {

// Forward price for delivery at T of an asset with continuous dividend yield q:
//   F = S e^((r - q) T)
[[nodiscard]] inline double forward_price(double S, double T, double r, double q = 0.0) noexcept {
    return S * std::exp((r - q) * T);
}

struct Dividend {
    double time;    // payment date, 0 < time <= T
    double amount;  // cash amount
};

// Forward price of an asset paying known discrete dividends before T:
//   F = (S - sum_i D_i e^(-r t_i)) e^(r T)
[[nodiscard]] inline double forward_price(double S, double T, double r,
                                          std::span<const Dividend> dividends) {
    double pv = 0.0;
    for (const auto& d : dividends) {
        if (d.time < 0.0 || d.time > T) throw std::invalid_argument("forward_price: dividend outside [0, T]");
        pv += d.amount * discount_factor(r, d.time);
    }
    return (S - pv) * std::exp(r * T);
}

// Value today of a long forward contract with delivery price K and time to
// maturity T: S e^(-qT) - K e^(-rT) = (F - K) e^(-rT). Zero when K = F.
[[nodiscard]] inline double long_forward_value(double S, double K, double T, double r,
                                               double q = 0.0) noexcept {
    return S * std::exp(-q * T) - K * discount_factor(r, T);
}

// ------------------------------------------------------------ 1.10.1 -------
// Long a call and short a put with the same strike K and maturity T is a
// synthetic long forward with delivery price K: C - P = long_forward_value.
// Put-Call parity is that identity, and setting C = P recovers K = F.
[[nodiscard]] inline double synthetic_forward_value(double C, double P) noexcept { return C - P; }

}  // namespace p4mfe::ch01
