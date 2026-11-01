#pragma once
// Standard normal density and cumulative distribution, used throughout the
// book (Black-Scholes, Greeks, implied volatility, ...).

#include <cmath>
#include <numbers>

namespace p4mfe {

// Standard normal density: phi(x) = exp(-x^2/2) / sqrt(2*pi).
[[nodiscard]] inline double norm_pdf(double x) noexcept {
    return std::exp(-0.5 * x * x) * std::numbers::inv_sqrtpi / std::numbers::sqrt2;
}

// Standard normal cumulative distribution N(x), via the complementary error
// function so that the left tail keeps full relative precision.
[[nodiscard]] inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

}  // namespace p4mfe
