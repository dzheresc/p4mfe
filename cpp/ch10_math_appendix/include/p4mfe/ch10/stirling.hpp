#pragma once
// Chapter 10, section 10.7: Stirling's formula.
//
//   n! ~ sqrt(2 pi n) (n / e)^n,   i.e.  n! / (sqrt(2 pi n) (n/e)^n) -> 1,
// with the relative error O(1/n). More precisely (Robbins),
//   sqrt(2 pi n) (n/e)^n e^(1/(12n+1)) < n! < sqrt(2 pi n) (n/e)^n e^(1/(12n)),
// and the asymptotic series
//   ln n! = n ln n - n + ln(2 pi n)/2 + 1/(12n) - 1/(360n^3) + 1/(1260n^5) - ...
// A consequence used for binomial trees: C(2n, n) ~ 4^n / sqrt(pi n)
// (4^n overflows a double for n > 511).

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace p4mfe::ch10 {

// sqrt(2 pi n) (n/e)^n for real n > 0.
[[nodiscard]] inline double stirling(double n) {
    if (!(n > 0.0)) throw std::domain_error("stirling: need n > 0");
    return std::sqrt(2.0 * std::numbers::pi * n) * std::pow(n / std::numbers::e, n);
}

// ln n! from Stirling's series with `terms` correction terms (0..5).
[[nodiscard]] inline double log_stirling(double n, unsigned terms = 0) {
    if (!(n > 0.0)) throw std::domain_error("log_stirling: need n > 0");
    // B_{2k} / (2k (2k-1)): 1/12, -1/360, 1/1260, -1/1680, 1/1188.
    static constexpr std::array<double, 5> c{1.0 / 12, -1.0 / 360, 1.0 / 1260, -1.0 / 1680, 1.0 / 1188};
    if (terms > c.size()) throw std::invalid_argument("log_stirling: at most 5 correction terms");
    double v = n * std::log(n) - n + 0.5 * std::log(2.0 * std::numbers::pi * n);
    double p = n;  // n^(2k-1)
    for (unsigned k = 0; k < terms; ++k, p *= n * n) v += c[k] / p;
    return v;
}

// ln n! exactly (via lgamma), for comparison.
[[nodiscard]] inline double log_factorial(double n) { return std::lgamma(n + 1.0); }

// Robbins' bounds (lower, upper) on n!, n >= 1.
[[nodiscard]] inline std::pair<double, double> stirling_bounds(double n) {
    const double s = stirling(n);
    return {s * std::exp(1.0 / (12.0 * n + 1.0)), s * std::exp(1.0 / (12.0 * n))};
}

// C(2n, n) exactly (as a double, via lgamma) and its Stirling approximation.
[[nodiscard]] inline double central_binomial(double n) {
    return std::exp(log_factorial(2.0 * n) - 2.0 * log_factorial(n));
}
[[nodiscard]] inline double central_binomial_stirling(double n) {
    return std::pow(4.0, n) / std::sqrt(std::numbers::pi * n);
}

}  // namespace p4mfe::ch10
