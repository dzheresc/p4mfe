#pragma once
// Chapter 10, section 10.5: the "Big O" and "little o" notations.
//
// As x -> x_0 (or x -> infinity):
//   f = O(g)  if |f(x)| <= C |g(x)| for x close to x_0 (|f/g| stays bounded),
//   f = o(g)  if f(x) / g(x) -> 0.
// So e^h = 1 + h + O(h^2) = 1 + h + o(h) as h -> 0, and
// sum_{k=1}^{n} k^2 = n^3/3 + O(n^2) as n -> infinity. Error terms of
// numerical methods are stated this way (Simpson's rule is O(h^4), central
// differences O(h^2)). Numerically, if f(h) ~ C h^p then
//   p ~ log2(|f(h)| / |f(h/2)|)   (h -> 0),
//   p ~ log2(|f(2n)| / |f(n)|)    (n -> infinity),
// and the ratios |f/g| along a sequence show boundedness or decay to zero.

#include <cmath>
#include <concepts>
#include <span>
#include <vector>

namespace p4mfe::ch10 {

// Observed exponent p of f(h) ~ C h^p as h -> 0, from h and h/2.
template <std::invocable<double> F>
[[nodiscard]] double observed_power(F&& f, double h) {
    return std::log2(std::abs(f(h)) / std::abs(f(0.5 * h)));
}

// Observed exponent p of f(n) ~ C n^p as n -> infinity, from n and 2n.
template <std::invocable<double> F>
[[nodiscard]] double observed_power_at_infinity(F&& f, double n) {
    return std::log2(std::abs(f(2.0 * n)) / std::abs(f(n)));
}

// |f(x) / g(x)| at each point: bounded for f = O(g), tending to 0 for f = o(g).
template <std::invocable<double> F, std::invocable<double> G>
[[nodiscard]] std::vector<double> order_ratios(F&& f, G&& g, std::span<const double> xs) {
    std::vector<double> r;
    r.reserve(xs.size());
    for (double x : xs) r.push_back(std::abs(f(x) / g(x)));
    return r;
}

}  // namespace p4mfe::ch10
