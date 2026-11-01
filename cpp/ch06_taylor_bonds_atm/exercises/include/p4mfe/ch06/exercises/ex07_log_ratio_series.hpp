#pragma once
// Exercise 7 (section 6.11): the Taylor series of ln((1 + x)/(1 - x)) and its
// radius of convergence.

#include <p4mfe/ch06/taylor_series.hpp>

#include <cmath>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 7 ---
// Compute the Taylor series expansion of ln((1 + x)/(1 - x)) around 0, and
// find its radius of convergence.
//
// ln((1 + x)/(1 - x)) = ln(1 + x) - ln(1 - x), and for |x| < 1
//   ln(1 + x) = sum_{k>=1} (-1)^(k+1) x^k / k,   ln(1 - x) = -sum_{k>=1} x^k / k.
// Subtracting, the even powers cancel and the odd ones double:
//   ln((1 + x)/(1 - x)) = 2 sum_{j>=0} x^(2j+1) / (2j + 1) = 2 (x + x^3/3 + x^5/5 + ...)
// (= 2 artanh x). Radius of convergence: by the ratio test on the nonzero
// terms, |a_{j+1}/a_j| = x^2 (2j+1)/(2j+3) -> x^2, so the series converges
// for |x| < 1 and diverges for |x| > 1: R = 1. At x = +-1 it is +-2 times the
// divergent series 1 + 1/3 + 1/5 + ..., matching ln((1+x)/(1-x)) -> +-infinity.
// The radius is 1 also because the function blows up at x = +-1, the
// singularities nearest to 0.
//
// Partial sum with the nonzero terms j = 0..n.
[[nodiscard]] inline double ex07_series(double x, unsigned n) noexcept {
    double power = x, sum = 0.0;
    for (unsigned j = 0; j <= n; ++j) {
        sum += power / (2.0 * j + 1.0);
        power *= x * x;
    }
    return 2.0 * sum;
}
// The same from chapter 6's ln(1 + x) partial sums (degree 2n + 1), |x| < 1.
[[nodiscard]] inline double ex07_series_from_log1p(double x, unsigned n) {
    return log1p_series(x, 2 * n + 1) - log1p_series(-x, 2 * n + 1);
}
[[nodiscard]] inline double ex07_function(double x) { return std::log1p(x) - std::log1p(-x); }
inline constexpr double ex07_radius = 1.0;

}  // namespace p4mfe::ch06::ex
