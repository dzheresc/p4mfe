#pragma once
// Exercises 2-3 (section 8.12): double integrals over a triangle and over the
// region between a parabola and a line.

#include <p4mfe/ch08/double_integrals.hpp>

#include <cmath>
#include <cstddef>

namespace p4mfe::ch08::ex {

// ------------------------------------------------------------ Exercise 2 ---
// Let D be the domain bounded by the x-axis, the y-axis and the line x + y = 1.
// Compute the integral over D of (x - y)/(x + y) dx dy.
//
// Change variables s = x + y, t = x - y, i.e. x = (s + t)/2, y = (s - t)/2.
// The Jacobian of (s, t) -> (x, y) is
//   det [[1/2, 1/2], [1/2, -1/2]] = -1/2,   so dx dy = 1/2 ds dt.
// (x, y) is in D iff x >= 0, y >= 0, x + y <= 1 iff 0 <= s <= 1 and -s <= t <= s.
// Therefore
//   integral_D (x - y)/(x + y) dx dy = integral_0^1 integral_{-s}^{s} (t/s) (1/2) dt ds
//                                    = integral_0^1 (1/(2s)) [t^2/2]_{-s}^{s} ds = 0.
// The inner integral vanishes because t/s is odd in t on the symmetric
// interval [-s, s]. Equivalently: D is symmetric under x <-> y and the
// integrand changes sign, so the integral is 0. (The integrand is bounded,
// |x - y| <= x + y on D, so the integral exists despite the corner at 0.)
[[nodiscard]] inline double ex02_integrand(double x, double y) { return x + y > 0.0 ? (x - y) / (x + y) : 0.0; }

inline constexpr double ex02_exact = 0.0;

// Directly over D: 0 <= x <= 1, 0 <= y <= 1 - x.
[[nodiscard]] inline double ex02_direct(std::size_t n = 200) {
    return double_integral_region(
        ex02_integrand, 0.0, 1.0, [](double) { return 0.0; }, [](double x) { return 1.0 - x; }, n, n);
}

// In (s, t): integral_0^1 integral_{-s}^{s} (t/s) |det J| dt ds, |det J| = 1/2.
[[nodiscard]] inline double ex02_change_of_variables(std::size_t n = 200) {
    return double_integral_region(
        [](double s, double t) { return s > 0.0 ? 0.5 * t / s : 0.0; }, 0.0, 1.0, [](double s) { return -s; },
        [](double s) { return s; }, n, n);
}

// The integral of |x - y|/(x + y) over D, which is not 0: by the same change
// of variables it is integral_0^1 (1/(2s)) * s^2 ds = 1/4. A check that the
// quadrature resolves the integrand and not just the symmetry.
inline constexpr double ex02_abs_exact = 0.25;
[[nodiscard]] inline double ex02_abs_direct(std::size_t n = 200) {
    return double_integral_region(
        [](double x, double y) { return x + y > 0.0 ? std::abs(x - y) / (x + y) : 0.0; }, 0.0, 1.0,
        [](double) { return 0.0; }, [](double x) { return 1.0 - x; }, n, n);
}

// ------------------------------------------------------------ Exercise 3 ---
// Compute the integral of f(x, y) = x^2 - 2y on the region bounded by the
// parabola y = (x + 1)^2 and the line y = 5x - 1.
//
// Intersections: (x + 1)^2 = 5x - 1 <=> x^2 - 3x + 2 = 0 <=> x = 1 or x = 2
// (points (1, 4) and (2, 9)). For 1 < x < 2 the line is above the parabola:
// 5x - 1 - (x + 1)^2 = -(x - 1)(x - 2) > 0. So the region is
//   1 <= x <= 2,   (x + 1)^2 <= y <= 5x - 1,
// and
//   integral_1^2 integral_{(x+1)^2}^{5x-1} (x^2 - 2y) dy dx
//     = integral_1^2 [x^2 y - y^2]_{(x+1)^2}^{5x-1} dx
//     = integral_1^2 ( x^2 (5x - 1 - (x + 1)^2) - (5x - 1)^2 + (x + 1)^4 ) dx.
// Expanding: x^2(-x^2 + 3x - 2) = -x^4 + 3x^3 - 2x^2,
//            (x + 1)^4 - (5x - 1)^2 = x^4 + 4x^3 - 19x^2 + 14x,
// so the inner integral is 7x^3 - 21x^2 + 14x = 7x(x - 1)(x - 2), and
//   integral_1^2 (7x^3 - 21x^2 + 14x) dx = [7x^4/4 - 7x^3 + 7x^2]_1^2
//     = (28 - 56 + 28) - (7/4 - 7 + 7) = -7/4.
// The answer is -7/4 = -1.75.
[[nodiscard]] inline double ex03_integrand(double x, double y) { return x * x - 2.0 * y; }
[[nodiscard]] inline double ex03_inner(double x) { return 7.0 * x * x * x - 21.0 * x * x + 14.0 * x; }

inline constexpr double ex03_exact = -1.75;

[[nodiscard]] inline double ex03_numeric(std::size_t n = 100) {
    return double_integral_region(
        ex03_integrand, 1.0, 2.0, [](double x) { return (x + 1.0) * (x + 1.0); },
        [](double x) { return 5.0 * x - 1.0; }, n, n);
}

}  // namespace p4mfe::ch08::ex
