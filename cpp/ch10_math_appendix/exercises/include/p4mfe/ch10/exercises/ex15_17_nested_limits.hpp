#pragma once
// Exercises 15-17 (section 10.10): an infinite nested radical, an infinite
// continued fraction and an infinite power tower, as limits of iterations.

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace p4mfe::ch10::ex {

// ----------------------------------------------------------- Exercise 15 ---
// For a > 0 compute sqrt(a + sqrt(a + sqrt(a + ...))).
//
// The expression means the limit of x_1 = sqrt(a), x_{n+1} = sqrt(a + x_n).
// Let L = (1 + sqrt(1 + 4a))/2, the positive root of L^2 = L + a.
// Increasing: x_2 = sqrt(a + sqrt a) > sqrt a = x_1, and x_{n+1} > x_n implies
// x_{n+2} = sqrt(a + x_{n+1}) > sqrt(a + x_n) = x_{n+1}.
// Bounded: x_1 = sqrt a < L (as L^2 = L + a > a), and x_n < L implies
// x_{n+1} < sqrt(a + L) = L.
// So x_n converges, and its limit x > 0 satisfies x = sqrt(a + x), x^2 - x - a = 0:
//   sqrt(a + sqrt(a + ...)) = (1 + sqrt(1 + 4a)) / 2.
// E.g. a = 2 gives 2, a = 6 gives 3, a = 1 gives the golden ratio.
// Convergence is linear: L - x_{n+1} = (L - x_n) / (L + x_{n+1}) ~ (L - x_n) / (2L).
[[nodiscard]] inline double ex15_limit(double a) {
    if (!(a > 0.0)) throw std::domain_error("ex15: need a > 0");
    return 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * a));
}
// x_n (n >= 1 radicals).
[[nodiscard]] inline double ex15_iterate(double a, unsigned n) {
    double x = 0.0;
    for (unsigned i = 0; i < n; ++i) x = std::sqrt(a + x);
    return x;
}

// ----------------------------------------------------------- Exercise 16 ---
// For a > 0 compute a + 1/(a + 1/(a + ...)).
//
// The value is the limit of x_1 = a, x_{n+1} = a + 1/x_n. All x_n >= a > 0, so
// x_{n+1} x_n = a x_n + 1 >= a^2 + 1 and
//   |x_{n+2} - x_{n+1}| = |1/x_{n+1} - 1/x_n| = |x_{n+1} - x_n| / (x_n x_{n+1})
//                      <= |x_{n+1} - x_n| / (1 + a^2).
// The steps shrink geometrically, so (x_n) is Cauchy and converges; the limit
// L >= a > 0 satisfies L = a + 1/L, L^2 - a L - 1 = 0:
//   a + 1/(a + 1/(a + ...)) = (a + sqrt(a^2 + 4)) / 2.
// E.g. a = 1 gives the golden ratio, a = 2 gives 1 + sqrt 2. The iterates
// alternate around L (odd ones below, even ones above).
[[nodiscard]] inline double ex16_limit(double a) {
    if (!(a > 0.0)) throw std::domain_error("ex16: need a > 0");
    return 0.5 * (a + std::sqrt(a * a + 4.0));
}
// x_n (n >= 1 levels; x_1 = a).
[[nodiscard]] inline double ex16_iterate(double a, unsigned n) {
    if (n == 0) throw std::invalid_argument("ex16: need n >= 1");
    double x = a;
    for (unsigned i = 1; i < n; ++i) x = a + 1.0 / x;
    return x;
}

// ----------------------------------------------------------- Exercise 17 ---
// (i) Find x > 0 with x^(x^(x^...)) = 2.
// (ii) Find the largest x > 0 for which x^(x^(x^...)) = b for some b > 0, and
// the largest possible b.
//
// The tower is the limit of t_1 = x, t_{n+1} = x^(t_n). If it equals b, then
// b = x^b, i.e. x = b^(1/b).
//
// (i) b = 2 gives x = 2^(1/2) = sqrt 2. Check that the tower of sqrt 2 does
// converge to 2: t_1 = sqrt 2 < 2, and t_n < 2 implies t_{n+1} = sqrt2^(t_n)
// < sqrt2^2 = 2; t_2 = sqrt2^sqrt2 > sqrt2 = t_1 and the map t -> sqrt2^t is
// increasing, so t_n increases. Its limit t <= 2 solves t = sqrt2^t, whose
// solutions are 2 and 4 (t ln 2 / 2 = ln t), so t = 2. Note: b = 4 would
// also give x = 4^(1/4) = sqrt 2, but the tower of sqrt 2 is 2, not 4, so
// "= 4" has no solution.
//
// (ii) x = b^(1/b) = e^(ln b / b). The function g(b) = ln b / b has
// g'(b) = (1 - ln b)/b^2, positive for b < e and negative for b > e, so its
// maximum is g(e) = 1/e. Hence x <= e^(1/e) = 1.444667..., and x = e^(1/e)
// is attained with b = e: for x = e^(1/e) the iterates increase and stay
// below e (t < e implies x^t < x^e = e), and the only solution of t = x^t is
// t = e. For x > e^(1/e) the equation t = x^t has no solution and the
// tower diverges to infinity.
// The largest b: for x in (1, e^(1/e)] the tower converges to the smaller
// root of b = x^b, which is <= e, with b = e at x = e^(1/e). (For x < 1 the
// tower, when it converges, has b < 1.) So the largest b is e.
// At x = e^(1/e) the convergence is slow: x^t = e e^((t-e)/e) ~ t + (t - e)^2 / (2e),
// so e - t_n ~ 2e / n.

inline constexpr double ex17_x_max = 1.444667861009766;  // e^(1/e)

[[nodiscard]] inline double ex17_x_for(double b) {
    if (!(b > 0.0)) throw std::domain_error("ex17: need b > 0");
    return std::pow(b, 1.0 / b);
}
// t_n of the tower (n >= 1 levels). Stops early (returning +infinity) once
// t exceeds `cap`, the signature of divergence.
[[nodiscard]] inline double ex17_tower(double x, unsigned n, double cap = 1e6) {
    if (!(x > 0.0) || n == 0) throw std::invalid_argument("ex17: need x > 0 and n >= 1");
    double t = x;
    for (unsigned i = 1; i < n; ++i) {
        t = std::pow(x, t);
        if (t > cap) return HUGE_VAL;
    }
    return t;
}

}  // namespace p4mfe::ch10::ex
