#pragma once
// Exercises 3-4 (section 7.6): a one-sided second order difference for f'(a)
// and the central difference for the fourth derivative.

#include <concepts>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 3 ---
// Find a second order finite difference approximation for f'(a) using f(a),
// f(a + h) and f(a + 2h). (Used, e.g., for derivative boundary conditions such
// as dH/dt + dH/dR = 0 at R = 0 for continuously sampled Asian options.)
//
// Taylor's formula around a:
//   f(a + h)  = f + h f' + h^2/2 f'' + h^3/6 f''' + O(h^4),
//   f(a + 2h) = f + 2h f' + 2h^2 f'' + 4h^3/3 f''' + O(h^4).
// Eliminate f'': 4 f(a + h) - f(a + 2h) = 3 f + 2h f' - 2h^3/3 f''' + O(h^4),
// so
//   f'(a) = (-3 f(a) + 4 f(a + h) - f(a + 2h)) / (2h) + h^2/3 f'''(a) + O(h^3).
// The approximation is second order, with error  -h^2/3 f'''(a) + O(h^3).
// It is exact for quadratics. (With -h instead of h it is the backward
// version, used at the right end of a grid.)
template <std::invocable<double> F>
[[nodiscard]] double Ex03OneSided(F&& f, double a, double h) {
  return (-3.0 * f(a) + 4.0 * f(a + h) - f(a + 2.0 * h)) / (2.0 * h);
}
// Leading error term, approximation - f'(a) ~ -h^2 f'''(a) / 3.
[[nodiscard]] constexpr double Ex03LeadingError(double f3, double h) noexcept {
  return -h * h * f3 / 3.0;
}

// ------------------------------------------------------------ Exercise 4 ---
// Let f be infinitely differentiable. Find a central finite difference
// approximation for f^(4)(a) using f(a - 2h), f(a - h), f(a), f(a + h),
// f(a + 2h). What is its order?
//
// With symmetric points the odd derivatives cancel in the symmetric sums
//   f(a + h) + f(a - h)   = 2f + h^2 f'' + h^4/12 f^(4) + 2h^6/720 f^(6) +
//   O(h^8), f(a + 2h) + f(a - 2h) = 2f + 4h^2 f'' + 16h^4/12 f^(4) + 128h^6/720
//   f^(6) + O(h^8).
// Eliminate f'' (second sum minus 4 times the first):
//   f(a+2h) - 4f(a+h) + 6f(a) - 4f(a-h) + f(a-2h) = h^4 f^(4) + h^6/6 f^(6) +
//   O(h^8),
// so
//   f^(4)(a) = (f(a-2h) - 4f(a-h) + 6f(a) - 4f(a+h) + f(a+2h)) / h^4 - h^2/6
//   f^(6)(a) + O(h^4).
// The weights (1, -4, 6, -4, 1) are those of the fourth difference (binomial
// coefficients). The approximation is second order, with error
//   h^2/6 f^(6)(a) + O(h^4).
// It is exact for polynomials of degree 5. Round-off grows like eps |f| / h^4,
// so in double precision h should not be too small (h ~ eps^(1/6) ~ 2e-3).
template <std::invocable<double> F>
[[nodiscard]] double Ex04FourthDerivative(F&& f, double a, double h) {
  return (f(a - 2.0 * h) - 4.0 * f(a - h) + 6.0 * f(a) - 4.0 * f(a + h) +
          f(a + 2.0 * h)) /
         (h * h * h * h);
}
[[nodiscard]] constexpr double Ex04LeadingError(double f6, double h) noexcept {
  return h * h * f6 / 6.0;
}

}  // namespace p4mfe::ch07::ex
