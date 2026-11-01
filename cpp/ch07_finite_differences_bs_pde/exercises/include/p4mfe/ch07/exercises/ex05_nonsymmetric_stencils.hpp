#pragma once
// Exercise 5 (section 7.6): what symmetry buys in finite differences.

#include <concepts>

namespace p4mfe::ch07::ex {

// ------------------------------------------------------------ Exercise 5 ---
// The central differences (7.7) and (7.9) approximate f'(a) and f''(a) to
// second order using a and the symmetric points a - h, a + h. What happens
// without symmetry?
//
// (i)  A second order approximation of f'(a) from f(a), f(a - h), f(a + 2h).
// (ii) A first order approximation of f''(a) from the same values; in general
//      a second order one does not exist.
// Let beta < a < gamma with a - beta = C (gamma - a), C a constant.
// (iii) An approximation of f'(a) from f(a), f(beta), f(gamma), with residual
//       O(|gamma - a|^2).
// (iv)  An approximation of f''(a) from the same values, first order in
//       |gamma - a|; second order is not possible unless a = (beta + gamma)/2.
//
// General case first. Write k = gamma - a > 0, so a - beta = C k. Then
//   f(gamma) = f + k f' + k^2/2 f'' + k^3/6 f''' + O(k^4),
//   f(beta)  = f - Ck f' + C^2k^2/2 f'' - C^3k^3/6 f''' + O(k^4).
//
// (iii) Eliminate f'' with weights C^2 and -1:
//   C^2 f(gamma) - f(beta) = (C^2 - 1) f + C(C + 1) k f' + C^2(C + 1) k^3/6 f''' + O(k^4),
// so
//   f'(a) ~ [C^2 f(gamma) - f(beta) - (C^2 - 1) f(a)] / (C (C + 1) k),
// with error (approximation - f'(a)) = C k^2/6 f'''(a) + O(k^3) = O(|gamma - a|^2).
// In terms of the points (multiply through by k^2; C k = a - beta):
//   f'(a) ~ [(a-beta)^2 f(gamma) - (gamma-a)^2 f(beta) - ((a-beta)^2 - (gamma-a)^2) f(a)]
//           / ((a - beta)(gamma - a)(gamma - beta)),
// error (a - beta)(gamma - a)/6 f'''(a) + higher order terms.
// For C = 1 this is the central difference (f(gamma) - f(beta)) / (2k).
//
// (iv) Eliminate f' with weights C and 1:
//   C f(gamma) + f(beta) = (C + 1) f + C(C + 1) k^2/2 f'' + C(1 - C^2) k^3/6 f''' + O(k^4),
// so
//   f''(a) ~ 2 [C f(gamma) + f(beta) - (C + 1) f(a)] / (C (C + 1) k^2)
//          = 2 [(a-beta) f(gamma) + (gamma-a) f(beta) - (gamma-beta) f(a)]
//            / ((a - beta)(gamma - a)(gamma - beta)),
// with error (1 - C) k/3 f'''(a) + O(k^2) = (beta + gamma - 2a)/3 f'''(a) + O(k^2).
// This is first order in k. A second order formula would need weights
// matching f, f', f'' (three linear equations, three unknowns: the weights
// above are the only solution) and also killing f''': only when
// C (1 - C^2) = 0, i.e. C = 1, a = (beta + gamma)/2. Otherwise the error for
// f(x) = (x - a)^3 is exactly (beta + gamma - 2a), never O(k^2).
//
// (i), (ii) are the case beta = a - h, gamma = a + 2h (C = 1/2, k = 2h):
//   f'(a)  ~ (3 f(a) - 4 f(a - h) + f(a + 2h)) / (6h),     error  h^2/3 f'''(a),
//   f''(a) ~ (2 f(a - h) - 3 f(a) + f(a + 2h)) / (3h^2),   error  h/3 f'''(a).
// (ii) is only first order: a - h, a + 2h are not symmetric about a.
template <std::invocable<double> F>
[[nodiscard]] double ex05_first_derivative(F&& f, double a, double beta, double gamma) {
    const double u = a - beta, v = gamma - a;
    return (u * u * f(gamma) - v * v * f(beta) - (u * u - v * v) * f(a)) / (u * v * (gamma - beta));
}
template <std::invocable<double> F>
[[nodiscard]] double ex05_second_derivative(F&& f, double a, double beta, double gamma) {
    const double u = a - beta, v = gamma - a;
    return 2.0 * (u * f(gamma) + v * f(beta) - (gamma - beta) * f(a)) / (u * v * (gamma - beta));
}
// Leading error terms (approximation minus exact) in terms of f'''(a).
[[nodiscard]] constexpr double ex05_first_leading_error(double a, double beta, double gamma, double f3) noexcept {
    return (a - beta) * (gamma - a) / 6.0 * f3;
}
[[nodiscard]] constexpr double ex05_second_leading_error(double a, double beta, double gamma, double f3) noexcept {
    return (beta + gamma - 2.0 * a) / 3.0 * f3;
}

// (i) and (ii): the points a - h, a, a + 2h.
template <std::invocable<double> F>
[[nodiscard]] double ex05_first_derivative_h(F&& f, double a, double h) {
    return (3.0 * f(a) - 4.0 * f(a - h) + f(a + 2.0 * h)) / (6.0 * h);
}
template <std::invocable<double> F>
[[nodiscard]] double ex05_second_derivative_h(F&& f, double a, double h) {
    return (2.0 * f(a - h) - 3.0 * f(a) + f(a + 2.0 * h)) / (3.0 * h * h);
}

}  // namespace p4mfe::ch07::ex
