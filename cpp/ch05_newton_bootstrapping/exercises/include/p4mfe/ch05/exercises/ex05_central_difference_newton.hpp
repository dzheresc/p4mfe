#pragma once
// Exercise 5 (section 5.7): the approximate Newton's method with a central
// difference Jacobian, compared with forward differences (section 5.2.2).

#include <p4mfe/ch05/newton_nd.hpp>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 5 ---
// F: R^3 -> R^3,
//   F1(x) = x1^3 + 2 x1 x2 + x3^2 - x2 x3 + 9
//   F2(x) = 2 x1^2 + 2 x1 x2^2 + x2^3 x3^2 - x2^2 x3 - 2
//   F3(x) = x1 x2 x3 + x1^3 - x3^2 - x1 x2^2 - 4.
// (i) Solve F(x) = 0 with the approximate Newton's method of Table 5.4,
//     replacing D_F(x_old) by the central difference Jacobian
//       (Delta_c F)_ij = (F_i(x + h e_j) - F_i(x - h e_j)) / (2h),
//     h = 1e-6, tol_consec = 1e-6, tol_approx = 1e-9, from x0 = (1, 2, 3) and
//     x0 = (2, 2, 2).
// (ii) Compare with the forward difference version of section 5.2.2.
//
// This is `approximate_newton_nd` with Difference::Central. Results:
//   x0 = (1, 2, 3): x = (-1.690551, 1.983107, -0.884558) in 9 iterations,
//                   with central differences, forward differences and the
//                   exact Jacobian alike;
//   x0 = (2, 2, 2): a long chaotic excursion. The first step already jumps
//                   to (-0.45, -5.35, 16.47) and the iterates wander as far
//                   as |x| ~ 400 before a root catches them. Usually that is
//                   the exact root (-1, 3, 1) (substitute it): on one g++
//                   Release build central differences take 41 steps,
//                   forward 166, the exact Jacobian 40; with NumPy 127, 81
//                   and 40. On the GitHub CI runner, though, one of the
//                   difference schemes lands on (-1.690551, 1.983107,
//                   -0.884558) instead. Neither the step count nor the root
//                   reached from (2, 2, 2) is reproducible across machines.
// (ii) Close to a root the two schemes behave the same: the Jacobian error
// (O(h) forward, O(h^2) central, plus rounding of order eps/h) is far too
// small to slow quadratic convergence, so from (1, 2, 3) both take the same
// steps and reach the same root. Far from a root the method is chaotic and
// the tiny Jacobian differences (and even rounding: compare C++ and NumPy)
// send the iterates on different paths, so from (2, 2, 2) the iteration
// count, and even which root is found, depend on the scheme and the machine. Central
// differences cost 2N = 6 evaluations of F per Jacobian against N = 3 for
// forward ones, which buys nothing here. The Jacobian is
// non-singular at both roots (det = 1918.8 and -2332).
[[nodiscard]] inline Vector ex05_F(const Vector& x) {
    const double x1 = x[0], x2 = x[1], x3 = x[2];
    return {x1 * x1 * x1 + 2 * x1 * x2 + x3 * x3 - x2 * x3 + 9,
            2 * x1 * x1 + 2 * x1 * x2 * x2 + x2 * x2 * x2 * x3 * x3 - x2 * x2 * x3 - 2,
            x1 * x2 * x3 + x1 * x1 * x1 - x3 * x3 - x1 * x2 * x2 - 4};
}
// Exact Jacobian, for comparison.
[[nodiscard]] inline Matrix ex05_jacobian(const Vector& x) {
    const double x1 = x[0], x2 = x[1], x3 = x[2];
    return Matrix{{3 * x1 * x1 + 2 * x2, 2 * x1 - x3, 2 * x3 - x2},
                  {4 * x1 + 2 * x2 * x2, 4 * x1 * x2 + 3 * x2 * x2 * x3 * x3 - 2 * x2 * x3,
                   2 * x2 * x2 * x2 * x3 - x2 * x2},
                  {x2 * x3 + 3 * x1 * x1 - x2 * x2, x1 * x3 - 2 * x1 * x2, x1 * x2 - 2 * x3}};
}

inline constexpr Tolerances ex05_tolerances{1e-9, 1e-6, 1000};
inline constexpr double ex05_h = 1e-6;

[[nodiscard]] inline NdRootResult ex05_solve(const Vector& x0, Difference scheme = Difference::Central) {
    return approximate_newton_nd(ex05_F, x0, ex05_tolerances, ex05_h, scheme);
}
[[nodiscard]] inline NdRootResult ex05_solve_exact(const Vector& x0) {
    return newton_nd(ex05_F, ex05_jacobian, x0, ex05_tolerances);
}

}  // namespace p4mfe::ch05::ex
