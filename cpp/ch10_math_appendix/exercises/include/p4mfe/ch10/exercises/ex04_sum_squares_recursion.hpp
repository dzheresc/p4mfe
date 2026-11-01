#pragma once
// Exercise 4 (section 10.10): S(n, 2) = n(n+1)(2n+1)/6 from a linear recursion
// with characteristic polynomial (z - 1)^4.

#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>

#include <array>
#include <vector>

namespace p4mfe::ch10::ex {

// ------------------------------------------------------------ Exercise 4 ---
// x_n = sum_{k=1}^{n} k^2 satisfies x_{n+1} = x_n + (n+1)^2 (10.91), x_1 = 1.
//
// (i) Substituting n + 1 for n: x_{n+2} = x_{n+1} + (n+2)^2 (10.92).
// Subtracting (10.91): x_{n+2} - x_{n+1} = x_{n+1} - x_n + (n+2)^2 - (n+1)^2, i.e.
//   x_{n+2} = 2 x_{n+1} - x_n + 2n + 3  (10.93),  x_1 = 1, x_2 = 5.
//
// (ii) Substituting n + 1 in (10.93): x_{n+3} = 2 x_{n+2} - x_{n+1} + 2(n+1) + 3
// (10.94). Subtracting (10.93):
//   x_{n+3} - x_{n+2} = 2 x_{n+2} - 3 x_{n+1} + x_n + 2, i.e.
//   x_{n+3} = 3 x_{n+2} - 3 x_{n+1} + x_n + 2,  x_1 = 1, x_2 = 5, x_3 = 14.
//
// (iii) Once more: substituting n + 1 gives
// x_{n+4} = 3 x_{n+3} - 3 x_{n+2} + x_{n+1} + 2, and subtracting the previous
// equation removes the constant 2:
//   x_{n+4} - x_{n+3} = 3 x_{n+3} - 6 x_{n+2} + 4 x_{n+1} - x_n, i.e.
//   x_{n+4} - 4 x_{n+3} + 6 x_{n+2} - 4 x_{n+1} + x_n = 0  (10.95).
// Its characteristic polynomial is z^4 - 4z^3 + 6z^2 - 4z + 1 = (z - 1)^4: the
// single root 1 with multiplicity 4, so the general solution is
//   x_n = (c_0 + c_1 n + c_2 n^2 + c_3 n^3) 1^n,
// a cubic polynomial in n. The four conditions x_1 = 1, x_2 = 5, x_3 = 14,
// x_4 = 30 fix the cubic (it interpolates four points), and
// p(n) = n(n+1)(2n+1)/6 takes the values 1, 5, 14, 30 at n = 1, 2, 3, 4. By
// uniqueness of the interpolating polynomial, c = (0, 1/6, 1/2, 1/3) and
//   x_n = n(n+1)(2n+1)/6 for all n >= 1.
// Since x_n = S(n, 2) by definition, S(n, 2) = n(n+1)(2n+1)/6.
//
// Numerically: the recursion (10.95) is run with chapter 10's recursion_terms,
// and the cubic is found with chapter 10's interpolating_polynomial.

// Coefficients a_0..a_3 of x_{n+4} = a_0 x_n + a_1 x_{n+1} + a_2 x_{n+2} + a_3 x_{n+3}.
inline constexpr std::array<double, 4> ex04_recursion{-1.0, 4.0, -6.0, 4.0};

// x_1, ..., x_count from (10.95) and x_1..x_4 = 1, 5, 14, 30 (index 0 holds x_1).
[[nodiscard]] inline std::vector<double> ex04_terms(std::size_t count) {
    const std::array<double, 4> init{1.0, 5.0, 14.0, 30.0};
    return recursion_terms(ex04_recursion, init, count);
}

// x_1, ..., x_count from the second order form (10.93) (affine, so iterated directly).
[[nodiscard]] inline std::vector<double> ex04_terms_10_93(std::size_t count) {
    std::vector<double> x{1.0, 5.0};
    for (std::size_t n = 1; x.size() < count; ++n) x.push_back(2.0 * x[n] - x[n - 1] + 2.0 * static_cast<double>(n) + 3.0);
    x.resize(count);
    return x;
}

// c_0..c_3 of x_n = sum c_m n^m, the cubic through (n, x_n), n = 1..4.
[[nodiscard]] inline Vector ex04_cubic() {
    const std::array<double, 4> ns{1.0, 2.0, 3.0, 4.0}, xs{1.0, 5.0, 14.0, 30.0};
    return interpolating_polynomial(ns, xs);
}

}  // namespace p4mfe::ch10::ex
