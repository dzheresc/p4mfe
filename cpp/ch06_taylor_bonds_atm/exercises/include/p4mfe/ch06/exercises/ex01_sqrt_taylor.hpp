#pragma once
// Exercise 1 (section 6.11): the cubic Taylor approximation of sqrt(1 + x).

#include <array>
#include <cmath>
#include <p4mfe/ch06/taylor.hpp>
#include <p4mfe/ch06/taylor_series.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Show that the cubic Taylor approximation of sqrt(1 + x) around 0 is
//   sqrt(1 + x) ~ 1 + x/2 - x^2/8 + x^3/16.
//
// f(x) = (1 + x)^(1/2) has derivatives
//   f'(x)   =  (1/2) (1 + x)^(-1/2),   f'(0)   =  1/2,
//   f''(x)  = -(1/4) (1 + x)^(-3/2),   f''(0)  = -1/4,
//   f'''(x) =  (3/8) (1 + x)^(-5/2),   f'''(0) =  3/8,
// so P_3(x) = f(0) + f'(0) x + f''(0) x^2/2 + f'''(0) x^3/6
//           = 1 + x/2 - x^2/8 + x^3/16.
// This is also the binomial series (6.3.1) with p = 1/2 cut after k = 3.
// The next derivative, f''''(x) = -(15/16)(1 + x)^(-7/2), gives the error
//   sqrt(1 + x) - P_3(x) = -(5/128) x^4 + O(x^5).
inline constexpr std::array<double, 4> ex01_derivatives_at_zero{1.0, 0.5, -0.25,
                                                                0.375};
inline constexpr std::array<double, 4> ex01_coefficients{1.0, 0.5, -0.125,
                                                         0.0625};
inline constexpr double ex01_error_coefficient = -5.0 / 128.0;

[[nodiscard]] constexpr double ex01_cubic(double x) noexcept {
  return ex01_coefficients[0] +
         x * (ex01_coefficients[1] +
              x * (ex01_coefficients[2] + x * ex01_coefficients[3]));
}
// The same polynomial built from the derivatives (chapter 6's
// taylor_polynomial).
[[nodiscard]] inline double ex01_cubic_from_derivatives(double x) {
  return taylor_polynomial(ex01_derivatives_at_zero, 0.0, x);
}

}  // namespace p4mfe::ch06::ex
