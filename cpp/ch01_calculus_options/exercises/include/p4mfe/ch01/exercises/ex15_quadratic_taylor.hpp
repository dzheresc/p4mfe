#pragma once
// Exercise 15 (section 1.12): gradient, Hessian and the exact quadratic
// Taylor expansion of a quadratic form.
//
// Let f(x) = 2 x1^2 - x1 x2 + 3 x2 x3 - x3^2,  x = (x1, x2, x3).
// (i)  Compute Df(1, -1, 0) and D^2 f(1, -1, 0).
// (ii) Show that  f(x) = f(a) + Df(a)(x - a) + 1/2 (x - a)^t D^2 f(a) (x - a)
//      for a = (1, -1, 0)^t.
//
// Solution.
//   Df(x) = (4 x1 - x2,  -x1 + 3 x3,  3 x2 - 2 x3),  so Df(1,-1,0) = (5, -1,
//   -3); D^2 f = [[ 4, -1,  0],
//            [-1,  0,  3],
//            [ 0,  3, -2]]   (constant);   f(a) = 2 + 1 + 0 - 0 = 3.
// (ii) With x = a + d, d = (x1 - 1, x2 + 1, x3), expand:
//   f(a + d) = 2(1 + d1)^2 - (1 + d1)(d2 - 1) + 3(d2 - 1) d3 - d3^2
//            = 3 + (5 d1 - d2 - 3 d3) + (2 d1^2 - d1 d2 + 3 d2 d3 - d3^2),
// and the quadratic part equals 1/2 d^t D^2 f d (the off-diagonal entries are
// counted twice). A quadratic polynomial equals its second-order Taylor
// expansion exactly.

#include <array>
#include <cstddef>

namespace p4mfe::ch01::ex {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

[[nodiscard]] constexpr double Ex15F(const Vec3& x) noexcept {
  return 2.0 * x[0] * x[0] - x[0] * x[1] + 3.0 * x[1] * x[2] - x[2] * x[2];
}

[[nodiscard]] constexpr Vec3 Ex15Gradient(const Vec3& x) noexcept {
  return {4.0 * x[0] - x[1], -x[0] + 3.0 * x[2], 3.0 * x[1] - 2.0 * x[2]};
}

[[nodiscard]] constexpr Mat3 Ex15Hessian() noexcept {
  return {{{4.0, -1.0, 0.0}, {-1.0, 0.0, 3.0}, {0.0, 3.0, -2.0}}};
}

inline constexpr Vec3 kEx15A{1.0, -1.0, 0.0};

// Right-hand side of (1.66) around the point a.
[[nodiscard]] constexpr double Ex15Taylor2(const Vec3& x,
                                           const Vec3& a = kEx15A) noexcept {
  const Vec3 g = Ex15Gradient(a);
  const Mat3 hessian = Ex15Hessian();
  const Vec3 d{x[0] - a[0], x[1] - a[1], x[2] - a[2]};
  double lin = 0.0, quad = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    lin += g[i] * d[i];
    for (std::size_t j = 0; j < 3; ++j) quad += d[i] * hessian[i][j] * d[j];
  }
  return Ex15F(a) + lin + 0.5 * quad;
}

}  // namespace p4mfe::ch01::ex
