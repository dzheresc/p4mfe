// Sections 9.1-9.1.1: constrained extrema by Lagrange multipliers, solved by
// Newton's method on the Lagrange system and classified on the tangent space.

#include <cstdio>
#include <p4mfe/ch09/lagrange.hpp>

namespace {
const char* name(p4mfe::ch09::CriticalPointType t) {
  using T = p4mfe::ch09::CriticalPointType;
  return t == T::kMinimum   ? "minimum"
         : t == T::kMaximum ? "maximum"
         : t == T::kSaddle  ? "saddle"
                            : "degenerate";
}
}  // namespace

int main() {
  using namespace p4mfe::ch09;

  const auto f1 = [](const Vector& v) { return v[0] + 2 * v[1]; };
  const auto g1 = [](const Vector& v) {
    return Vector{v[0] * v[0] + v[1] * v[1] - 5};
  };
  std::printf("x + 2y on x^2 + y^2 = 5\n");
  for (const Vector& x0 : {Vector{1, 1}, Vector{-1, -1}}) {
    const auto r = lagrange_stationary_point(f1, g1, x0);
    std::printf(
        "  from (%5.2f, %5.2f): x = (%.8f, %.8f), lambda = %.8f, f = %.8f, %s "
        "(%zu Newton steps)\n",
        x0[0], x0[1], r.x[0], r.x[1], r.lambda[0], f1(r.x),
        name(classify_constrained(f1, g1, r.x, r.lambda)), r.iterations);
  }

  const auto f2 = [](const Vector& v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  };
  const auto g2 = [](const Vector& v) {
    return Vector{v[0] + v[1] + v[2] - 3, v[0] - v[1] - 1};
  };
  const auto r2 = lagrange_stationary_point(f2, g2, {0, 0, 0});
  std::printf(
      "\nx^2 + y^2 + z^2 on x + y + z = 3, x - y = 1\n  x = (%.8f, %.8f, "
      "%.8f), lambda = (%.6f, %.6f), %s\n",
      r2.x[0], r2.x[1], r2.x[2], r2.lambda[0], r2.lambda[1],
      name(classify_constrained(f2, g2, r2.x, r2.lambda)));

  const auto f3 = [](const Vector& v) { return v[0] * v[1] * v[2]; };
  const auto g3 = [](const Vector& v) {
    return Vector{2 * (v[0] * v[1] + v[1] * v[2] + v[2] * v[0]) - 6};
  };
  const auto r3 = lagrange_stationary_point(f3, g3, {0.8, 1.2, 1.1});
  std::printf(
      "\nBox volume xyz with surface 6\n  x = (%.8f, %.8f, %.8f), volume %.8f, "
      "%s\n",
      r3.x[0], r3.x[1], r3.x[2], f3(r3.x),
      name(classify_constrained(f3, g3, r3.x, r3.lambda)));
}
