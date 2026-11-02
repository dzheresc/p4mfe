#pragma once
// Exercises 10-11 (section 3.11): the Cauchy-Schwarz inequality for sums and
// for integrals, proved with the quadratic-in-alpha trick of Lemma 3.8.

#include <cmath>
#include <cstddef>
#include <p4mfe/ch02/integration.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch03::ex {

// ----------------------------------------------------------- Exercise 10 ---
// Prove that
//   (sum x_i y_i)^2 <= (sum x_i^2)(sum y_i^2),  for all x_i, y_i real, i = 1:n.
//
// For every real alpha,
//   0 <= sum (x_i + alpha y_i)^2 = A + 2 B alpha + C alpha^2,
// with A = sum x_i^2, B = sum x_i y_i, C = sum y_i^2. If C = 0 then all y_i
// are 0 and both sides are 0. Otherwise this quadratic in alpha is never
// negative, so it has at most one real root and its discriminant is <= 0:
//   4 B^2 - 4 A C <= 0,   i.e.   B^2 <= A C.
// Equality holds iff x + alpha y = 0 for some alpha, i.e. x and y are
// proportional.
struct CauchySchwarzSides {
  double lhs;  // ex10: B^2;  ex11: B
  double rhs;  // ex10: A C;  ex11: sqrt(A) sqrt(C)
};

[[nodiscard]] inline CauchySchwarzSides Ex10Sides(std::span<const double> x,
                                                  std::span<const double> y) {
  if (x.size() != y.size())
    throw std::invalid_argument("ex10: x and y must have the same length");
  double sum_xx = 0.0, sum_xy = 0.0, sum_yy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sum_xx += x[i] * x[i];
    sum_xy += x[i] * y[i];
    sum_yy += y[i] * y[i];
  }
  return {sum_xy * sum_xy, sum_xx * sum_yy};
}

// The quadratic sum (x_i + alpha y_i)^2, nonnegative for every alpha; its
// minimum, at alpha* = -B/C, is A - B^2/C = (AC - B^2)/C.
[[nodiscard]] inline double Ex10Quadratic(std::span<const double> x,
                                          std::span<const double> y,
                                          double alpha) {
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i)
    s += (x[i] + alpha * y[i]) * (x[i] + alpha * y[i]);
  return s;
}

// ----------------------------------------------------------- Exercise 11 ---
// Use the technique from the proof of Lemma 3.8 to show that
//   int_a^b f g dx <= (int_a^b f^2 dx)^(1/2) (int_a^b g^2 dx)^(1/2)
// for any two continuous functions f, g. Hint: int_a^b (f + alpha g)^2 dx >= 0.
//
// Same argument with sums replaced by integrals: for every alpha,
//   0 <= int (f + alpha g)^2 = A + 2 B alpha + C alpha^2,
// with A = int f^2, B = int f g, C = int g^2. If C = 0, g = 0 on [a, b]
// (g is continuous) and both sides vanish. Otherwise the discriminant is
// <= 0: B^2 <= A C, so |int f g| <= sqrt(A) sqrt(C), which is stronger than
// the inequality asked for.
// The sides are computed with Simpson's rule (n intervals).
template <class F, class G>
[[nodiscard]] CauchySchwarzSides Ex11Sides(F&& f, G&& g, double a, double b,
                                           std::size_t n = 1000) {
  const double int_ff =
      ch02::Integrate([&](double x) { return f(x) * f(x); }, a, b, n);
  const double int_fg =
      ch02::Integrate([&](double x) { return f(x) * g(x); }, a, b, n);
  const double int_gg =
      ch02::Integrate([&](double x) { return g(x) * g(x); }, a, b, n);
  return {int_fg, std::sqrt(int_ff) * std::sqrt(int_gg)};
}

}  // namespace p4mfe::ch03::ex
