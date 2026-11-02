#pragma once
// Exercise 2 (section 9.6): the arithmetic-geometric-harmonic means
// inequality, proved with Lagrange multipliers.

#include <cmath>
#include <cstddef>
#include <p4mfe/ch09/lagrange.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch09::ex {

// ------------------------------------------------------------ Exercise 2 ---
// Prove that for x_i > 0, i = 1:n,
//   (1/n) sum x_i  >=  (prod x_i)^(1/n)  >=  n / sum (1/x_i).          (9.69)
// The three terms are the arithmetic, geometric and harmonic means.
//
// Proof. For n = 1 all three terms equal x_1. Let n >= 2.
//
// (a) A >= G. Fix S > 0 and consider the problem
//   maximize P(x) = x_1 x_2 ... x_n   subject to   g(x) = x_1 + ... + x_n - S =
//   0
// on the closed simplex K = {x : x_i >= 0, sum x_i = S}. K is compact and P
// is continuous, so P attains its maximum on K. On the boundary of K some
// x_i = 0 and P = 0, while P(S/n, ..., S/n) = (S/n)^n > 0, so the maximum is
// attained at a point with all x_i > 0. There the only constraint is g = 0,
// whose gradient (1, ..., 1) is never zero, so the point solves the Lagrange
// system for L = P + lambda g:
//   dP/dx_i + lambda = prod_{j != i} x_j + lambda = P(x)/x_i + lambda = 0,
// i.e. x_i = -P(x)/lambda for every i. All x_i are equal, and sum x_i = S
// gives x_i = S/n. This is the only stationary point with x > 0, so it is the
// unique maximizer:
//   prod x_i <= (S/n)^n  for all x_i >= 0 with sum x_i = S,
// with equality only when all x_i = S/n. Given any x_i > 0, take
// S = sum x_i and take the n-th root:
//   (prod x_i)^(1/n) <= (1/n) sum x_i,   equality iff x_1 = ... = x_n.
//
// (b) G >= H. Apply (a) to y_i = 1/x_i > 0:
//   (1/n) sum 1/x_i >= (prod 1/x_i)^(1/n) = 1 / (prod x_i)^(1/n),
// and rearranging (both sides are positive)
//   (prod x_i)^(1/n) >= n / sum (1/x_i),   equality iff x_1 = ... = x_n.
// Together (a) and (b) are (9.69). QED
//
// The code checks the inequality on many vectors, and repeats the Lagrange
// argument numerically: maximizing sum ln x_i (= ln P, same maximizer, and
// smooth where P is positive) subject to sum x_i = n a from an unequal start
// converges to x = (a, ..., a) with lambda = -1/a, and the second-order test
// on the tangent space (H_L = -diag(1/x_i^2), negative definite) says maximum.

[[nodiscard]] inline double arithmetic_mean(const std::vector<double>& x) {
  if (x.empty()) throw std::invalid_argument("arithmetic_mean: empty input");
  double s = 0.0;
  for (double v : x) s += v;
  return s / static_cast<double>(x.size());
}

// Computed as exp(mean of ln x_i) so the product cannot overflow.
[[nodiscard]] inline double geometric_mean(const std::vector<double>& x) {
  if (x.empty()) throw std::invalid_argument("geometric_mean: empty input");
  double s = 0.0;
  for (double v : x) {
    if (!(v > 0.0)) throw std::invalid_argument("geometric_mean: need x_i > 0");
    s += std::log(v);
  }
  return std::exp(s / static_cast<double>(x.size()));
}

[[nodiscard]] inline double harmonic_mean(const std::vector<double>& x) {
  if (x.empty()) throw std::invalid_argument("harmonic_mean: empty input");
  double s = 0.0;
  for (double v : x) {
    if (!(v > 0.0)) throw std::invalid_argument("harmonic_mean: need x_i > 0");
    s += 1.0 / v;
  }
  return static_cast<double>(x.size()) / s;
}

// The Lagrange problem of the proof: maximize sum ln x_i subject to
// sum x_i = n a, solved by Newton's method from x0 (all x0_i > 0).
[[nodiscard]] inline LagrangeResult ex02_lagrange_max(const Vector& x0,
                                                      double a) {
  const double target = static_cast<double>(x0.size()) * a;
  const auto f = [](const Vector& x) {
    double s = 0.0;
    for (double v : x) s += std::log(v);
    return s;
  };
  const auto g = [target](const Vector& x) {
    double s = -target;
    for (double v : x) s += v;
    return Vector{s};
  };
  return lagrange_stationary_point(f, g, x0);
}

[[nodiscard]] inline CriticalPointType ex02_classify(const LagrangeResult& r,
                                                     double a) {
  const double target = static_cast<double>(r.x.size()) * a;
  const auto f = [](const Vector& x) {
    double s = 0.0;
    for (double v : x) s += std::log(v);
    return s;
  };
  const auto g = [target](const Vector& x) {
    double s = -target;
    for (double v : x) s += v;
    return Vector{s};
  };
  return classify_constrained(f, g, r.x, r.lambda);
}

}  // namespace p4mfe::ch09::ex
