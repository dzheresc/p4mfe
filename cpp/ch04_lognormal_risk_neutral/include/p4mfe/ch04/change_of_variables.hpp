#pragma once
// Chapter 4, section 4.1: the density of a function of a random variable.
//
// If X has density f_X and Y = g(X) with g strictly monotone and
// differentiable, then with h = g^(-1),
//   f_Y(y) = f_X(h(y)) |h'(y)|
// for y in the range of g, and 0 elsewhere. The argument: for increasing g,
//   F_Y(y) = P(g(X) <= y) = P(X <= h(y)) = F_X(h(y)),
// and differentiating gives f_X(h(y)) h'(y); for decreasing g the inequality
// flips and the sign of h' with it.
//
// The functions here return the density of Y as a callable, so it can be fed
// straight into ch03::ContinuousDistribution to compute moments.

#include <cmath>
#include <concepts>
#include <p4mfe/ch01/calculus.hpp>
#include <stdexcept>

namespace p4mfe::ch04 {

// f_Y(y) = f_X(h(y)) |h'(y)|, with h = g^(-1) and its derivative dh given.
// in_range(y) says whether y is in the range of g (outside, f_Y = 0); by
// default every y is.
template <std::invocable<double> FX, std::invocable<double> H,
          std::invocable<double> DH>
[[nodiscard]] auto transformed_pdf(FX f_X, H h, DH dh) {
  return [=](double y) { return f_X(h(y)) * std::abs(dh(y)); };
}
template <std::invocable<double> FX, std::invocable<double> H,
          std::invocable<double> DH, std::predicate<double> InRange>
[[nodiscard]] auto transformed_pdf(FX f_X, H h, DH dh, InRange in_range) {
  return
      [=](double y) { return in_range(y) ? f_X(h(y)) * std::abs(dh(y)) : 0.0; };
}

// Same, with h' taken by a central finite difference (chapter 1), for when
// only the inverse function is at hand.
template <std::invocable<double> FX, std::invocable<double> H,
          std::predicate<double> InRange>
[[nodiscard]] auto transformed_pdf_numeric(FX f_X, H h, InRange in_range) {
  return [=](double y) {
    return in_range(y) ? f_X(h(y)) * std::abs(ch01::derivative(h, y)) : 0.0;
  };
}

// Y = a X + b, a != 0: f_Y(y) = f_X((y - b) / a) / |a|. With X standard
// normal this gives the N(b, a^2) density.
template <std::invocable<double> FX>
[[nodiscard]] auto linear_transform_pdf(FX f_X, double a, double b) {
  if (a == 0.0)
    throw std::invalid_argument("linear_transform_pdf: a must be nonzero");
  return [=](double y) { return f_X((y - b) / a) / std::abs(a); };
}

// Y = e^X: h(y) = ln y, h'(y) = 1/y, so f_Y(y) = f_X(ln y) / y for y > 0.
// With X normal this is the lognormal density of section 4.2.
template <std::invocable<double> FX>
[[nodiscard]] auto exp_transform_pdf(FX f_X) {
  return [=](double y) { return y > 0.0 ? f_X(std::log(y)) / y : 0.0; };
}

// A non-monotone example: Y = X^2. Both x = sqrt(y) and x = -sqrt(y) map to
// y, so the formula above is applied on each branch and the two added:
//   f_Y(y) = (f_X(sqrt(y)) + f_X(-sqrt(y))) / (2 sqrt(y)),  y > 0.
// With X standard normal this is the chi-squared density with one degree of
// freedom, e^(-y/2) / sqrt(2 pi y).
template <std::invocable<double> FX>
[[nodiscard]] auto square_transform_pdf(FX f_X) {
  return [=](double y) {
    if (y <= 0.0) return 0.0;
    const double s = std::sqrt(y);
    return (f_X(s) + f_X(-s)) / (2.0 * s);
  };
}

}  // namespace p4mfe::ch04
