#pragma once
// Exercise 6 (section 5.7): the dividend yield that makes the Delta of an
// at-the-money call equal to 0.5.

#include <cmath>
#include <p4mfe/ch05/solvers.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 6 ---
// A six months at-the-money call on an asset with volatility 30% paying
// dividends continuously at rate q; interest rates are constant at 4%. Show
// that there is a unique positive q with Delta(C) = 0.5, find it with
// Newton's method, and compare it with r + sigma^2/2.
//
// At the money (K = S): Delta(q) = e^(-qT) N(d1(q)), d1(q) = (r - q +
// sigma^2/2) sqrt(T)/sigma. Solve g(q) = e^(-qT) N(d1(q)) - 1/2 = 0, with
//   g'(q) = -T e^(-qT) N(d1) - e^(-qT) phi(d1) sqrt(T)/sigma < 0.
// Uniqueness and existence: g is strictly decreasing;
//   g(0) = N((r + sigma^2/2) sqrt(T)/sigma) - 1/2 > 0 (d1 > 0), and
//   g(r + sigma^2/2) = e^(-qT) N(0) - 1/2 = (e^(-qT) - 1)/2 < 0,
// so g has exactly one root q*, and 0 < q* < r + sigma^2/2.
// Newton's method from q0 = 0 (iterates 0, 0.065544, 0.066905, ...,
// increasing monotonically) gives, in 4 steps,
//   q* = 0.066906 (6.6906%),
// below r + sigma^2/2 = 0.085: at q = r + sigma^2/2 the N(d1) factor is
// exactly 1/2, and the extra discount e^(-qT) < 1 pushes Delta below 1/2, so
// the threshold must be lower. (Compare chapter 4, exercise 10, and the
// approximation q* ~ (r + sigma^2/2)/(1 + sigma sqrt(pi T/2)) = 0.06715.)
struct Ex06Market {
  double maturity = 0.5, sigma = 0.3, r = 0.04;
};

[[nodiscard]] inline double Ex06D1(double q, const Ex06Market& m = {}) {
  return (m.r - q + 0.5 * m.sigma * m.sigma) * std::sqrt(m.maturity) / m.sigma;
}
[[nodiscard]] inline double Ex06Delta(double q, const Ex06Market& m = {}) {
  return std::exp(-q * m.maturity) * norm_cdf(Ex06D1(q, m));
}
[[nodiscard]] inline double Ex06DeltaDq(double q, const Ex06Market& m = {}) {
  const double d1 = Ex06D1(q, m);
  return -std::exp(-q * m.maturity) *
         (m.maturity * norm_cdf(d1) +
          norm_pdf(d1) * std::sqrt(m.maturity) / m.sigma);
}
[[nodiscard]] inline RootResult Ex06Solve(double q0 = 0.0,
                                          const Ex06Market& m = {}) {
  return Newton([&](double q) { return Ex06Delta(q, m) - 0.5; },
                [&](double q) { return Ex06DeltaDq(q, m); }, q0,
                {1e-12, 1e-12, 100});
}
[[nodiscard]] inline double Ex06UpperBound(const Ex06Market& m = {}) {
  return m.r + 0.5 * m.sigma * m.sigma;
}

}  // namespace p4mfe::ch05::ex
