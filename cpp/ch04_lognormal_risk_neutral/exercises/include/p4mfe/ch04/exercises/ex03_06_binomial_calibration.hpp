#pragma once
// Exercises 3-6 (section 4.9): moments of a one-period binomial step and of
// a lognormal step, and calibrating a binomial tree to the lognormal model
// (the Cox-Ross-Rubinstein tree and the equal-probability tree).

#include <cmath>
#include <stdexcept>

namespace p4mfe::ch04::ex {

// ------------------------------------------------------------ Exercise 3 ---
// One period binomial model between t and t + dt: S(t + dt) = S(t) u with
// risk-neutral probability p, or S(t) d with probability 1 - p (u > 1 > d).
// Show that
//   E_RN[S(t + dt)]   = (p u + (1 - p) d) S(t);              (4.53)
//   E_RN[S^2(t + dt)] = (p u^2 + (1 - p) d^2) S^2(t).        (4.54)
//
// S(t + dt) takes two values, so its expectation is the probability-weighted
// sum: p (S u) + (1 - p)(S d) = (p u + (1 - p) d) S. Likewise S^2(t + dt) is
// S^2 u^2 with probability p and S^2 d^2 with probability 1 - p.
// ex03_moment(k) = p (S u)^k + (1 - p)(S d)^k, the k-th moment by enumeration.
[[nodiscard]] inline double ex03_moment(int k, double S, double u, double d,
                                        double p) {
  return p * std::pow(S * u, k) + (1.0 - p) * std::pow(S * d, k);
}

// ------------------------------------------------------------ Exercise 4 ---
// If S(t) is lognormal with drift mu = r and volatility sigma (no dividends),
//   ln(S(t + dt)/S(t)) = (r - sigma^2/2) dt + sigma sqrt(dt) Z.
// Show that
//   E_RN[S(t + dt)]   = e^(r dt) S(t);                       (4.55)
//   E_RN[S^2(t + dt)] = e^((2r + sigma^2) dt) S^2(t).        (4.56)
//
// For Z ~ N(0, 1), E[e^(aZ)] = e^(a^2/2) (complete the square in the
// integral of e^(az) phi(z)). Then
//   E[S(t+dt)] = S e^((r - sigma^2/2) dt) E[e^(sigma sqrt(dt) Z)]
//              = S e^((r - sigma^2/2) dt + sigma^2 dt / 2) = S e^(r dt),
//   E[S^2(t+dt)] = S^2 e^((2r - sigma^2) dt) E[e^(2 sigma sqrt(dt) Z)]
//                = S^2 e^((2r - sigma^2) dt + 2 sigma^2 dt) = S^2 e^((2r +
//                sigma^2) dt).
[[nodiscard]] inline double ex04_first_moment(double S, double r, double dt) {
  return S * std::exp(r * dt);
}
[[nodiscard]] inline double ex04_second_moment(double S, double r, double sigma,
                                               double dt) {
  return S * S * std::exp((2.0 * r + sigma * sigma) * dt);
}

// ------------------------------------------------------------ Exercise 5 ---
// Calibrate a binomial tree to the lognormal model: find u, d, p with
//   p u + (1 - p) d     = e^(r dt)                           (4.57)
//   p u^2 + (1 - p) d^2 = e^((2r + sigma^2) dt)              (4.58)
// (two equations, three unknowns: the solution is not unique).
//
// (i) Show that (4.57)-(4.58) are equivalent to
//   p = (e^(r dt) - d) / (u - d)                             (4.59)
//   (e^(r dt) - d)(u - e^(r dt)) = e^(2r dt)(e^(sigma^2 dt) - 1).   (4.60)
//
// (4.57) is p (u - d) = e^(r dt) - d, i.e. (4.59). Substitute it in (4.58):
//   p u^2 + (1 - p) d^2 = p (u - d)(u + d) + d^2 = (e^(r dt) - d)(u + d) + d^2
//                       = e^(r dt)(u + d) - u d.
// So (4.58) reads e^(r dt)(u + d) - u d = e^((2r + sigma^2) dt). Subtract
// e^(2r dt) from both sides; the left side factors:
//   e^(r dt)(u + d) - u d - e^(2r dt) = (e^(r dt) - d)(u - e^(r dt)),
// and the right side is e^(2r dt)(e^(sigma^2 dt) - 1): this is (4.60). Every
// step is reversible, so the systems are equivalent.
//
// (ii) Cox-Ross-Rubinstein: add u d = 1. With d = 1/u, the line above
// e^(r dt)(u + d) - u d = e^((2r + sigma^2) dt) becomes
//   e^(r dt)(u + 1/u) = 1 + e^((2r + sigma^2) dt)
//   u + 1/u = e^(-r dt) + e^((r + sigma^2) dt) = 2A,
// so u^2 - 2A u + 1 = 0 and, with u > 1 > d,
//   u = A + sqrt(A^2 - 1),  d = A - sqrt(A^2 - 1) = 1/u,
//   A = (e^(-r dt) + e^((r + sigma^2) dt)) / 2,  p = (e^(r dt) - d)/(u - d).
// A > 1 by AM-GM (the two exponentials multiply to e^(sigma^2 dt) > 1), so
// the root is real. For small dt, u ~ e^(sigma sqrt(dt)), the textbook CRR.
struct BinomialStep {
  double u;
  double d;
  double p;
};
[[nodiscard]] inline BinomialStep ex05_crr(double r, double sigma, double dt) {
  const double A =
      0.5 * (std::exp(-r * dt) + std::exp((r + sigma * sigma) * dt));
  const double root = std::sqrt(A * A - 1.0);
  const double u = A + root, d = A - root;
  return {u, d, (std::exp(r * dt) - d) / (u - d)};
}
// Residuals of (4.57) and (4.58): both zero for a calibrated step.
struct MomentResiduals {
  double first;
  double second;
};
[[nodiscard]] inline MomentResiduals moment_residuals(const BinomialStep& s,
                                                      double r, double sigma,
                                                      double dt) {
  return {s.p * s.u + (1.0 - s.p) * s.d - std::exp(r * dt),
          s.p * s.u * s.u + (1.0 - s.p) * s.d * s.d -
              std::exp((2.0 * r + sigma * sigma) * dt)};
}
// Residual of (4.60) for given u, d.
[[nodiscard]] inline double ex05_residual_460(double u, double d, double r,
                                              double sigma, double dt) {
  const double g = std::exp(r * dt);
  return (g - d) * (u - g) - g * g * std::expm1(sigma * sigma * dt);
}

// ------------------------------------------------------------ Exercise 6 ---
// Find a binomial tree parametrization with risk-neutral probability 1/2:
// find u, d with p u + (1 - p) d = e^(r dt), p u^2 + (1 - p) d^2 = e^((2r +
// sigma^2) dt), p = 1/2.
//
// With p = 1/2: u + d = 2 e^(r dt) and u^2 + d^2 = 2 e^((2r + sigma^2) dt).
// Write u = e^(r dt) + x, d = e^(r dt) - x; then u^2 + d^2 = 2 e^(2r dt) + 2
// x^2, so
//   x^2 = e^(2r dt)(e^(sigma^2 dt) - 1),
//   u = e^(r dt)(1 + sqrt(e^(sigma^2 dt) - 1)),  d = e^(r dt)(1 -
//   sqrt(e^(sigma^2 dt) - 1)).
// (Taking u > d.) d > 0 needs e^(sigma^2 dt) < 2, i.e. sigma^2 dt < ln 2, which
// holds for any reasonable time step. For small dt, u ~ e^(r dt)(1 + sigma
// sqrt(dt)).
[[nodiscard]] inline BinomialStep ex06_equal_probability(double r, double sigma,
                                                         double dt) {
  const double x = std::sqrt(std::expm1(sigma * sigma * dt));
  if (x >= 1.0)
    throw std::domain_error("ex06_equal_probability: need sigma^2 dt < ln 2");
  const double g = std::exp(r * dt);
  return {g * (1.0 + x), g * (1.0 - x), 0.5};
}

}  // namespace p4mfe::ch04::ex
