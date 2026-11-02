#pragma once
// Exercises 5-9 (section 3.11): mean and variance of a three-step binomial
// stock price, of the uniform, exponential and Poisson distributions, and
// E[|X|], E[X^2] for a normal variable.

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/probability.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch03::ex {

// ------------------------------------------------------------ Exercise 5 ---
// Recall the stock evolution model from section 3.1: over each of three
// consecutive time intervals of length tau = 1/12, a stock with spot price
// S0 = 40 at time t = 0 will either go up by a factor u = 1.05 with
// probability p = 0.6, or down by a factor d = 0.96 with probability
// 1 - p = 0.4. Compute E[S_T] and var(S_T) for T = 3 tau.
//
// S_T = S0 u^j d^(3-j) with probability C(3,j) p^j (1-p)^(3-j). The factors
// over the three periods are independent, so
//   E[S_T]   = S0 (p u + (1-p) d)^3        = 40 * 1.014^3   = 41.703630,
//   E[S_T^2] = S0^2 (p u^2 + (1-p) d^2)^3  = 1600 * 1.03014^3,
//   var(S_T) = E[S_T^2] - E[S_T]^2         = 9.883487,
// so the standard deviation of S_T is 3.1438.
// (The four values 46.305, 42.336, 38.70720, 35.38944 have probabilities
// 0.216, 0.432, 0.288, 0.064; ex05_distribution builds them.)
struct Ex05Tree {
  double S0 = 40.0;
  double u = 1.05;
  double d = 0.96;
  double p = 0.6;
  int steps = 3;
};

[[nodiscard]] inline DiscreteDistribution ex05_distribution(
    const Ex05Tree& t = {}) {
  std::vector<double> v, pr;
  double binom = 1.0;  // C(steps, j)
  for (int j = 0; j <= t.steps; ++j) {
    v.push_back(t.S0 * std::pow(t.u, j) * std::pow(t.d, t.steps - j));
    pr.push_back(binom * std::pow(t.p, j) * std::pow(1.0 - t.p, t.steps - j));
    binom = binom * (t.steps - j) / (j + 1);
  }
  return {std::move(v), std::move(pr)};
}
[[nodiscard]] inline double ex05_mean(const Ex05Tree& t = {}) {
  return t.S0 * std::pow(t.p * t.u + (1.0 - t.p) * t.d, t.steps);
}
[[nodiscard]] inline double ex05_variance(const Ex05Tree& t = {}) {
  const double m2 =
      t.S0 * t.S0 *
      std::pow(t.p * t.u * t.u + (1.0 - t.p) * t.d * t.d, t.steps);
  const double m = ex05_mean(t);
  return m2 - m * m;
}

// ------------------------------------------------------------ Exercise 6 ---
// Calculate the mean and variance of the uniform distribution on [a, b].
//
// f(x) = 1/(b-a) on [a, b]:
//   E[X]   = (1/(b-a)) int_a^b x dx   = (b^2 - a^2)/(2(b-a)) = (a+b)/2,
//   E[X^2] = (b^3 - a^3)/(3(b-a))     = (a^2 + ab + b^2)/3,
//   var(X) = (a^2 + ab + b^2)/3 - (a+b)^2/4 = (b-a)^2/12.
[[nodiscard]] constexpr double ex06_mean(double a, double b) noexcept {
  return 0.5 * (a + b);
}
[[nodiscard]] constexpr double ex06_variance(double a, double b) noexcept {
  return (b - a) * (b - a) / 12.0;
}

// ------------------------------------------------------------ Exercise 7 ---
// The density of the exponential random variable X with parameter alpha > 0
// is f(x) = alpha e^(-alpha x) for x >= 0 and 0 for x < 0.
// (i)   Show that f is a density: int f = 1.
// (ii)  Show that E[X] = 1/alpha and var(X) = 1/alpha^2.
// (iii) Show that the cumulative distribution is F(x) = 1 - e^(-alpha x),
//       x >= 0, and 0 otherwise.
// (iv)  Show that P(X >= t) = e^(-alpha t). (So X is memoryless:
//       P(X >= t+s | X >= t) = e^(-alpha(t+s))/e^(-alpha t) = P(X >= s).)
//
// (i)   int_0^inf alpha e^(-alpha x) dx = [-e^(-alpha x)]_0^inf = 1, and f >=
// 0. (ii)  By parts, int_0^inf x^n alpha e^(-alpha x) dx = (n/alpha) int_0^inf
//       x^(n-1) alpha e^(-alpha x) dx, so E[X] = 1/alpha, E[X^2] = 2/alpha^2,
//       var(X) = 2/alpha^2 - 1/alpha^2 = 1/alpha^2.
// (iii) F(x) = int_0^x alpha e^(-alpha s) ds = 1 - e^(-alpha x) for x >= 0.
// (iv)  P(X >= t) = 1 - F(t) = e^(-alpha t).
// The checks use exponential_distribution() from probability.hpp, whose
// moments are computed by numerical integration.
[[nodiscard]] inline double ex07_cdf(double alpha, double x) {
  return x < 0.0 ? 0.0 : -std::expm1(-alpha * x);
}
[[nodiscard]] inline double ex07_tail(double alpha, double t) {
  return t < 0.0 ? 1.0 : std::exp(-alpha * t);
}
// P(X >= t + s | X >= t), which equals P(X >= s).
[[nodiscard]] inline double ex07_conditional_tail(double alpha, double t,
                                                  double s) {
  return ex07_tail(alpha, t + s) / ex07_tail(alpha, t);
}

// ------------------------------------------------------------ Exercise 8 ---
// Let X be normal with mean mu and standard deviation sigma > 0. Compute
// E[|X|] and E[X^2].
//
// Write X = mu + sigma Z. |X| changes sign at z0 = -mu/sigma, so
//   E|X| = int_{z0}^inf (mu + sigma z) phi(z) dz - int_{-inf}^{z0} (mu + sigma
//   z) phi(z) dz.
// With int z phi(z) dz = -phi(z),
//   int_{z0}^inf  (mu + sigma z) phi = mu (1 - N(z0)) + sigma phi(z0),
//   int_{-inf}^{z0} (mu + sigma z) phi = mu N(z0) - sigma phi(z0),
// so
//   E|X| = mu (1 - 2 N(-mu/sigma)) + 2 sigma phi(mu/sigma)
//        = mu (1 - 2 N(-mu/sigma)) + sigma sqrt(2/pi) e^(-mu^2/(2 sigma^2)).
// For mu = 0 this is sigma sqrt(2/pi). Also
//   E[X^2] = var(X) + E[X]^2 = sigma^2 + mu^2.
[[nodiscard]] inline double ex08_mean_abs(double mu, double sigma) {
  return mu * (1.0 - 2.0 * norm_cdf(-mu / sigma)) +
         2.0 * sigma * norm_pdf(mu / sigma);
}
[[nodiscard]] constexpr double ex08_second_moment(double mu,
                                                  double sigma) noexcept {
  return sigma * sigma + mu * mu;
}

// ------------------------------------------------------------ Exercise 9 ---
// Compute the expected value and variance of the Poisson distribution,
// P(X = k) = e^(-lambda) lambda^k / k!, k >= 0, lambda > 0.
//
//   E[X] = sum_{k>=1} k e^(-lambda) lambda^k/k! = lambda e^(-lambda) sum_{k>=1}
//   lambda^(k-1)/(k-1)!
//        = lambda e^(-lambda) e^lambda = lambda.
//   E[X(X-1)] = lambda^2 e^(-lambda) sum_{k>=2} lambda^(k-2)/(k-2)! = lambda^2,
//   var(X) = E[X(X-1)] + E[X] - E[X]^2 = lambda^2 + lambda - lambda^2 = lambda.
// ex09_distribution truncates the series where the tail is below 1e-15, to
// check both values by direct summation.
[[nodiscard]] inline DiscreteDistribution ex09_distribution(double lambda) {
  if (!(lambda > 0.0))
    throw std::invalid_argument("ex09: lambda must be positive");
  std::vector<double> v, pr;
  double pk = std::exp(-lambda), total = 0.0;
  for (int k = 0; k < 100000; ++k) {
    v.push_back(k);
    pr.push_back(pk);
    total += pk;
    if (k > lambda && 1.0 - total < 1e-15) break;
    pk *= lambda / (k + 1);
  }
  for (double& p : pr) p /= total;  // fold the tiny tail back in
  return {std::move(v), std::move(pr)};
}

}  // namespace p4mfe::ch03::ex
