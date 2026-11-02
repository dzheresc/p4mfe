#pragma once
// Chapter 3, sections 3.1-3.2.1: discrete and continuous random variables,
// expected value, variance, covariance and correlation.
//
// A discrete random variable is a list of values with their probabilities. A
// continuous one is given by its density f; expectations are integrals
//   E[g(X)] = integral g(x) f(x) dx
// computed with the improper-integral routine of chapter 2.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <p4mfe/ch02/integration.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch03 {

// ---------------------------------------------------------------- 3.1 ------
// P(X = values[i]) = probs[i]; probabilities are nonnegative and sum to 1.
struct DiscreteDistribution {
  std::vector<double> values;
  std::vector<double> probs;

  DiscreteDistribution(std::vector<double> v, std::vector<double> p)
      : values(std::move(v)), probs(std::move(p)) {
    if (values.size() != probs.size() || values.empty())
      throw std::invalid_argument(
          "DiscreteDistribution: values and probs must match and be nonempty");
    double total = 0.0;
    for (double q : probs) {
      if (q < 0.0)
        throw std::invalid_argument(
            "DiscreteDistribution: negative probability");
      total += q;
    }
    if (std::abs(total - 1.0) > 1e-12)
      throw std::invalid_argument(
          "DiscreteDistribution: probabilities must sum to 1");
  }
};

// E[g(X)] = sum_i g(x_i) p_i.
template <std::invocable<double> G>
[[nodiscard]] double expected_value(const DiscreteDistribution& d, G&& g) {
  double s = 0.0;
  for (std::size_t i = 0; i < d.values.size(); ++i)
    s += g(d.values[i]) * d.probs[i];
  return s;
}
[[nodiscard]] inline double mean(const DiscreteDistribution& d) {
  return expected_value(d, [](double x) { return x; });
}
// var(X) = E[(X - E[X])^2] = E[X^2] - E[X]^2.
[[nodiscard]] inline double variance(const DiscreteDistribution& d) {
  const double m = mean(d);
  return expected_value(d, [m](double x) { return (x - m) * (x - m); });
}

// ---------------------------------------------------------------- 3.2 ------
// A continuous random variable on [a, b] (either end may be infinite) with
// density f >= 0, integral of f equal to 1.
template <std::invocable<double> Pdf>
struct ContinuousDistribution {
  Pdf pdf;
  double a = -ch02::infinity;
  double b = ch02::infinity;
};
template <class Pdf>
ContinuousDistribution(Pdf, double, double) -> ContinuousDistribution<Pdf>;

// E[g(X)] = integral_a^b g(x) f(x) dx. Where the density underflows to 0,
// g is not evaluated, so a fast-growing g (e.g. e^(tx)) cannot turn 0 * inf
// into NaN far in the tails.
template <class Pdf, std::invocable<double> G>
[[nodiscard]] double expected_value(const ContinuousDistribution<Pdf>& d, G&& g,
                                    double tol = 1e-9) {
  const auto integrand = [&](double x) {
    const double p = d.pdf(x);
    return p == 0.0 ? 0.0 : g(x) * p;
  };
  return ch02::improper_integral(integrand, d.a, d.b, tol).value;
}
template <class Pdf>
[[nodiscard]] double mean(const ContinuousDistribution<Pdf>& d,
                          double tol = 1e-9) {
  return expected_value(d, [](double x) { return x; }, tol);
}
template <class Pdf>
[[nodiscard]] double variance(const ContinuousDistribution<Pdf>& d,
                              double tol = 1e-9) {
  const double m = mean(d, tol);
  return expected_value(d, [m](double x) { return (x - m) * (x - m); }, tol);
}
// Cumulative distribution F(t) = P(X <= t) = integral_a^t f(x) dx.
template <class Pdf>
[[nodiscard]] double cdf(const ContinuousDistribution<Pdf>& d, double t,
                         double tol = 1e-9) {
  if (t <= d.a) return 0.0;
  if (t >= d.b) return 1.0;
  return ch02::improper_integral(d.pdf, d.a, t, tol).value;
}

// Exponential distribution with parameter lambda: f(x) = lambda e^(-lambda x),
// x >= 0; mean 1/lambda, variance 1/lambda^2. The book's running example.
[[nodiscard]] inline auto exponential_distribution(double lambda) {
  if (lambda <= 0.0)
    throw std::invalid_argument(
        "exponential_distribution: lambda must be positive");
  return ContinuousDistribution{
      [lambda](double x) {
        return x < 0.0 ? 0.0 : lambda * std::exp(-lambda * x);
      },
      0.0, ch02::infinity};
}

// -------------------------------------------------------------- 3.2.1 ------
// Joint discrete distribution of (X, Y): P(X = x_i, Y = y_i) = p_i.
struct JointOutcome {
  double x;
  double y;
  double p;
};

// cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X] E[Y].
[[nodiscard]] inline double covariance(std::span<const JointOutcome> joint) {
  double ex = 0.0, ey = 0.0, exy = 0.0;
  for (const auto& o : joint) {
    ex += o.p * o.x;
    ey += o.p * o.y;
    exy += o.p * o.x * o.y;
  }
  return exy - ex * ey;
}
// corr(X, Y) = cov(X, Y) / (sigma(X) sigma(Y)), always in [-1, 1].
[[nodiscard]] inline double correlation(std::span<const JointOutcome> joint) {
  double ex = 0.0, ey = 0.0;
  for (const auto& o : joint) {
    ex += o.p * o.x;
    ey += o.p * o.y;
  }
  double vx = 0.0, vy = 0.0, c = 0.0;
  for (const auto& o : joint) {
    vx += o.p * (o.x - ex) * (o.x - ex);
    vy += o.p * (o.y - ey) * (o.y - ey);
    c += o.p * (o.x - ex) * (o.y - ey);
  }
  return c / std::sqrt(vx * vy);
}

// var(aX + bY) = a^2 var(X) + b^2 var(Y) + 2ab cov(X, Y).
[[nodiscard]] constexpr double variance_of_sum(double a, double var_x, double b,
                                               double var_y,
                                               double cov_xy) noexcept {
  return a * a * var_x + b * b * var_y + 2.0 * a * b * cov_xy;
}

// Sample estimators from data: mean, unbiased variance and covariance
// (divide by n - 1), and correlation.
[[nodiscard]] inline double sample_mean(std::span<const double> x) {
  if (x.empty()) throw std::invalid_argument("sample_mean: empty sample");
  double s = 0.0;
  for (double v : x) s += v;
  return s / static_cast<double>(x.size());
}
[[nodiscard]] inline double sample_covariance(std::span<const double> x,
                                              std::span<const double> y) {
  if (x.size() != y.size() || x.size() < 2)
    throw std::invalid_argument(
        "sample_covariance: need two samples of equal size >= 2");
  const double mx = sample_mean(x), my = sample_mean(y);
  double s = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) s += (x[i] - mx) * (y[i] - my);
  return s / static_cast<double>(x.size() - 1);
}
[[nodiscard]] inline double sample_variance(std::span<const double> x) {
  return sample_covariance(x, x);
}
[[nodiscard]] inline double sample_correlation(std::span<const double> x,
                                               std::span<const double> y) {
  return sample_covariance(x, y) /
         std::sqrt(sample_variance(x) * sample_variance(y));
}

}  // namespace p4mfe::ch03
