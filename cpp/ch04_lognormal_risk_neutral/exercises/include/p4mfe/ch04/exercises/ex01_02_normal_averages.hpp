#pragma once
// Exercises 1-2 (section 4.9): sums and averages of normal variables.

#include <cmath>
#include <cstddef>
#include <p4mfe/ch04/independent.hpp>
#include <vector>

namespace p4mfe::ch04::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Let X1 = Z and X2 = -Z be two independent random variables, where Z is the
// standard normal variable. Show that X1 + X2 is a normal variable of mean 0
// and variance 2, i.e., X1 + X2 = sqrt(2) Z.
//
// Note on the text: read literally, X1 = Z and X2 = -Z are the same draw with
// opposite signs, so they are not independent (cov(X1, X2) = -1) and
// X1 + X2 = 0 for every outcome. The intended statement is that X1 and X2 are
// independent, each with the law of Z (X2 = -Z in distribution, since the
// standard normal is symmetric). Then:
//
// The moment generating function of N(m, s^2) is E[e^(tX)] = e^(mt + s^2 t^2 /
// 2), and it determines the law. For independent X1, X2:
//   E[e^(t(X1 + X2))] = E[e^(t X1)] E[e^(t X2)] = e^(t^2/2) e^(t^2/2) = e^(2
//   t^2 / 2),
// which is the MGF of N(0, 2). So X1 + X2 ~ N(0, 2), the law of sqrt(2) Z
// (equal in distribution, not as random variables). In general
// sum a_i X_i ~ N(sum a_i mu_i, sum a_i^2 sigma_i^2) for independent normals,
// which is ch04::independent_normal_combination.
[[nodiscard]] inline NormalParams Ex01SumIndependent() {
  const std::vector<NormalParams> x_dist{{0.0, 1.0}, {0.0, 1.0}};
  return IndependentNormalSum(x_dist);
}
// The literal reading: X1 + X2 = z + (-z) = 0 for every draw z.
[[nodiscard]] inline double Ex01SumAsWritten(double z) noexcept {
  return z + (-z);
}

// ------------------------------------------------------------ Exercise 2 ---
// Assume that the normal random variables X1, ..., Xn of mean mu and variance
// sigma^2 are uncorrelated, i.e. cov(Xi, Xj) = 0 for all i != j. If
// S_n = (1/n) sum Xi is their average, show that E[S_n] = mu and
// var(S_n) = sigma^2 / n.
//
// Linearity of expectation: E[S_n] = (1/n) sum E[Xi] = (1/n) n mu = mu.
// Variance of a sum: var(sum Xi) = sum var(Xi) + 2 sum_{i<j} cov(Xi, Xj)
//   = n sigma^2 + 0, so var(S_n) = (1/n^2) n sigma^2 = sigma^2 / n.
// Only uncorrelatedness is used (normality is not needed for these two
// moments); the standard deviation of the average shrinks like 1/sqrt(n).
// If the Xi are also independent, S_n ~ N(mu, sigma^2/n) by Exercise 1.
[[nodiscard]] inline NormalParams Ex02Average(double mu, double sigma,
                                              std::size_t n) {
  const std::vector<double> a(n, 1.0 / static_cast<double>(n));
  const std::vector<NormalParams> x_dist(n, NormalParams{mu, sigma * sigma});
  return IndependentNormalCombination(a, x_dist);
}

}  // namespace p4mfe::ch04::ex
