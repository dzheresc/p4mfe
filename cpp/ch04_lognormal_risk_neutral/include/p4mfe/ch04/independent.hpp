#pragma once
// Chapter 4, sections 4.3-4.3.2: independent random variables.
//
// X and Y are independent if their joint density factors,
//   f(x, y) = f_X(x) f_Y(y),
// which gives E[g(X) h(Y)] = E[g(X)] E[h(Y)], hence cov(X, Y) = 0, and the
// density of the sum is the convolution
//   f_{X+Y}(z) = integral f_X(x) f_Y(z - x) dx.
//
// 4.3.1: a linear combination of independent normal variables is normal, with
//   mean sum_i a_i mu_i  and  variance sum_i a_i^2 sigma_i^2.
// 4.3.2: so a product of powers of independent lognormal variables is
// lognormal: ln(prod_i Y_i^(a_i)) = sum_i a_i ln(Y_i) is such a combination.

#include <concepts>
#include <cstddef>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <span>
#include <stdexcept>

namespace p4mfe::ch04 {

using ch03::NormalParams;

// ---------------------------------------------------------------- 4.3 ------
// Joint density of independent X and Y.
template <std::invocable<double> FX, std::invocable<double> FY>
[[nodiscard]] auto independent_joint_pdf(FX f_X, FY f_Y) {
  return [=](double x, double y) { return f_X(x) * f_Y(y); };
}

// Density of X + Y for independent X, Y, computed as the convolution
// integral at the point z (chapter 2's improper integral). [lo, hi] is where
// x must lie for f_X(x) f_Y(z - x) to be nonzero; pass it when the densities
// have bounded support (e.g. [0, z] for two variables on [0, inf)), since the
// integrator converges slowly across jumps of the integrand.
template <std::invocable<double> FX, std::invocable<double> FY>
[[nodiscard]] double sum_pdf(const FX& f_X, const FY& f_Y, double z,
                             double lo = -ch02::infinity,
                             double hi = ch02::infinity, double tol = 1e-10) {
  if (!(lo < hi)) return 0.0;
  return ch02::improper_integral([&](double x) { return f_X(x) * f_Y(z - x); },
                                 lo, hi, tol)
      .value;
}

// -------------------------------------------------------------- 4.3.1 ------
// sum_i a_i X_i for independent X_i ~ N(mu_i, sigma_i^2).
[[nodiscard]] inline NormalParams independent_normal_combination(
    std::span<const double> a, std::span<const NormalParams> X) {
  if (a.size() != X.size())
    throw std::invalid_argument("independent_normal_combination: sizes differ");
  NormalParams s{0.0, 0.0};
  for (std::size_t i = 0; i < a.size(); ++i) {
    s.mean += a[i] * X[i].mean;
    s.variance += a[i] * a[i] * X[i].variance;
  }
  return s;
}
// X_1 + ... + X_n.
[[nodiscard]] inline NormalParams independent_normal_sum(
    std::span<const NormalParams> X) {
  NormalParams s{0.0, 0.0};
  for (const auto& x : X) {
    s.mean += x.mean;
    s.variance += x.variance;
  }
  return s;
}

// -------------------------------------------------------------- 4.3.2 ------
// prod_i Y_i^(a_i) for independent lognormal Y_i, each given by the normal
// parameters of ln(Y_i). Returns the normal parameters of the logarithm of
// the product, which is therefore lognormal; feed them to lognormal_mean etc.
// Example: a_i = 1 gives Y_1 Y_2 ...; a = (1, -1) gives the ratio Y_1 / Y_2.
[[nodiscard]] inline NormalParams independent_lognormal_product(
    std::span<const double> a, std::span<const NormalParams> log_Y) {
  return independent_normal_combination(a, log_Y);
}
[[nodiscard]] inline NormalParams independent_lognormal_product(
    std::span<const NormalParams> log_Y) {
  return independent_normal_sum(log_Y);
}

}  // namespace p4mfe::ch04
