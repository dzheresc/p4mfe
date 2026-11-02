#pragma once
// Exercises 12-13 (section 1.12): limits of integrals.

#include <concepts>
#include <cstddef>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch01::ex {

// ----------------------------------------------------------- Exercise 12 ---
// Let f be continuous. Show that
//   lim_{h->0} 1/(2h) integral_{a-h}^{a+h} f(x) dx = f(a)   for all a.
// (Note: with F an antiderivative of f, this is the central difference
//  F'(a) = (F(a+h) - F(a-h))/(2h) + O(h^2), so the error is O(h^2) when f''
//  is continuous.)
//
// Proof. By the mean value theorem for integrals there is c_h in [a-h, a+h]
// with integral_{a-h}^{a+h} f = 2h f(c_h). As h -> 0, c_h -> a, and f is
// continuous, so f(c_h) -> f(a). (Equivalently: for eps > 0 pick delta with
// |f(x) - f(a)| < eps on |x - a| < delta; then for h < delta the average of
// f over [a-h, a+h] is within eps of f(a).)
//
// The local average itself, with an n-interval midpoint sum for the integral.
template <std::invocable<double> F>
[[nodiscard]] double ex12_local_average(F&& f, double a, double h,
                                        std::size_t n = 64) {
  return riemann_sum(f, a - h, a + h, n, RiemannRule::Midpoint) / (2.0 * h);
}

// ----------------------------------------------------------- Exercise 13 ---
// Let f(x) = 1/(sigma sqrt(2 pi)) exp(-(x - mu)^2 / (2 sigma^2)), and let g be
// continuous with |g(x)| <= C for all x. Show that
//   lim_{sigma -> 0+} integral_{-inf}^{inf} f(x) g(x) dx = g(mu).
// (Note: the normal densities converge, as distributions, to the delta
//  function at mu.)
//
// Proof. Substitute x = mu + sigma z:
//   integral f(x) g(x) dx = integral phi(z) g(mu + sigma z) dz,
// with phi the standard normal density, and integral phi = 1, so
//   |integral f g - g(mu)| <= integral phi(z) |g(mu + sigma z) - g(mu)| dz.
// Split at |z| = M. On |z| > M the integrand is <= 2C phi(z), total
// <= 2C P(|Z| > M), small for M large. On |z| <= M, |sigma z| <= sigma M -> 0
// and g is continuous at mu, so |g(mu + sigma z) - g(mu)| < eps for sigma
// small. Hence the difference is <= 2C P(|Z| > M) + eps, arbitrarily small.
// (This is dominated convergence with dominating function 2C phi.)
//
// The smoothed value integral phi(z) g(mu + sigma z) dz, truncated to
// |z| <= z_max (the tail mass beyond 10 is ~1.5e-23) and integrated with an
// n-interval midpoint sum.
template <std::invocable<double> G>
[[nodiscard]] double ex13_gaussian_average(G&& g, double mu, double sigma,
                                           std::size_t n = 4000,
                                           double z_max = 10.0) {
  return riemann_sum([&](double z) { return norm_pdf(z) * g(mu + sigma * z); },
                     -z_max, z_max, n, RiemannRule::Midpoint);
}

}  // namespace p4mfe::ch01::ex
