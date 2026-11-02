#pragma once
// Exercise 12 (section 7.6): the relation 1 + sigma^2 S^2/2 * Gamma/Theta ~ 0.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch07::ex {

// ----------------------------------------------------------- Exercise 12 ---
// Notation: tau = T - t is the time to maturity, Theta = dV/dt.
//
// (i) Show that 1 + sigma^2 S^2/2 * Gamma(C)/Theta(C) = 0 if q = 0 and r = 0.
//   With r = q = 0 the Black-Scholes Greeks (3.72)-(3.78) are
//     Gamma(C) = phi(d1) / (S sigma sqrt(tau)),
//     Theta(C) = -S sigma phi(d1) / (2 sqrt(tau)).
//   Then
//     sigma^2 S^2/2 * Gamma/Theta
//       = sigma^2 S^2/2 * [phi(d1) / (S sigma sqrt(tau))] * [-2 sqrt(tau) / (S
//       sigma phi(d1))] = -1,
//   so 1 + sigma^2 S^2/2 * Gamma/Theta = 0. (This is just the Black-Scholes PDE
//   Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0 with r = q = 0.)
//   Since Gamma(P) = Gamma(C) and, for r = q = 0, Theta(P) = Theta(C), the same
//   holds for puts.
//
// (ii) If q = 0 but r != 0, show that
//     1 + sigma^2 S^2/2 * Gamma(C)/Theta(C) = 1 / (1 + sigma/(2 r sqrt(tau)) *
//     N'(d2)/N(d2)).
//   With q = 0,
//     Theta(C) = -S sigma phi(d1) / (2 sqrt(tau)) - r K e^(-r tau) N(d2),
//     sigma^2 S^2/2 * Gamma(C) = S sigma phi(d1) / (2 sqrt(tau)).
//   By the identity S phi(d1) = K e^(-r tau) phi(d2) (section 3.6.1), with
//   A = K e^(-r tau) sigma phi(d2) / (2 sqrt(tau)) and B = r K e^(-r tau)
//   N(d2):
//     sigma^2 S^2/2 * Gamma = A,   Theta = -(A + B),
//   so
//     1 + sigma^2 S^2/2 * Gamma/Theta = 1 - A/(A + B) = B/(A + B) = 1/(1 +
//     A/B),
//   and A/B = sigma phi(d2) / (2 r sqrt(tau) N(d2)), with phi = N'. This is the
//   claimed formula. For r > 0 it lies in (0, 1); it tends to 0 as r -> 0,
//   recovering (i) (A/B -> infinity).
struct Ex12Market {
  double S = 50.0, K = 52.0, tau = 0.75, sigma = 0.3, r = 0.05;
};

[[nodiscard]] inline double ex12_lhs(
    const Ex12Market& m, ch03::OptionType type = ch03::OptionType::kCall) {
  const auto g = ch03::bs_greeks(type, m.S, m.K, m.tau, m.sigma, m.r, 0.0);
  return 1.0 + 0.5 * m.sigma * m.sigma * m.S * m.S * g.gamma / g.theta;
}
// The right-hand side of (ii) (for r != 0).
[[nodiscard]] inline double ex12_rhs(const Ex12Market& m) {
  const auto [d1, d2] = ch03::bs_d1_d2(m.S, m.K, m.tau, m.sigma, m.r, 0.0);
  return 1.0 / (1.0 + m.sigma / (2.0 * m.r * std::sqrt(m.tau)) * norm_pdf(d2) /
                          norm_cdf(d2));
}

}  // namespace p4mfe::ch07::ex
