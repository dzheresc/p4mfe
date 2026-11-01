#pragma once
// Exercise 11 (section 6.11): small time step expansion of the
// Cox-Ross-Rubinstein binomial tree parameters.

#include <cmath>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 11 ---
// In the Cox-Ross-Rubinstein parametrization of a binomial tree,
//   u = A + sqrt(A^2 - 1),  d = A - sqrt(A^2 - 1),  p = (e^(r dt) - d)/(u - d),   (6.131-6.133)
//   A = (e^(-r dt) + e^((r + sigma^2) dt)) / 2.
// Show that for a small time step dt
//   u ~ e^(sigma sqrt(dt)),  d ~ e^(-sigma sqrt(dt)),
//   p ~ 1/2 + (1/2)(r/sigma - sigma/2) sqrt(dt),                                 (6.134-6.136)
// i.e. that both sets of formulas have the same Taylor expansions once terms
// of order O(dt) and smaller are neglected.
//
// A: e^(-r dt) = 1 - r dt + O(dt^2) and e^((r + sigma^2) dt) = 1 + (r + sigma^2) dt + O(dt^2), so
//   A = 1 + sigma^2 dt / 2 + c dt^2 + O(dt^3),  c = (r^2 + (r + sigma^2)^2)/4,
//   A^2 - 1 = sigma^2 dt + O(dt^2),  sqrt(A^2 - 1) = sigma sqrt(dt) (1 + O(dt)).
// u, d:
//   u = 1 + sigma sqrt(dt) + sigma^2 dt / 2 + O(dt^(3/2)),
//   d = 1 - sigma sqrt(dt) + sigma^2 dt / 2 + O(dt^(3/2)),
// and e^(+-sigma sqrt(dt)) = 1 +- sigma sqrt(dt) + sigma^2 dt / 2 + O(dt^(3/2)): the
// same up to O(dt^(3/2)), in fact. Carrying one more order,
//   u - e^(sigma sqrt(dt)) = -(d - e^(-sigma sqrt(dt))) + O(dt^2)
//                          = (r (r + sigma^2) / (2 sigma) + 5 sigma^3 / 24) dt^(3/2) + O(dt^2).
// p: since (u + d)/2 = A and u - d = 2 sqrt(A^2 - 1),
//   p = (e^(r dt) - A + (u - d)/2)/(u - d) = 1/2 + (e^(r dt) - A) / (2 sqrt(A^2 - 1)),
// and e^(r dt) - A = (r - sigma^2/2) dt + O(dt^2), so
//   p = 1/2 + (r - sigma^2/2) dt / (2 sigma sqrt(dt)) (1 + O(dt))
//     = 1/2 + (1/2)(r/sigma - sigma/2) sqrt(dt) + O(dt^(3/2)),
// which is (6.136). (The O(dt) term of p vanishes too, so the two p's also
// agree to O(dt^(3/2)).)
struct Ex11Tree {
    double u, d, p;
};

[[nodiscard]] inline double ex11_A(double r, double sigma, double dt) {
    return 0.5 * (std::exp(-r * dt) + std::exp((r + sigma * sigma) * dt));
}
// (6.131)-(6.133). sqrt(A^2 - 1) is computed as sqrt((A - 1)(A + 1)) with
// A - 1 from expm1, which avoids the cancellation in A^2 - 1 for small dt.
[[nodiscard]] inline Ex11Tree ex11_crr(double r, double sigma, double dt) {
    const double A = ex11_A(r, sigma, dt);
    const double Am1 = 0.5 * (std::expm1(-r * dt) + std::expm1((r + sigma * sigma) * dt));
    const double root = std::sqrt(Am1 * (A + 1.0));
    const double u = A + root, d = A - root;
    return {u, d, (std::exp(r * dt) - d) / (u - d)};
}
// (6.134)-(6.136).
[[nodiscard]] inline Ex11Tree ex11_approx(double r, double sigma, double dt) {
    const double s = sigma * std::sqrt(dt);
    return {std::exp(s), std::exp(-s), 0.5 + 0.5 * (r / sigma - 0.5 * sigma) * std::sqrt(dt)};
}
// Leading coefficient of u_CRR - e^(sigma sqrt(dt)) in powers of dt^(3/2).
[[nodiscard]] constexpr double ex11_u_coefficient(double r, double sigma) noexcept {
    return r * (r + sigma * sigma) / (2.0 * sigma) + 5.0 * sigma * sigma * sigma / 24.0;
}

}  // namespace p4mfe::ch06::ex
