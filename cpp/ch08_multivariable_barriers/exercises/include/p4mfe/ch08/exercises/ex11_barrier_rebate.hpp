#pragma once
// Exercise 11 (section 8.12): the rebate boundary condition of a
// down-and-out call in the heat equation variables.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <p4mfe/ch08/barrier.hpp>
#include <p4mfe/ch08/exercises/ex09_10_heat_change_of_variables.hpp>
#include <p4mfe/ch08/heat_equation.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 11 ---
// What does the boundary condition V(B, t) = R for a down-and-out call with
// barrier B and rebate R > 0 correspond to for u(x, tau), where
// V(S, t) = exp(-a x - b tau) u(x, tau) with x, tau, a, b as in exercise 9?
//
// The barrier S = B is the line x = x_B = ln(B/K) in the (x, tau) plane, and
// t ranges over [0, T] iff tau ranges over [0, tau_final], tau_final = sigma^2
// T/2. From u(x, tau) = exp(a x + b tau) V(S, t),
//   u(ln(B/K), tau) = exp(a ln(B/K) + b tau) R = R (B/K)^a e^(b tau),   0 <=
//   tau <= sigma^2 T/2.
// So the constant rebate becomes an exponentially growing (b > 0) Dirichlet
// condition on the left boundary x = ln(B/K) of the heat equation u_tau = u_xx
// on x > ln(B/K). Together with u(x, 0) = K e^(ax) max(e^x - 1, 0) for
// x > ln(B/K) (exercise 9) this determines u. For R = 0 it is the zero boundary
// condition of section 8.9.
//
// V(B, t) = R means the rebate is paid as soon as the barrier is hit. By
// linearity V = (down-and-out call without rebate) + R * (value of 1 paid at
// the hitting time if it is before T). The second term solves the heat problem
// with u(x, 0) = 0 and the boundary above, and is (Reiner-Rubinstein)
//   R [ (B/S)^(a + l) N(z) + (B/S)^(a - l) N(z - 2 l sigma sqrt(T)) ],
//   l = sqrt(a^2 + 2r/sigma^2) = sqrt(b),   z = ln(B/S)/(sigma sqrt(T)) + l
//   sigma sqrt(T).
// The check below solves the heat equation with the boundary condition derived
// above by explicit finite differences and compares with this closed form.
[[nodiscard]] inline double ex11_u_boundary(double tau, double B, double K,
                                            double R, double sigma, double r,
                                            double q = 0.0) {
  const auto c = ex09_coefficients(sigma, r, q);
  return R * std::pow(B / K, c.a) * std::exp(c.b * tau);
}

// Value of 1 paid when S first hits B < S, if before T (needs b >= 0, e.g. r >=
// 0).
[[nodiscard]] inline double ex11_rebate_value(double S, double B, double T,
                                              double sigma, double r,
                                              double q = 0.0) {
  if (S <= B) return 1.0;
  const auto c = ex09_coefficients(sigma, r, q);
  if (c.b < 0.0)
    throw std::invalid_argument(
        "ex11_rebate_value: needs a^2 + 2r/sigma^2 >= 0");
  const double l = std::sqrt(c.b), st = sigma * std::sqrt(T),
               h = std::log(B / S);
  const double z = h / st + l * st;
  return std::exp((c.a + l) * h) * norm_cdf(z) +
         std::exp((c.a - l) * h) * norm_cdf(z - 2.0 * l * st);
}

[[nodiscard]] inline double ex11_down_and_out_call_with_rebate(
    double S, double K, double B, double R, double T, double sigma, double r,
    double q = 0.0) {
  return down_and_out_call(S, K, B, T, sigma, r, q) +
         R * ex11_rebate_value(S, B, T, sigma, r, q);
}

// Explicit finite differences for u_tau = u_xx on [ln(B/K), x_max] with the
// boundary condition of the exercise on the left, the deep in-the-money value
// S e^(-q theta) - K e^(-r theta) on the right, and the call initial condition.
// Returns V(S, 0). B < K assumed.
[[nodiscard]] inline double ex11_heat_fd(double S, double K, double B, double R,
                                         double T, double sigma, double r,
                                         double q = 0.0, std::size_t nx = 400) {
  if (!(B < S && B < K))
    throw std::invalid_argument("ex11_heat_fd: need B < S and B < K");
  const auto c = ex09_coefficients(sigma, r, q);
  const double x_min = std::log(B / K);
  const double x_max = std::log(S / K) + 6.0 * sigma * std::sqrt(T);
  const double S_max = K * std::exp(x_max);
  const double tau_final = 0.5 * sigma * sigma * T;
  const double dx = (x_max - x_min) / static_cast<double>(nx);
  const auto nt =
      static_cast<std::size_t>(std::ceil(tau_final / (0.45 * dx * dx)));
  const auto u0 = [&](double x) { return ex09_call_u0(x, K, c.a); };
  const auto left = [&](double tau) {
    return ex11_u_boundary(tau, B, K, R, sigma, r, q);
  };
  const auto right = [&](double tau) {
    const double th = 2.0 * tau / (sigma * sigma);
    return std::exp(c.a * x_max + c.b * tau) *
           (S_max * std::exp(-q * th) - K * std::exp(-r * th));
  };
  const auto g =
      solve_heat_explicit(u0, left, right, x_min, x_max, tau_final, nx, nt);
  const double x = std::log(S / K);
  return std::exp(-c.a * x - c.b * tau_final) * interpolate(g.x, g.u, x);
}

}  // namespace p4mfe::ch08::ex
