#pragma once
// Chapter 8, section 8.8: the Black-Scholes PDE and the heat equation.
//
// With x = ln(S/K), tau = sigma^2 (T - t) / 2 and V(S, t) = K v(x, tau), the
// Black-Scholes PDE (section 7.4) becomes
//   v_tau = v_xx + (k1 - 1) v_x - k0 v,   k1 = 2 (r - q) / sigma^2, k0 = 2 r /
//   sigma^2.
// Setting v = e^(a x + b tau) u with
//   a = -(k1 - 1) / 2,   b = -a^2 - k0
// removes the v_x and v terms and leaves the heat equation u_tau = u_xx, with
// initial condition u(x, 0) = e^(-a x) h(K e^x) / K for the payoff h. Its
// solution is the convolution with the heat kernel,
//   u(x, tau) = integral u0(y) e^(-(x - y)^2 / (4 tau)) / sqrt(4 pi tau) dy =
//   E[u0(x + sqrt(2 tau) Z)],
// which for a call gives back the Black-Scholes formula.
//
// The heat equation is also the simplest PDE to solve by finite differences
// (chapter 7): the explicit scheme
//   u_i^(n+1) = u_i^n + alpha (u_(i+1)^n - 2 u_i^n + u_(i-1)^n),  alpha = dtau
//   / dx^2,
// is stable for alpha <= 1/2. `bs_heat_fd` prices any European payoff this
// way, with boundary values given in the original variables; barrier options
// (section 8.9) just put the left boundary at the barrier.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch08 {

using ch01::OptionType;

// ---------------------------------------------------------------- 8.8 ------
struct HeatTransform {
  double a;
  double b;
};

[[nodiscard]] inline HeatTransform heat_transform(double sigma, double r,
                                                  double q = 0.0) {
  if (!(sigma > 0.0))
    throw std::invalid_argument("heat_transform: need sigma > 0");
  const double k1 = 2.0 * (r - q) / (sigma * sigma);
  const double k0 = 2.0 * r / (sigma * sigma);
  const double a = -0.5 * (k1 - 1.0);
  return {a, -a * a - k0};
}

// (x, tau) for spot S and time to maturity T - t.
struct HeatPoint {
  double x;
  double tau;
};
[[nodiscard]] inline HeatPoint to_heat_variables(double S, double K,
                                                 double time_to_maturity,
                                                 double sigma) {
  return {std::log(S / K), 0.5 * sigma * sigma * time_to_maturity};
}

// V = K e^(a x + b tau) u and its inverse.
[[nodiscard]] inline double bs_from_heat(double u, double x, double tau,
                                         double K,
                                         const HeatTransform& c) noexcept {
  return K * std::exp(c.a * x + c.b * tau) * u;
}
[[nodiscard]] inline double heat_from_bs(double V, double x, double tau,
                                         double K,
                                         const HeatTransform& c) noexcept {
  return V / (K * std::exp(c.a * x + c.b * tau));
}

// u(x, tau) = E[u0(x + sqrt(2 tau) Z)], the heat kernel solution.
template <std::invocable<double> U0>
[[nodiscard]] double heat_solution(U0&& u0, double x, double tau,
                                   double tol = 1e-9) {
  if (tau == 0.0) return u0(x);
  const ch03::ContinuousDistribution Z{[](double z) { return norm_pdf(z); },
                                       -ch02::kInfinity, ch02::kInfinity};
  const double s = std::sqrt(2.0 * tau);
  return ch03::ExpectedValue(Z, [&](double z) { return u0(x + s * z); }, tol);
}

// European price of payoff h by solving the heat equation in closed form.
template <std::invocable<double> Payoff>
[[nodiscard]] double bs_price_via_heat(Payoff&& h, double S, double K, double T,
                                       double sigma, double r, double q = 0.0) {
  const auto c = heat_transform(sigma, r, q);
  const auto [x, tau] = to_heat_variables(S, K, T, sigma);
  const auto u0 = [&](double y) {
    return std::exp(-c.a * y) * h(K * std::exp(y)) / K;
  };
  return bs_from_heat(heat_solution(u0, x, tau), x, tau, K, c);
}

// Explicit finite differences for u_tau = u_xx on [x_min, x_max] x [0,
// tau_final].
struct HeatGrid {
  std::vector<double> x;
  std::vector<double> u;  // u(x_i, tau_final)
  double alpha;           // dtau / dx^2
};

template <std::invocable<double> U0, std::invocable<double> L,
          std::invocable<double> R>
[[nodiscard]] HeatGrid solve_heat_explicit(U0&& u0, L&& left, R&& right,
                                           double x_min, double x_max,
                                           double tau_final, std::size_t nx,
                                           std::size_t nt) {
  if (nx < 2 || nt < 1 || !(x_max > x_min))
    throw std::invalid_argument("solve_heat_explicit: bad grid");
  const double dx = (x_max - x_min) / static_cast<double>(nx);
  const double dtau = tau_final / static_cast<double>(nt);
  HeatGrid g{std::vector<double>(nx + 1), std::vector<double>(nx + 1),
             dtau / (dx * dx)};
  for (std::size_t i = 0; i <= nx; ++i) {
    g.x[i] = x_min + static_cast<double>(i) * dx;
    g.u[i] = u0(g.x[i]);
  }
  std::vector<double> next(nx + 1);
  for (std::size_t n = 1; n <= nt; ++n) {
    const double tau = static_cast<double>(n) * dtau;
    for (std::size_t i = 1; i < nx; ++i)
      next[i] = g.u[i] + g.alpha * (g.u[i + 1] - 2.0 * g.u[i] + g.u[i - 1]);
    next[0] = left(tau);
    next[nx] = right(tau);
    g.u.swap(next);
  }
  return g;
}

// Linear interpolation of grid values at x.
[[nodiscard]] inline double interpolate(const std::vector<double>& xs,
                                        const std::vector<double>& ys,
                                        double x) {
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  const auto it = std::upper_bound(xs.begin(), xs.end(), x);
  const std::size_t i = static_cast<std::size_t>(it - xs.begin());
  const double w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return (1.0 - w) * ys[i - 1] + w * ys[i];
}

// European price of payoff h on [S_min, S_max] by the explicit heat scheme.
// V_left(theta), V_right(theta) are the option values at S_min and S_max with
// theta years to maturity. nt = 0 picks the number of time steps so that
// alpha ~ 0.45.
template <std::invocable<double> Payoff, std::invocable<double> VL,
          std::invocable<double> VR>
[[nodiscard]] double bs_heat_fd(Payoff&& h, VL&& V_left, VR&& V_right, double S,
                                double K, double T, double sigma, double r,
                                double q, double S_min, double S_max,
                                std::size_t nx, std::size_t nt = 0) {
  const auto c = heat_transform(sigma, r, q);
  const double x_min = std::log(S_min / K), x_max = std::log(S_max / K);
  const double tau_final = 0.5 * sigma * sigma * T;
  if (nt == 0) {
    const double dx = (x_max - x_min) / static_cast<double>(nx);
    nt = static_cast<std::size_t>(std::ceil(tau_final / (0.45 * dx * dx)));
  }
  const auto u0 = [&](double x) {
    return heat_from_bs(h(K * std::exp(x)), x, 0.0, K, c);
  };
  const auto theta = [&](double tau) { return 2.0 * tau / (sigma * sigma); };
  const auto left = [&](double tau) {
    return heat_from_bs(V_left(theta(tau)), x_min, tau, K, c);
  };
  const auto right = [&](double tau) {
    return heat_from_bs(V_right(theta(tau)), x_max, tau, K, c);
  };
  const auto g =
      solve_heat_explicit(u0, left, right, x_min, x_max, tau_final, nx, nt);
  const double x = std::log(S / K);
  return bs_from_heat(interpolate(g.x, g.u, x), x, tau_final, K, c);
}

// Calls and puts, on [K e^(-L), K e^L] with L = |ln(S/K)| + 6 sigma sqrt(T).
[[nodiscard]] inline double european_heat_fd(OptionType type, double S,
                                             double K, double T, double sigma,
                                             double r, double q = 0.0,
                                             std::size_t nx = 400,
                                             std::size_t nt = 0) {
  const double L = std::abs(std::log(S / K)) + 6.0 * sigma * std::sqrt(T);
  const double S_min = K * std::exp(-L), S_max = K * std::exp(L);
  const auto h = [&](double s) { return ch01::Payoff(type, s, K); };
  if (type == OptionType::kCall)
    return bs_heat_fd(
        h, [](double) { return 0.0; },
        [&](double th) {
          return S_max * std::exp(-q * th) - K * std::exp(-r * th);
        },
        S, K, T, sigma, r, q, S_min, S_max, nx, nt);
  return bs_heat_fd(
      h,
      [&](double th) {
        return K * std::exp(-r * th) - S_min * std::exp(-q * th);
      },
      [](double) { return 0.0; }, S, K, T, sigma, r, q, S_min, S_max, nx, nt);
}

}  // namespace p4mfe::ch08
