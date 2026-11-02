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

[[nodiscard]] inline HeatTransform MakeHeatTransform(double sigma, double r,
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
[[nodiscard]] inline HeatPoint ToHeatVariables(double spot, double strike,
                                               double time_to_maturity,
                                               double sigma) {
  return {std::log(spot / strike), 0.5 * sigma * sigma * time_to_maturity};
}

// V = K e^(a x + b tau) u and its inverse.
[[nodiscard]] inline double BsFromHeat(double u, double x, double tau,
                                       double strike,
                                       const HeatTransform& c) noexcept {
  return strike * std::exp(c.a * x + c.b * tau) * u;
}
[[nodiscard]] inline double HeatFromBs(double value, double x, double tau,
                                       double strike,
                                       const HeatTransform& c) noexcept {
  return value / (strike * std::exp(c.a * x + c.b * tau));
}

// u(x, tau) = E[u0(x + sqrt(2 tau) Z)], the heat kernel solution.
template <std::invocable<double> U0>
[[nodiscard]] double HeatSolution(U0&& u0, double x, double tau,
                                  double tol = 1e-9) {
  if (tau == 0.0) return u0(x);
  const ch03::ContinuousDistribution std_normal{
      [](double z) { return norm_pdf(z); }, -ch02::kInfinity, ch02::kInfinity};
  const double s = std::sqrt(2.0 * tau);
  return ch03::ExpectedValue(
      std_normal, [&](double z) { return u0(x + s * z); }, tol);
}

// European price of payoff h by solving the heat equation in closed form.
template <std::invocable<double> Payoff>
[[nodiscard]] double BsPriceViaHeat(Payoff&& h, double spot, double strike,
                                    double maturity, double sigma, double r,
                                    double q = 0.0) {
  const auto c = MakeHeatTransform(sigma, r, q);
  const auto [x, tau] = ToHeatVariables(spot, strike, maturity, sigma);
  const auto u0 = [&](double y) {
    return std::exp(-c.a * y) * h(strike * std::exp(y)) / strike;
  };
  return BsFromHeat(HeatSolution(u0, x, tau), x, tau, strike, c);
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
[[nodiscard]] HeatGrid SolveHeatExplicit(U0&& u0, L&& left, R&& right,
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
[[nodiscard]] inline double Interpolate(const std::vector<double>& xs,
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
[[nodiscard]] double BsHeatFd(Payoff&& h, VL&& v_left, VR&& v_right,
                              double spot, double strike, double maturity,
                              double sigma, double r, double q, double spot_min,
                              double spot_max, std::size_t nx,
                              std::size_t nt = 0) {
  const auto c = MakeHeatTransform(sigma, r, q);
  const double x_min = std::log(spot_min / strike),
               x_max = std::log(spot_max / strike);
  const double tau_final = 0.5 * sigma * sigma * maturity;
  if (nt == 0) {
    const double dx = (x_max - x_min) / static_cast<double>(nx);
    nt = static_cast<std::size_t>(std::ceil(tau_final / (0.45 * dx * dx)));
  }
  const auto u0 = [&](double x) {
    return HeatFromBs(h(strike * std::exp(x)), x, 0.0, strike, c);
  };
  const auto theta = [&](double tau) { return 2.0 * tau / (sigma * sigma); };
  const auto left = [&](double tau) {
    return HeatFromBs(v_left(theta(tau)), x_min, tau, strike, c);
  };
  const auto right = [&](double tau) {
    return HeatFromBs(v_right(theta(tau)), x_max, tau, strike, c);
  };
  const auto g =
      SolveHeatExplicit(u0, left, right, x_min, x_max, tau_final, nx, nt);
  const double x = std::log(spot / strike);
  return BsFromHeat(Interpolate(g.x, g.u, x), x, tau_final, strike, c);
}

// Calls and puts, on [K e^(-L), K e^L] with L = |ln(S/K)| + 6 sigma sqrt(T).
[[nodiscard]] inline double EuropeanHeatFd(OptionType type, double spot,
                                           double strike, double maturity,
                                           double sigma, double r,
                                           double q = 0.0, std::size_t nx = 400,
                                           std::size_t nt = 0) {
  const double half_width =
      std::abs(std::log(spot / strike)) + 6.0 * sigma * std::sqrt(maturity);
  const double spot_min = strike * std::exp(-half_width),
               spot_max = strike * std::exp(half_width);
  const auto h = [&](double s) { return ch01::Payoff(type, s, strike); };
  if (type == OptionType::kCall)
    return BsHeatFd(
        h, [](double) { return 0.0; },
        [&](double th) {
          return spot_max * std::exp(-q * th) - strike * std::exp(-r * th);
        },
        spot, strike, maturity, sigma, r, q, spot_min, spot_max, nx, nt);
  return BsHeatFd(
      h,
      [&](double th) {
        return strike * std::exp(-r * th) - spot_min * std::exp(-q * th);
      },
      [](double) { return 0.0; }, spot, strike, maturity, sigma, r, q, spot_min,
      spot_max, nx, nt);
}

}  // namespace p4mfe::ch08
