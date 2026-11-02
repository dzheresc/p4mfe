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
// is stable for alpha <= 1/2. `BsHeatFd` prices any European payoff this
// way, with boundary values given in the original variables; barrier options
// (section 8.9) just put the left boundary at the barrier.
//
// The stability limit forces nt ~ nx^2, so the explicit scheme costs O(nx^3).
// `SolveHeatCrankNicolson`, `BsHeatCrankNicolson` and
// `EuropeanHeatCrankNicolson` are a faster alternative (not in the book) with
// the same interface: Crank-Nicolson is stable for any alpha, so nt ~ nx and
// the cost is O(nx^2).

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
      [](double z) { return NormPdf(z); }, -ch02::kInfinity, ch02::kInfinity};
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

// Crank-Nicolson for the same problem: the average of the explicit and the
// fully implicit step,
//   -alpha/2 u_(i-1)^(n+1) + (1 + alpha) u_i^(n+1) - alpha/2 u_(i+1)^(n+1)
//     = alpha/2 u_(i-1)^n + (1 - alpha) u_i^n + alpha/2 u_(i+1)^n,
// is unconditionally stable and second order in dtau, so nt can grow like nx
// instead of nx^2 and the cost drops from O(nx^3) to O(nx^2). The payoff kink
// makes plain Crank-Nicolson ring when alpha is large, so the first
// `smoothing_steps` steps are each replaced by two backward Euler half steps
// (Rannacher start). A backward Euler half step has the same matrix as the
// Crank-Nicolson step, so the tridiagonal system is factored once and each
// step costs O(nx) with no allocation.
template <std::invocable<double> U0, std::invocable<double> L,
          std::invocable<double> R>
[[nodiscard]] HeatGrid SolveHeatCrankNicolson(U0&& u0, L&& left, R&& right,
                                              double x_min, double x_max,
                                              double tau_final, std::size_t nx,
                                              std::size_t nt,
                                              std::size_t smoothing_steps = 2) {
  if (nx < 2 || nt < 1 || !(x_max > x_min))
    throw std::invalid_argument("solve_heat_crank_nicolson: bad grid");
  const double dx = (x_max - x_min) / static_cast<double>(nx);
  const double dtau = tau_final / static_cast<double>(nt);
  HeatGrid g{std::vector<double>(nx + 1), std::vector<double>(nx + 1),
             dtau / (dx * dx)};
  for (std::size_t i = 0; i <= nx; ++i) {
    g.x[i] = x_min + static_cast<double>(i) * dx;
    g.u[i] = u0(g.x[i]);
  }
  // Forward elimination of the constant matrix tridiag(-beta, 1 + alpha,
  // -beta), beta = alpha / 2, on the nx - 1 interior nodes. With pivots p_i,
  // the multipliers m_i = beta / p_(i-1), the inverse pivots and the back
  // substitution coefficients c_i = beta / p_i are stored, so that each sweep
  // carries a single multiply-add from node to node.
  const double beta = 0.5 * g.alpha;
  std::vector<double> multiplier(nx), inv_pivot(nx), back(nx);
  for (std::size_t i = 1; i < nx; ++i) {
    multiplier[i] = i == 1 ? 0.0 : beta * inv_pivot[i - 1];
    inv_pivot[i] = 1.0 / (1.0 + g.alpha - beta * multiplier[i]);
    back[i] = beta * inv_pivot[i];
  }
  // One step from tau to tau_next. crank_nicolson: the right-hand side is
  // (I + beta A) u; otherwise a backward Euler step of dtau / 2, which has the
  // same matrix, with right-hand side u. Both add the boundary terms at
  // tau_next. The forward sweep runs while the right-hand side is formed, and
  // d holds the eliminated right-hand side, divided by the pivots, for the
  // back substitution.
  std::vector<double> d(nx);
  const auto step = [&](double tau_next, bool crank_nicolson) {
    double* u = g.u.data();
    double* rhs = d.data();
    const double u_left = left(tau_next), u_right = right(tau_next);
    const double keep = crank_nicolson ? 1.0 - g.alpha : 1.0;
    const double mix = crank_nicolson ? beta : 0.0;
    double prev = 0.0;
    for (std::size_t i = 1; i < nx; ++i) {
      double b = keep * u[i] + mix * (u[i - 1] + u[i + 1]);
      if (i == 1) b += beta * u_left;
      if (i == nx - 1) b += beta * u_right;
      prev = b + multiplier[i] * prev;
      rhs[i] = prev * inv_pivot[i];
    }
    u[0] = u_left;
    u[nx] = u_right;
    double next = 0.0;
    for (std::size_t i = nx - 1; i >= 1; --i) {
      next = rhs[i] + back[i] * next;
      u[i] = next;
    }
  };
  for (std::size_t n = 1; n <= nt; ++n) {
    const double tau = static_cast<double>(n) * dtau;
    if (n <= smoothing_steps) {
      step(tau - 0.5 * dtau, false);
      step(tau, false);
    } else {
      step(tau, true);
    }
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

namespace internal {

// V(S, T) for payoff h on [S_min, S_max] from a heat-equation grid solver
// solve_grid(u0, left, right, x_min, x_max, tau_final) -> HeatGrid.
template <class SolveGrid, std::invocable<double> Payoff,
          std::invocable<double> VL, std::invocable<double> VR>
[[nodiscard]] double BsHeatOnGrid(SolveGrid&& solve_grid, Payoff&& h,
                                  VL&& v_left, VR&& v_right, double spot,
                                  double strike, double maturity, double sigma,
                                  double r, double q, double spot_min,
                                  double spot_max) {
  const auto c = MakeHeatTransform(sigma, r, q);
  const double x_min = std::log(spot_min / strike),
               x_max = std::log(spot_max / strike);
  const double tau_final = 0.5 * sigma * sigma * maturity;
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
  const HeatGrid g = solve_grid(u0, left, right, x_min, x_max, tau_final);
  const double x = std::log(spot / strike);
  return BsFromHeat(Interpolate(g.x, g.u, x), x, tau_final, strike, c);
}

// Calls and puts on [K e^(-L), K e^L], L = |ln(S/K)| + 6 sigma sqrt(T), from
// bs_heat(h, v_left, v_right, spot_min, spot_max) -> price.
template <class BsHeat>
[[nodiscard]] double EuropeanHeat(BsHeat&& bs_heat, OptionType type,
                                  double spot, double strike, double maturity,
                                  double sigma, double r, double q) {
  const double half_width =
      std::abs(std::log(spot / strike)) + 6.0 * sigma * std::sqrt(maturity);
  const double spot_min = strike * std::exp(-half_width),
               spot_max = strike * std::exp(half_width);
  const auto h = [&](double s) { return ch01::Payoff(type, s, strike); };
  const auto zero = [](double) { return 0.0; };
  if (type == OptionType::kCall)
    return bs_heat(
        h, zero,
        [&](double th) {
          return spot_max * std::exp(-q * th) - strike * std::exp(-r * th);
        },
        spot_min, spot_max);
  return bs_heat(
      h,
      [&](double th) {
        return strike * std::exp(-r * th) - spot_min * std::exp(-q * th);
      },
      zero, spot_min, spot_max);
}

}  // namespace internal

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
  const auto solve_grid = [&](const auto& u0, const auto& left,
                              const auto& right, double x_min, double x_max,
                              double tau_final) {
    if (nt == 0) {
      const double dx = (x_max - x_min) / static_cast<double>(nx);
      nt = static_cast<std::size_t>(std::ceil(tau_final / (0.45 * dx * dx)));
    }
    return SolveHeatExplicit(u0, left, right, x_min, x_max, tau_final, nx, nt);
  };
  return internal::BsHeatOnGrid(solve_grid, h, v_left, v_right, spot, strike,
                                maturity, sigma, r, q, spot_min, spot_max);
}

// The same price by Crank-Nicolson. nt = 0 takes nt = nx / 4 time steps (at
// least 8), which keeps the time error well below the space error.
template <std::invocable<double> Payoff, std::invocable<double> VL,
          std::invocable<double> VR>
[[nodiscard]] double BsHeatCrankNicolson(Payoff&& h, VL&& v_left, VR&& v_right,
                                         double spot, double strike,
                                         double maturity, double sigma,
                                         double r, double q, double spot_min,
                                         double spot_max, std::size_t nx,
                                         std::size_t nt = 0) {
  if (nt == 0) nt = std::max<std::size_t>(nx / 4, 8);
  const auto solve_grid = [&](const auto& u0, const auto& left,
                              const auto& right, double x_min, double x_max,
                              double tau_final) {
    return SolveHeatCrankNicolson(u0, left, right, x_min, x_max, tau_final, nx,
                                  nt);
  };
  return internal::BsHeatOnGrid(solve_grid, h, v_left, v_right, spot, strike,
                                maturity, sigma, r, q, spot_min, spot_max);
}

// Calls and puts, on [K e^(-L), K e^L] with L = |ln(S/K)| + 6 sigma sqrt(T).
[[nodiscard]] inline double EuropeanHeatFd(OptionType type, double spot,
                                           double strike, double maturity,
                                           double sigma, double r,
                                           double q = 0.0, std::size_t nx = 400,
                                           std::size_t nt = 0) {
  const auto bs_heat = [&](const auto& h, const auto& v_left,
                           const auto& v_right, double spot_min,
                           double spot_max) {
    return BsHeatFd(h, v_left, v_right, spot, strike, maturity, sigma, r, q,
                    spot_min, spot_max, nx, nt);
  };
  return internal::EuropeanHeat(bs_heat, type, spot, strike, maturity, sigma, r,
                                q);
}

[[nodiscard]] inline double EuropeanHeatCrankNicolson(
    OptionType type, double spot, double strike, double maturity, double sigma,
    double r, double q = 0.0, std::size_t nx = 400, std::size_t nt = 0) {
  const auto bs_heat = [&](const auto& h, const auto& v_left,
                           const auto& v_right, double spot_min,
                           double spot_max) {
    return BsHeatCrankNicolson(h, v_left, v_right, spot, strike, maturity,
                               sigma, r, q, spot_min, spot_max, nx, nt);
  };
  return internal::EuropeanHeat(bs_heat, type, spot, strike, maturity, sigma, r,
                                q);
}

}  // namespace p4mfe::ch08
