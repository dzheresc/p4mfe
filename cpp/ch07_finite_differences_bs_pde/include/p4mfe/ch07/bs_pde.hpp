#pragma once
// Chapter 7, section 7.4: the Black-Scholes PDE.
//
// The value V(S, t) of any derivative on an asset following a lognormal
// model with volatility sigma and dividend yield q satisfies
//   dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0,
// with the payoff as terminal condition V(S, T). The PDE does not involve the
// drift mu of the asset.
//
// 7.4.1 Financial interpretation. A portfolio long one derivative and short
// Delta = dV/dS units of the asset has no dS term over dt; with no risk it
// must earn the risk-free rate, which is the PDE. `delta_hedge_pnl` makes this
// concrete: sell the option at its Black-Scholes price, hold Delta shares,
// finance with cash at r and rebalance n times. The final P&L has mean ~0 and
// standard deviation O(1/sqrt(n)), whatever mu is.
//
// 7.4.2 The PDE and the Greeks. In Greek letters,
//   Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0,
// and since the PDE is linear it holds for any portfolio of derivatives on
// the same asset. For a Delta-neutral portfolio, Theta = r V - 1/2 sigma^2
// S^2 Gamma: long Gamma pays for itself through time decay. Chapter 3's
// bs_pde_residual checks the closed-form Greeks; chapter 6's theta_from_pde
// solves the equation for Theta. Here the PDE is checked for any V(S, t) by
// finite differences.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch07 {

using ch03::Greeks;
using ch03::OptionType;

// ---------------------------------------------------------------- 7.4 ------
// Left-hand side of the PDE for V(S, t), all derivatives by central
// differences (chapter 1's step choices). Zero up to O(h^2) iff V solves it.
template <std::invocable<double, double> V>
[[nodiscard]] double bs_pde_residual_fd(V&& v, double S, double t, double sigma,
                                        double r, double q = 0.0) {
  const auto in_S = [&](double s) { return v(s, t); };
  const auto in_t = [&](double u) { return v(S, u); };
  const double theta = ch01::Derivative(in_t, t);
  const double delta = ch01::Derivative(in_S, S);
  const double gamma = ch01::SecondDerivative(in_S, S);
  return theta + 0.5 * sigma * sigma * S * S * gamma + (r - q) * S * delta -
         r * v(S, t);
}

// Elementary solutions V = S^a e^(c t), with c = r - a (r - q) - 1/2 sigma^2 a
// (a - 1).
//   a = 0: cash, e^(r t).   a = 1: the asset with dividends reinvested, S e^(q
//   t).
// European prices are superpositions of such solutions in disguise.
[[nodiscard]] inline double power_solution_rate(double a, double sigma,
                                                double r,
                                                double q = 0.0) noexcept {
  return r - a * (r - q) - 0.5 * sigma * sigma * a * (a - 1.0);
}
[[nodiscard]] inline double power_solution(double a, double S, double t,
                                           double sigma, double r,
                                           double q = 0.0) noexcept {
  return std::pow(S, a) * std::exp(power_solution_rate(a, sigma, r, q) * t);
}

// Value at time t of a long forward with delivery price K maturing at T:
// S e^(-q (T - t)) - K e^(-r (T - t)). Solves the PDE (it is linear in S).
[[nodiscard]] inline double forward_contract_value(double S, double t, double K,
                                                   double T, double r,
                                                   double q = 0.0) noexcept {
  return S * std::exp(-q * (T - t)) - K * std::exp(-r * (T - t));
}

// -------------------------------------------------------------- 7.4.2 ------
// The PDE in terms of a value and its Greeks (any portfolio on one asset).
[[nodiscard]] constexpr double pde_residual(double V, const Greeks& g, double S,
                                            double sigma, double r,
                                            double q = 0.0) noexcept {
  return g.theta + 0.5 * sigma * sigma * S * S * g.gamma +
         (r - q) * S * g.delta - r * V;
}

// Greeks of a portfolio sum(w_i * position_i) are the weighted sums.
[[nodiscard]] inline Greeks combine_greeks(const std::vector<double>& w,
                                           const std::vector<Greeks>& gs) {
  if (w.size() != gs.size())
    throw std::invalid_argument("combine_greeks: size mismatch");
  Greeks out{};
  for (std::size_t i = 0; i < w.size(); ++i) {
    out.delta += w[i] * gs[i].delta;
    out.gamma += w[i] * gs[i].gamma;
    out.vega += w[i] * gs[i].vega;
    out.theta += w[i] * gs[i].theta;
    out.rho += w[i] * gs[i].rho;
  }
  return out;
}

// -------------------------------------------------------------- 7.4.1 ------
// Lognormal path S_0, S_h, ..., S_T on n equal steps with real-world drift
// mu (chapter 4's price_from_normal for each step).
template <std::uniform_random_bit_generator G>
[[nodiscard]] std::vector<double> gbm_path(double S0, double T, double sigma,
                                           double mu, double q, std::size_t n,
                                           G& rng) {
  if (n == 0) throw std::invalid_argument("gbm_path: need n >= 1");
  std::normal_distribution<double> z;
  const double dt = T / static_cast<double>(n);
  std::vector<double> path(n + 1);
  path[0] = S0;
  for (std::size_t k = 0; k < n; ++k)
    path[k + 1] = ch04::price_from_normal(z(rng), path[k], dt, sigma, mu, q);
  return path;
}

// Sell one option at its Black-Scholes price, hold Delta shares, keep the
// rest in cash at r, receive dividends q S dt on the shares, and rebalance at
// each point of `path` (equally spaced on [0, T]). Returns the final P&L:
// cash + shares * S_T - payoff.
[[nodiscard]] inline double delta_hedge_pnl(OptionType type, double K, double T,
                                            double sigma, double r, double q,
                                            const std::vector<double>& path) {
  if (path.size() < 2)
    throw std::invalid_argument("delta_hedge_pnl: need at least two prices");
  const std::size_t n = path.size() - 1;
  const double dt = T / static_cast<double>(n);
  const double growth = std::exp(r * dt);
  const double carry =
      std::exp(q * dt) -
      1.0;  // dividends on one share over dt, per unit of price
  double shares = ch03::BsGreeks(type, path[0], K, T, sigma, r, q).delta;
  double cash =
      ch03::BsPrice(type, path[0], K, T, sigma, r, q) - shares * path[0];
  for (std::size_t k = 1; k <= n; ++k) {
    const double S = path[k];
    cash = cash * growth + shares * path[k - 1] * carry;
    if (k == n) break;
    const double tau = T - static_cast<double>(k) * dt;
    const double new_shares =
        ch03::BsGreeks(type, S, K, tau, sigma, r, q).delta;
    cash -= (new_shares - shares) * S;
    shares = new_shares;
  }
  return cash + shares * path[n] - ch01::Payoff(type, path[n], K);
}

struct HedgeStats {
  double mean;     // average P&L
  double std_dev;  // standard deviation of P&L
};

// P&L statistics of `delta_hedge_pnl` over n_paths paths with drift mu.
[[nodiscard]] inline HedgeStats delta_hedge_stats(
    OptionType type, double S0, double K, double T, double sigma, double r,
    double q, double mu, std::size_t n_steps, std::size_t n_paths,
    std::uint64_t seed = 42) {
  if (n_paths < 2)
    throw std::invalid_argument("delta_hedge_stats: need n_paths >= 2");
  std::mt19937_64 rng(seed);
  double sum = 0.0, sum2 = 0.0;
  for (std::size_t i = 0; i < n_paths; ++i) {
    const double pnl = delta_hedge_pnl(
        type, K, T, sigma, r, q, gbm_path(S0, T, sigma, mu, q, n_steps, rng));
    sum += pnl;
    sum2 += pnl * pnl;
  }
  const double m = sum / static_cast<double>(n_paths);
  const double var = (sum2 - static_cast<double>(n_paths) * m * m) /
                     static_cast<double>(n_paths - 1);
  return {m, std::sqrt(std::max(var, 0.0))};
}

}  // namespace p4mfe::ch07
