#pragma once
// Chapter 6, sections 6.4-6.4.1: percentage and log returns, asset and
// portfolio returns.
//
// Between times t1 < t2, with an optional dividend D paid in (t1, t2],
//   percentage return  R = (S2 + D - S1) / S1,
//   log return         r = ln((S2 + D) / S1) = ln(1 + R).
// Taylor's formula for ln(1 + R) around 0 gives
//   r = R - R^2/2 + R^3/3 - ...,
// so r ~ R for small R, and r < R always (ln(1+x) < x for x != 0).
//
// Over consecutive periods log returns add, percentage returns compound:
//   r(t0, tn) = sum_i r_i,   1 + R(t0, tn) = prod_i (1 + R_i).
//
// Portfolio (6.4.1): n_i units of asset i, value V = sum_i n_i S_i, weights
// w_i = n_i S_i / V (sum to 1). The portfolio percentage return is exactly the
// weighted average of the asset percentage returns, R_V = sum_i w_i R_i; for
// log returns this holds only approximately,
//   r_V = ln(sum_i w_i e^(r_i)) ~ sum_i w_i r_i.

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch06 {

// ---------------------------------------------------------------- 6.4 ------
[[nodiscard]] inline double PercentageReturn(double spot1, double spot2,
                                             double dividend = 0.0) {
  if (!(spot1 > 0.0))
    throw std::invalid_argument("percentage_return: S1 must be positive");
  return (spot2 + dividend - spot1) / spot1;
}

[[nodiscard]] inline double LogReturn(double spot1, double spot2,
                                      double dividend = 0.0) {
  if (!(spot1 > 0.0 && spot2 + dividend > 0.0))
    throw std::invalid_argument("log_return: prices must be positive");
  return std::log((spot2 + dividend) / spot1);
}

[[nodiscard]] inline double LogFromPercentage(double pct_return) {
  return std::log1p(pct_return);
}
[[nodiscard]] inline double PercentageFromLog(double r) {
  return std::expm1(r);
}

// Taylor polynomial of ln(1 + R) of degree `order` (1: R, 2: R - R^2/2, ...).
[[nodiscard]] inline double LogReturnTaylor(double pct_return,
                                            unsigned order = 2) noexcept {
  double power = 1.0, sum = 0.0;
  for (unsigned k = 1; k <= order; ++k) {
    power *= -pct_return;
    sum -= power / k;
  }
  return sum;
}

// Period returns of a price series S_0, ..., S_n (n returns).
[[nodiscard]] inline std::vector<double> PercentageReturns(
    std::span<const double> prices) {
  std::vector<double> pct_returns;
  for (std::size_t i = 1; i < prices.size(); ++i)
    pct_returns.push_back(PercentageReturn(prices[i - 1], prices[i]));
  return pct_returns;
}
[[nodiscard]] inline std::vector<double> LogReturns(
    std::span<const double> prices) {
  std::vector<double> r;
  for (std::size_t i = 1; i < prices.size(); ++i)
    r.push_back(LogReturn(prices[i - 1], prices[i]));
  return r;
}

// Return over the whole horizon from the period returns.
[[nodiscard]] inline double CompoundPercentageReturns(
    std::span<const double> pct_returns) noexcept {
  double g = 1.0;
  for (double x : pct_returns) g *= 1.0 + x;
  return g - 1.0;
}
[[nodiscard]] inline double SumLogReturns(std::span<const double> r) noexcept {
  double s = 0.0;
  for (double x : r) s += x;
  return s;
}

// -------------------------------------------------------------- 6.4.1 ------
// Weights w_i = n_i S_i / sum_j n_j S_j.
[[nodiscard]] inline std::vector<double> PortfolioWeights(
    std::span<const double> units, std::span<const double> prices) {
  if (units.size() != prices.size())
    throw std::invalid_argument("portfolio_weights: size mismatch");
  double total_value = 0.0;
  for (std::size_t i = 0; i < units.size(); ++i)
    total_value += units[i] * prices[i];
  if (total_value == 0.0)
    throw std::invalid_argument("portfolio_weights: portfolio value is zero");
  std::vector<double> w(units.size());
  for (std::size_t i = 0; i < units.size(); ++i)
    w[i] = units[i] * prices[i] / total_value;
  return w;
}

namespace detail {
inline void CheckSameSize(std::span<const double> a,
                          std::span<const double> b) {
  if (a.size() != b.size())
    throw std::invalid_argument("weights and returns must have the same size");
}
}  // namespace detail

// R_V = sum_i w_i R_i (exact).
[[nodiscard]] inline double PortfolioPercentageReturn(
    std::span<const double> w, std::span<const double> pct_returns) {
  detail::CheckSameSize(w, pct_returns);
  double s = 0.0;
  for (std::size_t i = 0; i < w.size(); ++i) s += w[i] * pct_returns[i];
  return s;
}

// r_V = ln(sum_i w_i e^(r_i)) (exact).
[[nodiscard]] inline double PortfolioLogReturn(std::span<const double> w,
                                               std::span<const double> r) {
  detail::CheckSameSize(w, r);
  double s = 0.0;
  for (std::size_t i = 0; i < w.size(); ++i) s += w[i] * std::exp(r[i]);
  return std::log(s);
}

// r_V ~ sum_i w_i r_i (first-order Taylor approximation).
[[nodiscard]] inline double PortfolioLogReturnApprox(
    std::span<const double> w, std::span<const double> r) {
  return PortfolioPercentageReturn(w, r);
}

}  // namespace p4mfe::ch06
