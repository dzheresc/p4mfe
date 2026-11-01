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
[[nodiscard]] inline double percentage_return(double S1, double S2, double dividend = 0.0) {
    if (!(S1 > 0.0)) throw std::invalid_argument("percentage_return: S1 must be positive");
    return (S2 + dividend - S1) / S1;
}

[[nodiscard]] inline double log_return(double S1, double S2, double dividend = 0.0) {
    if (!(S1 > 0.0 && S2 + dividend > 0.0)) throw std::invalid_argument("log_return: prices must be positive");
    return std::log((S2 + dividend) / S1);
}

[[nodiscard]] inline double log_from_percentage(double R) { return std::log1p(R); }
[[nodiscard]] inline double percentage_from_log(double r) { return std::expm1(r); }

// Taylor polynomial of ln(1 + R) of degree `order` (1: R, 2: R - R^2/2, ...).
[[nodiscard]] inline double log_return_taylor(double R, unsigned order = 2) noexcept {
    double power = 1.0, sum = 0.0;
    for (unsigned k = 1; k <= order; ++k) {
        power *= -R;
        sum -= power / k;
    }
    return sum;
}

// Period returns of a price series S_0, ..., S_n (n returns).
[[nodiscard]] inline std::vector<double> percentage_returns(std::span<const double> prices) {
    std::vector<double> R;
    for (std::size_t i = 1; i < prices.size(); ++i) R.push_back(percentage_return(prices[i - 1], prices[i]));
    return R;
}
[[nodiscard]] inline std::vector<double> log_returns(std::span<const double> prices) {
    std::vector<double> r;
    for (std::size_t i = 1; i < prices.size(); ++i) r.push_back(log_return(prices[i - 1], prices[i]));
    return r;
}

// Return over the whole horizon from the period returns.
[[nodiscard]] inline double compound_percentage_returns(std::span<const double> R) noexcept {
    double g = 1.0;
    for (double x : R) g *= 1.0 + x;
    return g - 1.0;
}
[[nodiscard]] inline double sum_log_returns(std::span<const double> r) noexcept {
    double s = 0.0;
    for (double x : r) s += x;
    return s;
}

// -------------------------------------------------------------- 6.4.1 ------
// Weights w_i = n_i S_i / sum_j n_j S_j.
[[nodiscard]] inline std::vector<double> portfolio_weights(std::span<const double> units,
                                                           std::span<const double> prices) {
    if (units.size() != prices.size()) throw std::invalid_argument("portfolio_weights: size mismatch");
    double V = 0.0;
    for (std::size_t i = 0; i < units.size(); ++i) V += units[i] * prices[i];
    if (V == 0.0) throw std::invalid_argument("portfolio_weights: portfolio value is zero");
    std::vector<double> w(units.size());
    for (std::size_t i = 0; i < units.size(); ++i) w[i] = units[i] * prices[i] / V;
    return w;
}

namespace detail {
inline void check_same_size(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::invalid_argument("weights and returns must have the same size");
}
}  // namespace detail

// R_V = sum_i w_i R_i (exact).
[[nodiscard]] inline double portfolio_percentage_return(std::span<const double> w, std::span<const double> R) {
    detail::check_same_size(w, R);
    double s = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) s += w[i] * R[i];
    return s;
}

// r_V = ln(sum_i w_i e^(r_i)) (exact).
[[nodiscard]] inline double portfolio_log_return(std::span<const double> w, std::span<const double> r) {
    detail::check_same_size(w, r);
    double s = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) s += w[i] * std::exp(r[i]);
    return std::log(s);
}

// r_V ~ sum_i w_i r_i (first-order Taylor approximation).
[[nodiscard]] inline double portfolio_log_return_approx(std::span<const double> w, std::span<const double> r) {
    return portfolio_percentage_return(w, r);
}

}  // namespace p4mfe::ch06
