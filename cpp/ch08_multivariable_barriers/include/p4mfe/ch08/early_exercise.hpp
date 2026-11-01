#pragma once
// Chapter 8, section 8.10: optimality of early exercise.
//
// An American call on an asset paying no dividends should never be exercised
// early: before maturity its value is at least the European value, and
//   C >= S - K e^(-r tau) > S - K   for r > 0, tau > 0,
// so selling the option always beats exercising it (by at least
// K (1 - e^(-r tau))). Hence American call = European call when q = 0.
// With dividends (q > 0), or for puts, early exercise can be optimal and the
// American option is worth strictly more.
//
// The binomial tree below (Cox-Ross-Rubinstein) makes this concrete: it
// prices both styles and shows a zero early exercise premium for calls with
// q = 0 and a positive one otherwise.

#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch08 {

using ch01::OptionType;

enum class ExerciseStyle { European, American };

// --------------------------------------------------------------- 8.10 ------
// CRR tree with n steps: u = e^(sigma sqrt(dt)), d = 1/u,
// p = (e^((r - q) dt) - d) / (u - d).
[[nodiscard]] inline double binomial_price(OptionType type, double S, double K, double T, double sigma, double r,
                                           double q, std::size_t n, ExerciseStyle style) {
    if (n == 0 || !(T > 0.0 && sigma > 0.0)) throw std::invalid_argument("binomial_price: need n >= 1, T, sigma > 0");
    const double dt = T / static_cast<double>(n);
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double p = (std::exp((r - q) * dt) - d) / (u - d);
    if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("binomial_price: step too large (p not in (0, 1))");
    const double disc = std::exp(-r * dt);
    std::vector<double> V(n + 1);
    for (std::size_t j = 0; j <= n; ++j)
        V[j] = ch01::payoff(type, S * std::pow(u, 2.0 * static_cast<double>(j) - static_cast<double>(n)), K);
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = 0; j <= i; ++j) {
            V[j] = disc * (p * V[j + 1] + (1.0 - p) * V[j]);
            if (style == ExerciseStyle::American) {
                const double Sij = S * std::pow(u, 2.0 * static_cast<double>(j) - static_cast<double>(i));
                V[j] = std::max(V[j], ch01::payoff(type, Sij, K));
            }
        }
    }
    return V[0];
}

// American minus European value on the same tree.
[[nodiscard]] inline double early_exercise_premium(OptionType type, double S, double K, double T, double sigma,
                                                   double r, double q, std::size_t n = 500) {
    return binomial_price(type, S, K, T, sigma, r, q, n, ExerciseStyle::American) -
           binomial_price(type, S, K, T, sigma, r, q, n, ExerciseStyle::European);
}

// What is lost by exercising a call now instead of selling it (q = 0):
// C - (S - K) >= K (1 - e^(-r tau)) > 0.
[[nodiscard]] inline double call_exercise_loss(double S, double K, double tau, double sigma, double r) {
    return ch03::bs_call(S, K, tau, sigma, r) - (S - K);
}
[[nodiscard]] inline double call_exercise_loss_bound(double K, double tau, double r) noexcept {
    return -K * std::expm1(-r * tau);
}

}  // namespace p4mfe::ch08
