#pragma once
// Chapter 6, sections 6.8-6.8.3: at-the-money approximations of the
// Black-Scholes formulas.
//
// 6.8.1 For an ATM option (K = S) with small r, q and sigma sqrt(T),
//   C ~ sigma S sqrt(T / (2 pi)) (1 - (r + q) T / 2) + (r - q) T S / 2,
//   P ~ sigma S sqrt(T / (2 pi)) (1 - (r + q) T / 2) - (r - q) T S / 2,
// and for r = q = 0 both reduce to
//   C ~ P ~ sigma S sqrt(T / (2 pi)) ~ 0.4 sigma S sqrt(T).
// Inverted, this is a quick ATM implied volatility estimate,
//   sigma ~ sqrt(2 pi) C / (S sqrt(T)).
//
// 6.8.2 Derivation: d1 and d2 are small at the money, so replace N by its
// first-order Taylor polynomial N(x) ~ 1/2 + x / sqrt(2 pi), and e^(-qT),
// e^(-rT) by 1 - qT, 1 - rT. With d1 - d2 = sigma sqrt(T),
//   C ~ S [ (r - q) T / 2 + (1 - (r + q) T / 2) sigma sqrt(T) / sqrt(2 pi) ]
// after dropping terms quadratic in r, q (the put follows from Put-Call
// parity). `atm_call_linearized` keeps the step before that last drop.
//
// 6.8.3 Precision (r = q = 0): with x = sigma sqrt(T) / 2 the exact price is
//   C = S (N(x) - N(-x)) = S sqrt(2/pi) int_0^x e^(-t^2/2) dt,
// and e^(-t^2/2) >= 1 - t^2/2 gives
//   0 <= (C_approx - C) / C_approx <= x^2 / 6 = sigma^2 T / 24.
// The approximation always overestimates, and its relative error is about
// sigma^2 T / 24: 0.1% for sigma = 0.2 and T = 0.5 (6 months).

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>
#include <stdexcept>

namespace p4mfe::ch06 {

namespace detail {
// 1 / sqrt(2 pi)
inline constexpr double inv_sqrt_2pi =
    std::numbers::inv_sqrtpi / std::numbers::sqrt2;
}  // namespace detail

// -------------------------------------------------------------- 6.8.1 ------
// r = q = 0: C ~ P ~ sigma S sqrt(T / (2 pi)).
[[nodiscard]] inline double atm_price_approx(double S, double T,
                                             double sigma) noexcept {
  return sigma * S * std::sqrt(T) * detail::inv_sqrt_2pi;
}

// With rates: the formulas above (argument order as in chapter 3, K = S).
[[nodiscard]] inline double atm_call_approx(double S, double T, double sigma,
                                            double r, double q = 0.0) noexcept {
  return atm_price_approx(S, T, sigma) * (1.0 - 0.5 * (r + q) * T) +
         0.5 * (r - q) * T * S;
}
[[nodiscard]] inline double atm_put_approx(double S, double T, double sigma,
                                           double r, double q = 0.0) noexcept {
  return atm_price_approx(S, T, sigma) * (1.0 - 0.5 * (r + q) * T) -
         0.5 * (r - q) * T * S;
}

// sigma ~ sqrt(2 pi) V / (S sqrt(T)) from an ATM price V (r = q = 0).
[[nodiscard]] inline double atm_implied_vol_approx(double price, double S,
                                                   double T) {
  if (!(S > 0.0 && T > 0.0))
    throw std::invalid_argument("atm_implied_vol_approx: need S, T > 0");
  return price / (S * std::sqrt(T) * detail::inv_sqrt_2pi);
}

// ATM Greeks to first order in sigma sqrt(T), r = q = 0 (d1 = -d2 = sigma
// sqrt(T)/2):
//   Delta_C ~ 1/2 + sigma sqrt(T) / (2 sqrt(2 pi)),  Delta_P = Delta_C - 1,
//   Gamma ~ 1 / (S sigma sqrt(2 pi T)),  Vega ~ S sqrt(T / (2 pi)),
//   Theta ~ -S sigma / (2 sqrt(2 pi T)),
//   Rho_C ~ S T (1/2 - sigma sqrt(T) / (2 sqrt(2 pi))),  Rho_P = Rho_C - S T.
[[nodiscard]] inline ch03::Greeks atm_greeks_approx(ch03::OptionType type,
                                                    double S, double T,
                                                    double sigma) noexcept {
  const double sT = sigma * std::sqrt(T);
  const double half_d = 0.5 * sT * detail::inv_sqrt_2pi;
  ch03::Greeks g{};
  g.delta = 0.5 + half_d;
  g.gamma = detail::inv_sqrt_2pi / (S * sT);
  g.vega = S * std::sqrt(T) * detail::inv_sqrt_2pi;
  g.theta = -0.5 * S * sigma * detail::inv_sqrt_2pi / std::sqrt(T);
  g.rho = S * T * (0.5 - half_d);
  if (type == ch03::OptionType::Put) {
    g.delta -= 1.0;
    g.rho -= S * T;
  }
  return g;
}

// -------------------------------------------------------------- 6.8.2 ------
// First-order Taylor polynomial of N around 0.
[[nodiscard]] constexpr double norm_cdf_linear(double x) noexcept {
  return 0.5 + x * detail::inv_sqrt_2pi;
}

// S e^(-qT) N_lin(d1) - S e^(-rT) N_lin(d2): Black-Scholes at K = S with N
// linearized, before the exponentials are expanded.
[[nodiscard]] inline double atm_call_linearized(double S, double T,
                                                double sigma, double r,
                                                double q = 0.0) {
  const auto [d1, d2] = ch03::bs_d1_d2(S, S, T, sigma, r, q);
  return S * std::exp(-q * T) * norm_cdf_linear(d1) -
         S * std::exp(-r * T) * norm_cdf_linear(d2);
}

// -------------------------------------------------------------- 6.8.3 ------
// Exact ATM price for r = q = 0: S (2 N(sigma sqrt(T) / 2) - 1); calls and puts
// agree.
[[nodiscard]] inline double atm_price_exact(double S, double T,
                                            double sigma) noexcept {
  const double x = 0.5 * sigma * std::sqrt(T);
  return S * (norm_cdf(x) - norm_cdf(-x));
}

// (C_approx - C) / C_approx, exact, for r = q = 0 (independent of S).
[[nodiscard]] inline double atm_relative_error(double T,
                                               double sigma) noexcept {
  const double a = atm_price_approx(1.0, T, sigma);
  return (a - atm_price_exact(1.0, T, sigma)) / a;
}

// Its bound and leading-order size, sigma^2 T / 24.
[[nodiscard]] constexpr double atm_relative_error_bound(double T,
                                                        double sigma) noexcept {
  return sigma * sigma * T / 24.0;
}

}  // namespace p4mfe::ch06
