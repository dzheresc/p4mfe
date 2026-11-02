#pragma once
// Exercise 4 (section 5.7): implied volatility of an at-the-money call by
// bisection, secant and Newton's method, and the ATM approximation (6.104).

#include <cmath>
#include <numbers>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch05/solvers.hpp>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 4 ---
// A three months at-the-money call on an asset with spot price 30 paying
// dividends continuously at 2% is worth 2.5; the risk free rate is 6%.
// (i) Compute the implied volatility with six decimal digits accuracy, using
//     bisection on [0.0001, 1], the secant method with initial guess 0.5, and
//     Newton's method with initial guess 0.5.
// (ii) With sigma_imp from Newton's method, use (6.104),
//        sigma_approx = sqrt(2 pi)/(S sqrt(T)) (C - (r - q) T S / 2) / (1 - (r
//        + q) T / 2),
//      and compute the relative error |sigma_approx - sigma_imp| / sigma_imp.
//
// f(sigma) = C_BS(sigma) - 2.5, f' = vega. "Six decimal digits" is the book's
// stopping rule with tol_consec = 1e-6 (tol_int for bisection) and
// tol_approx = 1e-9. The secant method needs a second starting point; as in
// the book's example after Table 5.3 we take x_{-1} = x_0 - 0.01 = 0.49.
//   bisection: 0.397049 after 30 halvings (|f| <= 1e-9 needs more halvings
//              than the interval length 1e-6 alone, which needs 20);
//   secant:    0.397049 after 3 steps;
//   Newton:    0.397049 after 3 steps (0.5, 0.396916, 0.397049, 0.397049).
// So sigma_imp = 0.397049 (39.7049%).
// (ii) sigma_approx = 0.396672, relative error 9.49e-4 (under 0.1%): the ATM
//      approximation is excellent for short maturities.
struct Ex04Market {
  double S = 30.0, K = 30.0, T = 0.25, r = 0.06, q = 0.02, C = 2.5;
};
inline constexpr Tolerances ex04_tolerances{1e-9, 1e-6, 200};

[[nodiscard]] inline double ex04_f(double sigma, const Ex04Market& m = {}) {
  return ch03::bs_call(m.S, m.K, m.T, sigma, m.r, m.q) - m.C;
}
[[nodiscard]] inline double ex04_vega(double sigma, const Ex04Market& m = {}) {
  return ch03::bs_greeks(ch01::OptionType::kCall, m.S, m.K, m.T, sigma, m.r,
                         m.q)
      .vega;
}

[[nodiscard]] inline RootResult ex04_bisection(const Ex04Market& m = {}) {
  return bisection([&](double s) { return ex04_f(s, m); }, 0.0001, 1.0,
                   ex04_tolerances);
}
[[nodiscard]] inline RootResult ex04_secant(const Ex04Market& m = {}) {
  return secant([&](double s) { return ex04_f(s, m); }, 0.49, 0.5,
                ex04_tolerances);
}
[[nodiscard]] inline RootResult ex04_newton(const Ex04Market& m = {}) {
  return newton([&](double s) { return ex04_f(s, m); },
                [&](double s) { return ex04_vega(s, m); }, 0.5,
                ex04_tolerances);
}

// (6.104)
[[nodiscard]] inline double ex04_sigma_approx(const Ex04Market& m = {}) {
  return std::sqrt(2.0 * std::numbers::pi) / (m.S * std::sqrt(m.T)) *
         (m.C - 0.5 * (m.r - m.q) * m.T * m.S) /
         (1.0 - 0.5 * (m.r + m.q) * m.T);
}
[[nodiscard]] inline double ex04_relative_error(const Ex04Market& m = {}) {
  const double s = ex04_newton(m).x;
  return std::abs(ex04_sigma_approx(m) - s) / s;
}

}  // namespace p4mfe::ch05::ex
