#pragma once
// Exercises 9-10 (section 8.12): the change of variables from the
// Black-Scholes PDE to the heat equation, its initial condition for calls and
// puts, and the system that determines a and b.

#include <algorithm>
#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>

namespace p4mfe::ch08::ex {

// Notation of exercises 9-11:
//   V(S, t) = exp(-a x - b tau) u(x, tau),
//   x = ln(S/K),   tau = (T - t) sigma^2 / 2,
//   a = (r - q)/sigma^2 - 1/2,   b = ((r - q)/sigma^2 + 1/2)^2 + 2q/sigma^2.
// (The chapter library's heat_transform writes V = K e^(a' x + b' tau) u' with
// a' = -a, b' = -b, so u = K u'.)
struct Ex09Coefficients {
  double a;
  double b;
};
[[nodiscard]] inline Ex09Coefficients Ex09ComputeCoefficients(double sigma,
                                                              double r,
                                                              double q = 0.0) {
  const double k = (r - q) / (sigma * sigma);
  return {k - 0.5, (k + 0.5) * (k + 0.5) + 2.0 * q / (sigma * sigma)};
}

// ------------------------------------------------------------ Exercise 9 ---
// (i) Show that V(S, T) = max(S - K, 0) becomes u(x, 0) = K e^(ax) max(e^x - 1,
// 0).
//   t = T corresponds to tau = 0, and S = K e^x. From the change of variables,
//     u(x, tau) = exp(a x + b tau) V(S, t),
//   so at tau = 0
//     u(x, 0) = e^(ax) V(K e^x, T) = e^(ax) max(K e^x - K, 0) = K e^(ax)
//     max(e^x - 1, 0),
//   using K > 0 to pull K out of the max.
// (ii) For the put, V(S, T) = max(K - S, 0) gives in the same way
//     u(x, 0) = e^(ax) max(K - K e^x, 0) = K e^(ax) max(1 - e^x, 0).
//
// Check: u(x, tau) = e^(ax + b tau) V(K e^x, t) with T - t = 2 tau/sigma^2 must
// solve the heat equation u_tau = u_xx (exercise 10 shows why), and must reduce
// to the formulas above at tau = 0.
[[nodiscard]] inline double Ex09CallU0(double x, double strike, double a) {
  return strike * std::exp(a * x) * std::max(std::exp(x) - 1.0, 0.0);
}
[[nodiscard]] inline double Ex09PutU0(double x, double strike, double a) {
  return strike * std::exp(a * x) * std::max(1.0 - std::exp(x), 0.0);
}

// u(x, tau) from the Black-Scholes price; tau = 0 uses the payoff.
[[nodiscard]] inline double Ex09U(ch03::OptionType type, double x, double tau,
                                  double strike, double sigma, double r,
                                  double q = 0.0) {
  const auto c = Ex09ComputeCoefficients(sigma, r, q);
  const double spot = strike * std::exp(x);
  const double ttm = 2.0 * tau / (sigma * sigma);  // T - t
  const double value = ttm > 0.0
                           ? ch03::BsPrice(type, spot, strike, ttm, sigma, r, q)
                           : ch01::Payoff(type, spot, strike);
  return std::exp(c.a * x + c.b * tau) * value;
}

// u_tau - u_xx by central differences (step h in x and in tau).
[[nodiscard]] inline double Ex09HeatResidual(ch03::OptionType type, double x,
                                             double tau, double strike,
                                             double sigma, double r,
                                             double q = 0.0, double h = 1e-4) {
  const auto u = [&](double xx, double tt) {
    return Ex09U(type, xx, tt, strike, sigma, r, q);
  };
  const double u_tau = (u(x, tau + h) - u(x, tau - h)) / (2.0 * h);
  const double u_xx =
      (u(x + h, tau) - 2.0 * u(x, tau) + u(x - h, tau)) / (h * h);
  return u_tau - u_xx;
}

// ----------------------------------------------------------- Exercise 10 ---
// Show that the solution of the system
//   2a + 1 - 2(r - q)/sigma^2 = 0,
//   b + a^2 + a (1 - 2(r - q)/sigma^2) - 2r/sigma^2 = 0
// is a = (r - q)/sigma^2 - 1/2 and b = ((r - q)/sigma^2 + 1/2)^2.
//
// Where the system comes from. With x = ln(S/K), tau = (T - t) sigma^2/2 the
// Black-Scholes PDE V_t + sigma^2 S^2 V_SS/2 + (r - q) S V_S - r V = 0 becomes
// (S V_S = V_x, S^2 V_SS = V_xx - V_x, V_t = -sigma^2 V_tau / 2)
//   V_tau = V_xx + (2(r - q)/sigma^2 - 1) V_x - (2r/sigma^2) V.
// Substituting V = e^(-ax - b tau) u, dividing by e^(-ax - b tau):
//   u_tau - b u = u_xx - 2a u_x + a^2 u + (2(r-q)/sigma^2 - 1)(u_x - a u) -
//   (2r/sigma^2) u.
// u solves u_tau = u_xx exactly when the u_x and u coefficients vanish:
//   u_x:  -2a + 2(r - q)/sigma^2 - 1 = 0,
//   u:    b + a^2 + a (1 - 2(r - q)/sigma^2) - 2r/sigma^2 = 0,
// which is the system above.
//
// Solution. The first equation gives a = (r - q)/sigma^2 - 1/2. Then
// 1 - 2(r - q)/sigma^2 = -2a, and the second equation becomes
//   b + a^2 - 2a^2 - 2r/sigma^2 = 0   =>   b = a^2 + 2r/sigma^2.
// Since a + 1 = (r - q)/sigma^2 + 1/2 and (a + 1)^2 = a^2 + 2a + 1 = a^2 + 2(r
// - q)/sigma^2,
//   b = (a + 1)^2 + 2q/sigma^2 = ((r - q)/sigma^2 + 1/2)^2 + 2q/sigma^2.
// This is the b of exercise 9. The value printed in exercise 10,
// ((r - q)/sigma^2 + 1/2)^2, is the solution only when q = 0; for q > 0 it
// leaves a residual of -2q/sigma^2 in the second equation.
struct Ex10Residuals {
  double first;
  double second;
};
[[nodiscard]] inline Ex10Residuals Ex10ComputeResiduals(double a, double b,
                                                        double sigma, double r,
                                                        double q = 0.0) {
  const double s2 = sigma * sigma;
  return {2.0 * a + 1.0 - 2.0 * (r - q) / s2,
          b + a * a + a * (1.0 - 2.0 * (r - q) / s2) - 2.0 * r / s2};
}
[[nodiscard]] inline Ex09Coefficients Ex10Solve(double sigma, double r,
                                                double q = 0.0) {
  const double s2 = sigma * sigma;
  const double a = (r - q) / s2 - 0.5;
  return {a, a * a + 2.0 * r / s2};
}
// The b printed in the exercise (correct for q = 0).
[[nodiscard]] inline double Ex10PrintedB(double sigma, double r,
                                         double q = 0.0) {
  const double k = (r - q) / (sigma * sigma);
  return (k + 0.5) * (k + 0.5);
}

}  // namespace p4mfe::ch08::ex
