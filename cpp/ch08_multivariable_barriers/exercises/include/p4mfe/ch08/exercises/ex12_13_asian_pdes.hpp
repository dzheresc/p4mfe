#pragma once
// Exercises 12-13 (section 8.12): reducing the PDEs of geometric and
// arithmetic Asian options to two variables.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 12 ---
// Assume V(S, I, t) satisfies
//   V_t + ln(S) V_I + sigma^2 S^2 V_SS / 2 + r S V_S - r V = 0.     (8.52)
// With V(S, I, t) = F(y, t), y = (I + (T - t) ln S)/T, show that
//   F_t + sigma^2 (T - t)^2/(2 T^2) F_yy + (r - sigma^2/2) (T - t)/T F_y - r F
//   = 0.
//
// Proof. y_t = -ln(S)/T, y_I = 1/T, y_S = (T - t)/(T S), y_SS = -(T - t)/(T
// S^2). By the chain rule (t appears in V both directly and through y):
//   V_t  = F_t + F_y y_t = F_t - ln(S) F_y / T,
//   V_I  = F_y / T,
//   V_S  = F_y (T - t)/(T S),
//   V_SS = F_yy (T - t)^2/(T^2 S^2) - F_y (T - t)/(T S^2).
// Substituting in (8.52):
//   F_t - ln(S) F_y/T + ln(S) F_y/T
//     + sigma^2/2 [F_yy (T - t)^2/T^2 - F_y (T - t)/T]
//     + r F_y (T - t)/T - r F
//   = F_t + sigma^2 (T - t)^2/(2T^2) F_yy + (r - sigma^2/2)(T - t)/T F_y - r F
//   = 0.
// The ln S terms cancel, which is why this y works.
//
// A closed-form solution. I_t = integral_0^t ln S(u) du (the continuously
// sampled geometric average is exp(I_T/T)), and y_T = I_T/T. Under the
// risk-neutral measure dy = (T - t)/T ((r - sigma^2/2) dt + sigma dW), so given
// y at t, y_T is normal with
//   mean m = y + (r - sigma^2/2) (T - t)^2/(2T),   variance v = sigma^2 (T -
//   t)^3/(3 T^2).
// The geometric Asian call (payoff (e^(y_T) - K)^+) is therefore
//   F(y, t) = e^(-r (T - t)) [ e^(m + v/2) N(d1) - K N(d2) ],
//   d2 = (m - ln K)/sqrt(v), d1 = d2 + sqrt(v).
// The tests check by finite differences that F solves the PDE for F and that
// V(S, I, t) = F(y, t) solves (8.52).
struct AsianMarket {
  double K = 50.0, T = 1.0, sigma = 0.3, r = 0.05;
};

[[nodiscard]] inline double ex12_y(double S, double I, double t, double T) {
  return (I + (T - t) * std::log(S)) / T;
}

[[nodiscard]] inline double ex12_geometric_asian_call(
    double y, double t, const AsianMarket& m = {}) {
  const double tau = m.T - t;
  if (tau <= 0.0) return std::max(std::exp(y) - m.K, 0.0);
  const double s2 = m.sigma * m.sigma;
  const double mean = y + (m.r - 0.5 * s2) * tau * tau / (2.0 * m.T);
  const double sd = std::sqrt(s2 * tau * tau * tau / (3.0 * m.T * m.T));
  const double d2 = (mean - std::log(m.K)) / sd;
  return std::exp(-m.r * tau) *
         (std::exp(mean + 0.5 * sd * sd) * norm_cdf(d2 + sd) -
          m.K * norm_cdf(d2));
}

// Left side of the PDE for F, by central differences in y and t.
template <std::invocable<double, double> F>
[[nodiscard]] double ex12_F_residual(F&& f, double y, double t, double T,
                                     double sigma, double r, double h = 1e-4) {
  const double Ft = (f(y, t + h) - f(y, t - h)) / (2.0 * h);
  const double Fy = (f(y + h, t) - f(y - h, t)) / (2.0 * h);
  const double Fyy = (f(y + h, t) - 2.0 * f(y, t) + f(y - h, t)) / (h * h);
  const double w = (T - t) / T;
  return Ft + 0.5 * sigma * sigma * w * w * Fyy +
         (r - 0.5 * sigma * sigma) * w * Fy - r * f(y, t);
}

// Left side of (8.52), by central differences (relative step h in S).
template <std::invocable<double, double, double> V>
[[nodiscard]] double ex12_V_residual(V&& v, double S, double I, double t,
                                     double sigma, double r, double h = 1e-4) {
  const double hS = h * S;
  const double Vt = (v(S, I, t + h) - v(S, I, t - h)) / (2.0 * h);
  const double VI = (v(S, I + h, t) - v(S, I - h, t)) / (2.0 * h);
  const double VS = (v(S + hS, I, t) - v(S - hS, I, t)) / (2.0 * hS);
  const double VSS =
      (v(S + hS, I, t) - 2.0 * v(S, I, t) + v(S - hS, I, t)) / (hS * hS);
  return Vt + std::log(S) * VI + 0.5 * sigma * sigma * S * S * VSS +
         r * S * VS - r * v(S, I, t);
}

// ----------------------------------------------------------- Exercise 13 ---
// Assume V(S, I, t) satisfies
//   V_t + S V_I + sigma^2 S^2 V_SS / 2 + r S V_S - r V = 0.          (8.53)
// With V(S, I, t) = S H(R, t), R = I/S (8.54), show that
//   H_t + sigma^2 R^2 H_RR / 2 + (1 - r R) H_R = 0.                  (8.55)
//
// Proof. R_I = 1/S, R_S = -I/S^2 = -R/S. Then
//   V_t  = S H_t,
//   V_I  = S H_R R_I = H_R,
//   V_S  = H + S H_R R_S = H - R H_R,
//   V_SS = d/dS (H - R H_R) = H_R R_S - R_S H_R - R H_RR R_S = R^2 H_RR / S.
// Substituting in (8.53):
//   S H_t + S H_R + sigma^2 S^2 R^2 H_RR/(2S) + r S (H - R H_R) - r S H
//   = S [ H_t + sigma^2 R^2 H_RR/2 + (1 - r R) H_R ] = 0,
// and dividing by S > 0 gives (8.55). The r H terms cancel.
//
// An exact solution: H(R, t) = alpha(t) + beta(t) R with
// beta = e^(-r (T - t)), alpha = (1 - e^(-r (T - t)))/r. Indeed
// H_t + (1 - rR) H_R = alpha' + beta' R + beta - r R beta = 0 because
// alpha' = -beta and beta' = r beta. In the original variables
// V = S (1 - e^(-r (T - t)))/r + e^(-r (T - t)) I, the value of the payoff
// I_T = integral_0^T S(u) du.
// The tests check (8.53) for this V, (8.55) for this H, and the identity
// residual_(8.53)(S H(I/S, t)) = S * residual_(8.55)(H) for a generic smooth H.
[[nodiscard]] inline double ex13_H_linear(double R, double t, double T,
                                          double r) {
  const double beta = std::exp(-r * (T - t));
  return (1.0 - beta) / r + beta * R;
}

template <std::invocable<double, double> H>
[[nodiscard]] double ex13_H_residual(H&& f, double R, double t, double sigma,
                                     double r, double h = 1e-4) {
  const double hR = h * std::max(1.0, std::abs(R));
  const double Ht = (f(R, t + h) - f(R, t - h)) / (2.0 * h);
  const double HR = (f(R + hR, t) - f(R - hR, t)) / (2.0 * hR);
  const double HRR = (f(R + hR, t) - 2.0 * f(R, t) + f(R - hR, t)) / (hR * hR);
  return Ht + 0.5 * sigma * sigma * R * R * HRR + (1.0 - r * R) * HR;
}

template <std::invocable<double, double, double> V>
[[nodiscard]] double ex13_V_residual(V&& v, double S, double I, double t,
                                     double sigma, double r, double h = 1e-4) {
  const double hS = h * S, hI = h * std::max(1.0, std::abs(I));
  const double Vt = (v(S, I, t + h) - v(S, I, t - h)) / (2.0 * h);
  const double VI = (v(S, I + hI, t) - v(S, I - hI, t)) / (2.0 * hI);
  const double VS = (v(S + hS, I, t) - v(S - hS, I, t)) / (2.0 * hS);
  const double VSS =
      (v(S + hS, I, t) - 2.0 * v(S, I, t) + v(S - hS, I, t)) / (hS * hS);
  return Vt + S * VI + 0.5 * sigma * sigma * S * S * VSS + r * S * VS -
         r * v(S, I, t);
}

}  // namespace p4mfe::ch08::ex
