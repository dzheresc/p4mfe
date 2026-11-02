#pragma once
// Exercise 18 (section 8.12): Delta and Gamma of a down-and-out call.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch08/barrier.hpp>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 18 ---
// Compute the Delta and Gamma of a down-and-out call with B < K using (8.46):
//   V(S) = C(S) - g(S) C(X),   g(S) = (B/S)^p,   X = B^2/S,   p = 2(r -
//   q)/sigma^2 - 1,
// for S > B, where C, Delta_C = e^(-qT) N(d1), Gamma_C = e^(-qT) phi(d1)/(S
// sigma sqrt(T)) are the Black-Scholes call value and Greeks with strike K.
//
// Derivatives of the pieces:
//   g'(S)  = -p g(S)/S,              g''(S) = p (p + 1) g(S)/S^2,
//   X'(S)  = -B^2/S^2 = -X/S,        X''(S) = 2X/S^2,
//   d/dS C(X)   = Delta_C(X) X' = -(X/S) Delta_C(X),
//   d2/dS2 C(X) = Gamma_C(X) X'^2 + Delta_C(X) X'' = (X^2/S^2) Gamma_C(X) +
//   (2X/S^2) Delta_C(X).
// Product rule:
//   Delta_V = Delta_C(S) - g' C(X) - g (d/dS C(X))
//           = Delta_C(S) + (g/S) [ p C(X) + X Delta_C(X) ],
//   Gamma_V = Gamma_C(S) - [ g'' C(X) + 2 g' (d/dS C(X)) + g (d2/dS2 C(X)) ]
//           = Gamma_C(S) - (g/S^2) [ p (p + 1) C(X) + 2 (p + 1) X Delta_C(X) +
//           X^2 Gamma_C(X) ].
// Explicitly, with d1(X) = (ln(B^2/(S K)) + (r - q + sigma^2/2) T)/(sigma
// sqrt(T)):
//   X Delta_C(X)   = (B^2/S) e^(-qT) N(d1(X)),
//   X^2 Gamma_C(X) = (B^2/S) e^(-qT) phi(d1(X))/(sigma sqrt(T)).
//
// For the exercise 16 data (S = K = 40, B = 35, T = 1/2, sigma = 30%, r = 5%):
//   Delta = 0.703253,  Gamma = 0.018758,
// against Delta = 0.588589 and Gamma = 0.045852 for the vanilla call. The
// knock-out has the larger Delta (its value must drop to 0 at S = B), and its
// Gamma turns negative close to the barrier: at S = 36, Delta = 0.674361 and
// Gamma = -0.009534.
struct DocGreeks {
  double delta;
  double gamma;
};

[[nodiscard]] inline DocGreeks Ex18Greeks(double spot, double strike,
                                          double barrier, double maturity,
                                          double sigma, double r,
                                          double q = 0.0) {
  const double p = 2.0 * (r - q) / (sigma * sigma) - 1.0;
  const double g = std::pow(barrier / spot, p);
  const double reflected_spot = barrier * barrier / spot;
  const auto gs = ch03::BsGreeks(ch03::OptionType::kCall, spot, strike,
                                 maturity, sigma, r, q);
  const auto gx = ch03::BsGreeks(ch03::OptionType::kCall, reflected_spot,
                                 strike, maturity, sigma, r, q);
  const double cx = ch03::BsCall(reflected_spot, strike, maturity, sigma, r, q);
  return {gs.delta + g / spot * (p * cx + reflected_spot * gx.delta),
          gs.gamma - g / (spot * spot) *
                         (p * (p + 1.0) * cx +
                          2.0 * (p + 1.0) * reflected_spot * gx.delta +
                          reflected_spot * reflected_spot * gx.gamma)};
}

// Central finite differences of the library's down_and_out_call, for checking.
[[nodiscard]] inline DocGreeks Ex18GreeksFd(double spot, double strike,
                                            double barrier, double maturity,
                                            double sigma, double r,
                                            double q = 0.0, double h = 2e-4) {
  const auto v = [&](double s) {
    return DownAndOutCall(s, strike, barrier, maturity, sigma, r, q);
  };
  const double hs = h * spot;
  return {(v(spot + hs) - v(spot - hs)) / (2.0 * hs),
          (v(spot + hs) - 2.0 * v(spot) + v(spot - hs)) / (hs * hs)};
}

}  // namespace p4mfe::ch08::ex
