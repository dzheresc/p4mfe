#pragma once
// Chapter 8, section 8.9: barrier options.
//
// A down-and-out call pays (S_T - K)^+ unless S hits the barrier B < S_0
// before T, in which case it dies. Its value solves the Black-Scholes PDE
// (7.4) for S > B with V(B, t) = 0. Through the heat equation (8.8) this is a
// heat problem on a half-line with zero boundary value, solved by the method
// of images. Back in the original variables: if V(S, t) solves the PDE, so
// does its image
//   V*(S, t) = (B/S)^(k1 - 1) V(B^2/S, t),   k1 = 2 (r - q) / sigma^2,
// and V* = V on S = B. So with W the value of the payoff restricted to the
// "alive" side of the barrier, the knock-out value is W(S) - W*(S):
//   down-and-out call: W = value of (S_T - K)^+ 1{S_T > B}
//                        = C(K)                              if B <= K,
//                        = C(B) + (B - K) e^(-rT) N(d2(B))   if B > K;
//   up-and-out put:    W = value of (K - S_T)^+ 1{S_T < B}  (mirror image).
// Knock-in = vanilla - knock-out (in-out parity).

#include <cmath>
#include <concepts>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/in_the_money.hpp>

namespace p4mfe::ch08 {

using ch01::OptionType;

// ---------------------------------------------------------------- 8.9 ------
[[nodiscard]] inline double image_exponent(double sigma, double r,
                                           double q = 0.0) noexcept {
  return 2.0 * (r - q) / (sigma * sigma) - 1.0;
}

// V*(S) = (B/S)^(k1 - 1) V(B^2/S).
template <std::invocable<double> V>
[[nodiscard]] double barrier_image(V&& v, double S, double B, double sigma,
                                   double r, double q = 0.0) {
  return std::pow(B / S, image_exponent(sigma, r, q)) * v(B * B / S);
}

[[nodiscard]] inline double down_and_out_call(double S, double K, double B,
                                              double T, double sigma, double r,
                                              double q = 0.0) {
  if (S <= B) return 0.0;
  const auto W = [&](double s) {
    if (B <= K) return ch03::BsCall(s, K, T, sigma, r, q);
    return ch03::BsCall(s, B, T, sigma, r, q) +
           (B - K) *
               ch04::CashOrNothing(OptionType::kCall, s, B, T, sigma, r, q);
  };
  return W(S) - barrier_image(W, S, B, sigma, r, q);
}

[[nodiscard]] inline double down_and_in_call(double S, double K, double B,
                                             double T, double sigma, double r,
                                             double q = 0.0) {
  return ch03::BsCall(S, K, T, sigma, r, q) -
         down_and_out_call(S, K, B, T, sigma, r, q);
}

[[nodiscard]] inline double up_and_out_put(double S, double K, double B,
                                           double T, double sigma, double r,
                                           double q = 0.0) {
  if (S >= B) return 0.0;
  const auto W = [&](double s) {
    if (B >= K) return ch03::BsPut(s, K, T, sigma, r, q);
    return ch03::BsPut(s, B, T, sigma, r, q) +
           (K - B) *
               ch04::CashOrNothing(OptionType::kPut, s, B, T, sigma, r, q);
  };
  return W(S) - barrier_image(W, S, B, sigma, r, q);
}

[[nodiscard]] inline double up_and_in_put(double S, double K, double B,
                                          double T, double sigma, double r,
                                          double q = 0.0) {
  return ch03::BsPut(S, K, T, sigma, r, q) -
         up_and_out_put(S, K, B, T, sigma, r, q);
}

}  // namespace p4mfe::ch08
