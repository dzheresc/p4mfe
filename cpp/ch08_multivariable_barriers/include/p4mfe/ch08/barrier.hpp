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
[[nodiscard]] inline double ImageExponent(double sigma, double r,
                                          double q = 0.0) noexcept {
  return 2.0 * (r - q) / (sigma * sigma) - 1.0;
}

// V*(S) = (B/S)^(k1 - 1) V(B^2/S).
template <std::invocable<double> V>
[[nodiscard]] double BarrierImage(V&& v, double spot, double barrier,
                                  double sigma, double r, double q = 0.0) {
  return std::pow(barrier / spot, ImageExponent(sigma, r, q)) *
         v(barrier * barrier / spot);
}

[[nodiscard]] inline double DownAndOutCall(double spot, double strike,
                                           double barrier, double maturity,
                                           double sigma, double r,
                                           double q = 0.0) {
  if (spot <= barrier) return 0.0;
  const auto vanilla = [&](double s) {
    if (barrier <= strike)
      return ch03::BsCall(s, strike, maturity, sigma, r, q);
    return ch03::BsCall(s, barrier, maturity, sigma, r, q) +
           (barrier - strike) * ch04::CashOrNothing(OptionType::kCall, s,
                                                    barrier, maturity, sigma, r,
                                                    q);
  };
  return vanilla(spot) - BarrierImage(vanilla, spot, barrier, sigma, r, q);
}

[[nodiscard]] inline double DownAndInCall(double spot, double strike,
                                          double barrier, double maturity,
                                          double sigma, double r,
                                          double q = 0.0) {
  return ch03::BsCall(spot, strike, maturity, sigma, r, q) -
         DownAndOutCall(spot, strike, barrier, maturity, sigma, r, q);
}

[[nodiscard]] inline double UpAndOutPut(double spot, double strike,
                                        double barrier, double maturity,
                                        double sigma, double r,
                                        double q = 0.0) {
  if (spot >= barrier) return 0.0;
  const auto vanilla = [&](double s) {
    if (barrier >= strike) return ch03::BsPut(s, strike, maturity, sigma, r, q);
    return ch03::BsPut(s, barrier, maturity, sigma, r, q) +
           (strike - barrier) * ch04::CashOrNothing(OptionType::kPut, s,
                                                    barrier, maturity, sigma, r,
                                                    q);
  };
  return vanilla(spot) - BarrierImage(vanilla, spot, barrier, sigma, r, q);
}

[[nodiscard]] inline double UpAndInPut(double spot, double strike,
                                       double barrier, double maturity,
                                       double sigma, double r, double q = 0.0) {
  return ch03::BsPut(spot, strike, maturity, sigma, r, q) -
         UpAndOutPut(spot, strike, barrier, maturity, sigma, r, q);
}

}  // namespace p4mfe::ch08
