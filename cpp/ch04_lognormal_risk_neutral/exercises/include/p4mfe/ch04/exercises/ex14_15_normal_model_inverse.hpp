#pragma once
// Exercises 14-15 (section 4.9): call and put under a normal (Bachelier)
// model, and derivatives paying 1/S(T) and (1/K - 1/S(T))^+ on a lognormal
// asset.

#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch04::ex {

// ----------------------------------------------------------- Exercise 14 ---
// The asset follows a normal process, dS = mu dt + sigma dX:
//   S(t2) = S(t1) + mu (t2 - t1) + sigma sqrt(t2 - t1) Z,  0 < t1 < t2.
// The risk-free rate is constant at r.
// (i) Use risk neutrality to find the value of a call with strike K and
//     maturity T, C(0) = e^(-rT) E_RN[max(S(T) - K, 0)], with
//     S(T) = S(0) + rT + sigma sqrt(T) Z.
// (ii) Use Put-Call parity to find the value of the put with strike K and
//      maturity T.
//
// (i) Let F = S(0) + rT (the mean of S(T)), s = sigma sqrt(T), d = (F - K)/s.
// S(T) > K  <=>  Z > -d, so
//   E[(S(T) - K)^+] = Integral_{-d}^inf (F - K + s z) phi(z) dz
//                   = (F - K)(1 - N(-d)) + s [-phi(z)]_{-d}^inf
//                   = (F - K) N(d) + s phi(d),
// using phi' = -z phi and phi(-d) = phi(d). Hence
//   C(0) = e^(-rT) [(F - K) N(d) + sigma sqrt(T) phi(d)],  F = S(0) + rT.
// (ii) Put-Call parity C - P = S(0) - K e^(-rT) holds for any model of a
// non-dividend-paying asset (it is a static replication), so
//   P(0) = C(0) - S(0) + K e^(-rT).
//
// Remark: the book's risk-neutral dynamics S(T) = S(0) + rT + sigma sqrt(T) Z
// do not make e^(-rt) S(t) a martingale (that needs E_RN[S(T)] = S(0) e^(rT)).
// So pricing the put directly, e^(-rT) E_RN[(K - S(T))^+]
// = e^(-rT) [(K - F) N(-d) + s phi(d)], differs from the parity value by
//   e^(-rT)(F - K) - (S(0) - K e^(-rT)) = e^(-rT)(S(0) + rT) - S(0).
// With the arbitrage-free mean F = S(0) e^(rT) the two puts coincide; each
// *_forward function takes F; ex14_call uses the book's S(0) + rT.
[[nodiscard]] inline double Ex14BookForward(double spot0, double r,
                                            double maturity) noexcept {
  return spot0 + r * maturity;
}
// e^(-rT) E_RN[(S(T) - K)^+] for S(T) ~ N(F, sigma^2 T).
[[nodiscard]] inline double Ex14CallForward(double forward, double strike,
                                            double maturity, double sigma,
                                            double r) {
  const double s = sigma * std::sqrt(maturity);
  const double d = (forward - strike) / s;
  return std::exp(-r * maturity) *
         ((forward - strike) * NormCdf(d) + s * NormPdf(d));
}
// e^(-rT) E_RN[(K - S(T))^+] computed directly, for the remark above.
[[nodiscard]] inline double Ex14PutForward(double forward, double strike,
                                           double maturity, double sigma,
                                           double r) {
  const double s = sigma * std::sqrt(maturity);
  const double d = (forward - strike) / s;
  return std::exp(-r * maturity) *
         ((strike - forward) * NormCdf(-d) + s * NormPdf(d));
}
// (i) with the book's F = S(0) + rT.
[[nodiscard]] inline double Ex14Call(double spot0, double strike,
                                     double maturity, double sigma, double r) {
  return Ex14CallForward(Ex14BookForward(spot0, r, maturity), strike, maturity,
                         sigma, r);
}
// (ii) P = C - S(0) + K e^(-rT).
[[nodiscard]] inline double Ex14PutParity(double spot0, double strike,
                                          double maturity, double sigma,
                                          double r) {
  return Ex14Call(spot0, strike, maturity, sigma, r) - spot0 +
         strike * std::exp(-r * maturity);
}

// ----------------------------------------------------------- Exercise 15 ---
// (i) Value at time 0 of a derivative that pays 1/S(T) at time T, on a
//     lognormal asset with volatility sigma paying dividends at rate q;
//     interest rates constant at r.
// (ii) Value at time 0 of the derivative with payoff V(T) = max(1/K - 1/S(T),
// 0).
//
// Under the risk-neutral measure ln S(T) = ln S_0 + (r - q - sigma^2/2) T +
// sigma sqrt(T) Z, so Y = 1/S(T) is lognormal with
//   ln Y ~ N(m, s^2),  m = -ln S_0 - (r - q - sigma^2/2) T,  s = sigma sqrt(T).
// (i) E[Y] = e^(m + s^2/2) = (1/S_0) e^((-r + q + sigma^2) T), so
//   V(0) = e^(-rT) E_RN[1/S(T)] = (1/S_0) e^((q - 2r + sigma^2) T).
// (ii) 1/K - 1/S(T) > 0  <=>  S(T) > K, so
//   V(0) = e^(-rT) [(1/K) P(S(T) > K) - E[(1/S(T)) 1_{S(T) > K}]].
// P(S(T) > K) = N(d2). For the second term use E[e^(kX) 1_{X > a}] with
// X = ln S(T), k = -1, a = ln K (see ex12_13_power_payoffs.hpp):
//   E[(1/S(T)) 1_{S(T)>K}] = e^(-mX + s^2/2) N(d2 - sigma sqrt(T)),  mX = ln
//   S_0 + (r - q - sigma^2/2) T,
//                          = (1/S_0) e^((-r + q + sigma^2) T) N(d2 - sigma
//                          sqrt(T)).
// Hence
//   V(0) = (e^(-rT)/K) N(d2) - (1/S_0) e^((q - 2r + sigma^2) T) N(d2 - sigma
//   sqrt(T)),
// with d2 the Black-Scholes d2. (Note d2 - sigma sqrt(T) = d1 - 2 sigma
// sqrt(T).)
[[nodiscard]] inline double Ex15InverseClaim(double spot0, double maturity,
                                             double sigma, double r,
                                             double q = 0.0) {
  return std::exp((q - 2.0 * r + sigma * sigma) * maturity) / spot0;
}
[[nodiscard]] inline double Ex15InverseCall(double spot0, double strike,
                                            double maturity, double sigma,
                                            double r, double q = 0.0) {
  const double d2 = ch03::BsD1D2(spot0, strike, maturity, sigma, r, q).d2;
  return std::exp(-r * maturity) / strike * NormCdf(d2) -
         Ex15InverseClaim(spot0, maturity, sigma, r, q) *
             NormCdf(d2 - sigma * std::sqrt(maturity));
}

}  // namespace p4mfe::ch04::ex
