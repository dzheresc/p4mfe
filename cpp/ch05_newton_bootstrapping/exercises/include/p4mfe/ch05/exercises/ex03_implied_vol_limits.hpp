#pragma once
// Exercise 3 (section 5.7): the implied volatility equation f(x) = 0, its
// limits as x -> 0 and x -> infinity, monotonicity, and when it has a root.

#include <cmath>
#include <limits>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 3 ---
// Finding the implied volatility from the price C of a call is solving
//   f(x) = S e^(-qT) N(d1(x)) - K e^(-rT) N(d2(x)) - C = 0,
//   d1,2(x) = (ln(S/K) + (r - q +- x^2/2) T) / (x sqrt(T)).
// Write F = S e^((r-q)T) for the forward price, so ln(S/K) + (r - q)T =
// ln(F/K).
//
// (i) d1(x) = ln(F/K)/(x sqrt(T)) + x sqrt(T)/2 and
//     d2(x) = ln(F/K)/(x sqrt(T)) - x sqrt(T)/2.
//     As x -> infinity the first term goes to 0 and the second to +-infinity,
//     so d1 -> +infinity, d2 -> -infinity, N(d1) -> 1, N(d2) -> 0 and
//       lim_{x->inf} f(x) = S e^(-qT) - C.
//
// (ii) As x -> 0+, x sqrt(T)/2 -> 0 while ln(F/K)/(x sqrt(T)) tends to
//      -infinity, 0 or +infinity as F < K, F = K or F > K (for F = K it is 0
//      for every x). Hence d1 and d2 have the same limit: -infinity, 0 or
//      +infinity. Then
//        F < K:  N(d1), N(d2) -> 0, f -> -C;
//        F = K:  N(d1), N(d2) -> 1/2, f -> (S e^(-qT) - K e^(-rT))/2 - C = -C,
//                since F = K means S e^(-qT) = K e^(-rT);
//        F > K:  N(d1), N(d2) -> 1, f -> S e^(-qT) - K e^(-rT) - C.
//      So lim_{x->0} f = -C if F <= K and S e^(-qT) - K e^(-rT) - C if F > K.
//
// (iii) f'(x) = vega = S e^(-qT) sqrt(T) phi(d1(x)) > 0 (the identity
//       S e^(-qT) phi(d1) = K e^(-rT) phi(d2) cancels the d2 terms), so f is
//       strictly increasing on (0, infinity) and lies strictly between its
//       limits at 0 and infinity: the two chains of inequalities follow.
//
// (iv) A continuous strictly increasing f has a (unique) positive root iff
//      lim_{x->0} f < 0 < lim_{x->inf} f, i.e.
//        F <= K:  0 < C < S e^(-qT)
//        F >  K:  S e^(-qT) - K e^(-rT) < C < S e^(-qT).
//      Since S e^(-qT) - K e^(-rT) <= 0 exactly when F <= K, both cases read
//        max(S e^(-qT) - K e^(-rT), 0) < C < S e^(-qT),
//      the no-arbitrage range (3.99) of a call with the end points removed
//      (ch01::call_bounds). The end points correspond to sigma = 0 and
//      sigma = infinity, so no finite positive volatility attains them.
struct Ex03Market {
  double spot, strike, maturity, r, q, call;
};

[[nodiscard]] inline double Ex03F(double x, const Ex03Market& m) {
  return ch03::BsCall(m.spot, m.strike, m.maturity, x, m.r, m.q) - m.call;
}
[[nodiscard]] inline double Ex03Forward(const Ex03Market& m) {
  return m.spot * std::exp((m.r - m.q) * m.maturity);
}

// (i)
[[nodiscard]] inline double Ex03LimitInfinity(const Ex03Market& m) {
  return m.spot * std::exp(-m.q * m.maturity) - m.call;
}

// (ii): the common limit of d1 and d2, and of f, as x -> 0.
[[nodiscard]] inline double Ex03DLimitZero(const Ex03Market& m) {
  const double forward = Ex03Forward(m);
  if (forward < m.strike) return -std::numeric_limits<double>::infinity();
  if (forward > m.strike) return std::numeric_limits<double>::infinity();
  return 0.0;
}
[[nodiscard]] inline double Ex03LimitZero(const Ex03Market& m) {
  if (Ex03Forward(m) <= m.strike) return -m.call;
  return m.spot * std::exp(-m.q * m.maturity) -
         m.strike * std::exp(-m.r * m.maturity) - m.call;
}

// (iv)
struct Ex03Range {
  double lower,
      upper;  // open interval of call prices with an implied volatility
};
[[nodiscard]] inline Ex03Range Ex03PriceRange(double spot, double strike,
                                              double maturity, double r,
                                              double q) {
  const double s = spot * std::exp(-q * maturity);
  return {std::max(s - strike * std::exp(-r * maturity), 0.0), s};
}
[[nodiscard]] inline bool Ex03HasPositiveRoot(const Ex03Market& m) {
  return Ex03LimitZero(m) < 0.0 && Ex03LimitInfinity(m) > 0.0;
}

}  // namespace p4mfe::ch05::ex
