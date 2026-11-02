#pragma once
// Exercises 19-20 (section 8.12): calls on non-dividend-paying assets are
// worth more than their intrinsic value; with dividends they are not.

#include <algorithm>
#include <cmath>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch05/solvers.hpp>
#include <p4mfe/ch08/early_exercise.hpp>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 19 ---
// (i) Show that a European call on a non-dividend-paying asset is always worth
// more than max(S - K, 0).
//   Assume r >= 0 (as throughout; for r < 0 the claim fails deep in the money).
//   By Put-Call parity (q = 0)
//     C = P + S - K e^(-rT) > S - K e^(-rT) >= S - K,
//   using P > 0 (a Black-Scholes put has positive value) and e^(-rT) <= 1.
//   Also C > 0 (C = e^(-rT) E[(S_T - K)^+] and S_T > K with positive
//   probability). So C > max(S - K, 0).
// (ii) Conclude that American calls on non-dividend-paying assets are never
// optimal to exercise early.
//   At any t < T, the American call is worth at least the European call
//   (it has more rights): C_A(S, t) >= C(S, t) > max(S - K, 0). Exercising at
//   t gives max(S - K, 0) < C(S, t), and one could instead sell the option
//   (or the European call it dominates) for more. So early exercise is never
//   optimal and C_A = C. The check below compares American and European
//   binomial prices (chapter 8's binomial_price) for q = 0.
struct CallMarket {
  double strike = 40.0, maturity = 1.0, sigma = 0.3, r = 0.05, q = 0.0;
};

[[nodiscard]] inline double Ex19TimeValue(double spot,
                                          const CallMarket& m = {}) {
  return ch03::BsCall(spot, m.strike, m.maturity, m.sigma, m.r, m.q) -
         std::max(spot - m.strike, 0.0);
}

// The lower bound used in (i): S - K e^(-rT) - (S - K) = K (1 - e^(-rT)).
[[nodiscard]] inline double Ex19LowerBoundGap(const CallMarket& m = {}) {
  return m.strike * -std::expm1(-m.r * m.maturity);
}

// ----------------------------------------------------------- Exercise 20 ---
// Show that the Black-Scholes value of a call on an asset paying dividends
// continuously at rate q > 0 is smaller than S - K if S is large enough,
// however small q is.
//
// Since N(d1) < 1 and N(d2) > 0,
//   C = S e^(-qT) N(d1) - K e^(-rT) N(d2) < S e^(-qT).
// Hence
//   C - (S - K) < S e^(-qT) - S + K = K - S (1 - e^(-qT)),
// which is <= 0 as soon as S >= S0 = K / (1 - e^(-qT)). Since q > 0, 1 -
// e^(-qT) > 0 and S0 is finite, however small q is (S0 ~ K/(qT) for small qT).
// The exact crossover S* where C(S*) = S* - K lies in (K, S0) (at S = K,
// C > 0 = S - K) and is found by bisection (chapter 5). For small q,
// C - (S - K) -> K (1 - e^(-rT)) - S (1 - e^(-qT)) as S -> infinity, so
// S* ~ K (1 - e^(-rT))/(1 - e^(-qT)) ~ K r/q.
[[nodiscard]] inline double Ex20Threshold(const CallMarket& m) {
  return m.strike / -std::expm1(-m.q * m.maturity);
}

[[nodiscard]] inline double Ex20Crossover(const CallMarket& m) {
  const auto f = [&](double spot) {
    return ch03::BsCall(spot, m.strike, m.maturity, m.sigma, m.r, m.q) -
           (spot - m.strike);
  };
  const double spot_threshold = Ex20Threshold(m);
  return ch05::Bisection(f, m.strike, spot_threshold,
                         {1e-12, 1e-10 * spot_threshold, 400})
      .x;
}

}  // namespace p4mfe::ch08::ex
