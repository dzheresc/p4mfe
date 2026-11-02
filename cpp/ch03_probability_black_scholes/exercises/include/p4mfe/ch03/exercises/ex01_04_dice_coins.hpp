#pragma once
// Exercises 1-4 (section 3.11): a dice game, and the expected number of coin
// tosses until the first head and until k heads in a row.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch03::ex {

// ------------------------------------------------------------ Exercise 1 ---
// Let k be a positive integer with 2 <= k <= 12. You throw two fair dice. If
// the sum of the dice is k, you win w(k), or lose 1 otherwise. Find the
// smallest value of w(k) that makes the game worth playing.
//
// Of the 36 equally likely outcomes, n(k) = 6 - |k - 7| have sum k, so
// p(k) = n(k)/36. The expected gain is
//   E = w(k) p(k) - (1 - p(k)),
// and the game is worth playing when E >= 0, i.e.
//   w(k) >= (1 - p(k)) / p(k) = 36/n(k) - 1.
//   k     2,12   3,11   4,10   5,9   6,8   7
//   w(k)   35     17     11     8    6.2   5
// (At this w(k) the game is fair; any larger w(k) gives a positive edge.)
[[nodiscard]] inline int Ex01Ways(int k) {
  if (k < 2 || k > 12) throw std::invalid_argument("ex01: need 2 <= k <= 12");
  return 6 - std::abs(k - 7);
}
[[nodiscard]] inline double Ex01Prob(int k) { return Ex01Ways(k) / 36.0; }
[[nodiscard]] inline double Ex01BreakevenWin(int k) {
  return 36.0 / Ex01Ways(k) - 1.0;
}
[[nodiscard]] inline double Ex01ExpectedGain(int k, double w) {
  const double p = Ex01Prob(k);
  return w * p - (1.0 - p);
}

// ------------------------------------------------------------ Exercise 2 ---
// Let X be the number of times you must flip a fair coin until it lands
// heads. What are E[X] and var(X)?
//
// X is geometric: P(X = n) = (1 - p)^(n-1) p with p = 1/2. Differentiating
// the geometric series sum_{n>=0} x^n = 1/(1 - x) once and twice gives
//   sum n x^(n-1) = 1/(1-x)^2,   sum n(n-1) x^(n-2) = 2/(1-x)^3,
// so with x = 1 - p:
//   E[X] = p / p^2 = 1/p,   E[X(X-1)] = 2(1-p)/p^2,
//   var(X) = E[X(X-1)] + E[X] - E[X]^2 = (1-p)/p^2.
// For a fair coin, E[X] = 2 and var(X) = 2.
[[nodiscard]] inline double Ex02Mean(double p = 0.5) { return 1.0 / p; }
[[nodiscard]] inline double Ex02Variance(double p = 0.5) {
  return (1.0 - p) / (p * p);
}

// ---------------------------------------------------------- Exercises 3-4 --
// 3. What is the expected number of coin tosses of a fair coin in order to get
//    two heads in a row? What if the coin is biased and the probability of
//    getting heads is p?
// 4. What is the expected number of tosses in order to get k heads in a row
//    for a biased coin with probability of getting heads equal to p?
//
// Let E_k be the expected number of tosses until k heads in a row. To get k
// in a row you first need k - 1 in a row (E_{k-1} tosses on average), then
// one more toss: with probability p it is a head and you are done; with
// probability 1 - p it is a tail and you start over, needing E_k more. So
//   E_k = E_{k-1} + 1 + (1 - p) E_k   =>   E_k = (E_{k-1} + 1) / p,
// with E_0 = 0. Unrolling,
//   E_k = 1/p + 1/p^2 + ... + 1/p^k = (1 - p^k) / ((1 - p) p^k).
// Exercise 3 (k = 2): E_2 = (1 + p)/p^2, which is 6 for a fair coin.
// For p = 1/2 in general, E_k = 2^(k+1) - 2.
[[nodiscard]] inline double Ex04ExpectedTosses(int k, double p) {
  if (k < 0 || !(p > 0.0 && p <= 1.0))
    throw std::invalid_argument("ex04: need k >= 0, 0 < p <= 1");
  if (p == 1.0) return k;
  return (1.0 - std::pow(p, k)) / ((1.0 - p) * std::pow(p, k));
}
[[nodiscard]] inline double Ex04ExpectedTossesRecursive(int k, double p) {
  double e = 0.0;
  for (int j = 1; j <= k; ++j) e = (e + 1.0) / p;
  return e;
}
[[nodiscard]] inline double Ex03ExpectedTosses(double p = 0.5) {
  return (1.0 + p) / (p * p);
}

// Independent check: E[N] = sum_{n>=0} P(N > n), where the distribution of the
// current run of heads (0..k-1, not yet finished) is propagated toss by toss.
[[nodiscard]] inline double Ex04ExpectedTossesMarkov(int k, double p,
                                                     double tol = 1e-14) {
  std::vector<double> state(static_cast<std::size_t>(k), 0.0);
  state[0] = 1.0;
  double expected = 0.0;
  for (int n = 0; n < 10'000'000; ++n) {
    double alive = 0.0;
    for (double s : state) alive += s;
    expected += alive;  // P(N > n)
    if (alive < tol) break;
    std::vector<double> next(state.size(), 0.0);
    for (std::size_t j = 0; j < state.size(); ++j) {
      next[0] += (1.0 - p) * state[j];
      if (j + 1 < state.size()) next[j + 1] += p * state[j];
    }
    state = std::move(next);
  }
  return expected;
}

}  // namespace p4mfe::ch03::ex
