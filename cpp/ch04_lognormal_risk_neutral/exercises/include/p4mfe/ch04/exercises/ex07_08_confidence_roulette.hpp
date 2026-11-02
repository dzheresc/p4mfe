#pragma once
// Exercises 7-8 (section 4.9): confidence intervals for a lognormal price
// (real-world and risk-neutral), and the roulette question solved exactly
// and with the normal approximation.

#include <array>
#include <cmath>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch04::ex {

// ------------------------------------------------------------ Exercise 7 ---
// An asset with spot price 50 paying dividends continuously at q = 0.02 has
// lognormal distribution with mean (drift) mu = 0.08 and volatility
// sigma = 0.3; the risk-free rate is constant at r = 0.05.
// (i) Find 95% and 99% confidence intervals for the spot price in 15 days,
//     1 month, 2 months, 6 months and 1 year.
// (ii) Find the 95% and 99% risk-neutral confidence intervals for the same
//      horizons, i.e. assuming the drift of the asset is equal to r.
//
// ln(S_T/S_0) ~ N((mu - q - sigma^2/2) T, sigma^2 T). With
// z = N^(-1)((1 + c)/2) (1.959964 for c = 95%, 2.575829 for c = 99%):
//   P(-z <= Z <= z) = c  =>  S_T in [S_0 e^((mu - q - sigma^2/2)T - z sigma
//   sqrt(T)),
//                                    S_0 e^((mu - q - sigma^2/2)T + z sigma
//                                    sqrt(T))]
// with probability c (ch04::price_confidence_interval). The interval is
// symmetric in ln S_T, not in S_T: the upper half is wider.
// Assumption: 15 days = 1/24 year, k months = k/12 year.
//
//   horizon   (i) mu = 0.08                         (ii) mu = r = 0.05
//             95%               99%                 95%               99%
//   15 days   [44.3727, 56.4114] [42.7304, 58.5795]  [44.3173, 56.3409]
//   [42.6771, 58.5063] 1 month   [42.2471, 59.3238] [40.0528, 62.5738]
//   [42.1416, 59.1757] [39.9528, 62.4175] 2 months  [39.4280, 63.7245]
//   [36.5635, 68.7170]  [39.2314, 63.4066] [36.3811, 68.3743] 6 months
//   [33.2399, 76.3474] [29.1690, 87.0027]  [32.7451, 75.2107]
//   [28.7347, 85.7074] 1 year    [28.1919, 91.3787] [23.4360, 109.9221]
//   [27.3587, 88.6780] [22.7434, 106.6734]
// The risk-neutral intervals are shifted down by the factor e^(-(mu - r) T).
struct Ex07Asset {
  double spot0 = 50.0, q = 0.02, mu = 0.08, sigma = 0.3, r = 0.05;
};
inline constexpr std::array<double, 5> kEx07Horizons{1.0 / 24.0, 1.0 / 12.0,
                                                     2.0 / 12.0, 0.5, 1.0};
[[nodiscard]] inline PriceRange Ex07Interval(double confidence, double maturity,
                                             bool risk_neutral,
                                             const Ex07Asset& a = {}) {
  return PriceConfidenceInterval(confidence, a.spot0, maturity, a.sigma,
                                 risk_neutral ? a.r : a.mu, a.q);
}

// ------------------------------------------------------------ Exercise 8 ---
// If you play (American) roulette 100 times, betting $100 on black each time,
// what is the probability of winning at least $1000, and what is the
// probability of losing at least $1000? (American roulette: 18 red, 18 black
// and 2 green slots.)
//
// Each bet wins +100 with p = 18/38 = 9/19 and loses -100 with 1 - p = 10/19.
// With W ~ Binomial(100, p) the number of wins, the net gain is
//   G = 100 W - 100 (100 - W) = 200 W - 10000.
// G >= 1000  <=>  W >= 55;   G <= -1000  <=>  W <= 45.
// Exact (summing the binomial probabilities):
//   P(win >= $1000)  = P(W >= 55) = 0.076689,
//   P(lose >= $1000) = P(W <= 45) = 0.354794.
// Normal approximation (central limit theorem, Exercise 2 for the moments):
// one bet has mean 100 (2p - 1) = -100/19 = -5.263158 and variance
// 100^2 (1 - (2p - 1)^2) = 100^2 * 4 p (1 - p), so over 100 bets
//   E[G] = -526.3158,  sd(G) = 100 * 10 * sqrt(4 p (1 - p)) = 998.6140,
//   P(G >= 1000)  ~ 1 - N((1000 + 526.3158)/998.6140) = N(-1.528432) =
//   0.063202, P(G <= -1000) ~ N((-1000 + 526.3158)/998.6140) = N(-0.474341) =
//   0.317628.
// G moves in steps of 200, so a continuity correction (use 900 for 1000)
// gives 0.076603 and 0.354127, close to the exact values.
// The house edge makes losing $1000 about 4.6 times as likely as winning it.
inline constexpr double kEx08PBlack = 18.0 / 38.0;

// P(W = k) for W ~ Binomial(n, p), via lgamma to stay accurate for large n.
[[nodiscard]] inline double BinomialPmf(int k, int n, double p) {
  const double logc =
      std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
  return std::exp(logc + k * std::log(p) + (n - k) * std::log1p(-p));
}
struct RouletteOdds {
  double win;   // P(net gain >= threshold)
  double lose;  // P(net loss >= threshold)
};
// Exact: sum the pmf over the wins that reach the threshold.
[[nodiscard]] inline RouletteOdds Ex08Exact(int n_bets = 100,
                                            double bet = 100.0,
                                            double threshold = 1000.0,
                                            double p = kEx08PBlack) {
  RouletteOdds r{0.0, 0.0};
  for (int k = 0; k <= n_bets; ++k) {
    const double gain = bet * (2.0 * k - n_bets);
    if (gain >= threshold) r.win += BinomialPmf(k, n_bets, p);
    if (gain <= -threshold) r.lose += BinomialPmf(k, n_bets, p);
  }
  return r;
}
// Normal approximation; `continuity` moves the threshold half a step (one bet)
// inward.
[[nodiscard]] inline RouletteOdds Ex08Normal(bool continuity = false,
                                             int n_bets = 100,
                                             double bet = 100.0,
                                             double threshold = 1000.0,
                                             double p = kEx08PBlack) {
  const double n = n_bets;
  const double mean = n * bet * (2.0 * p - 1.0);
  const double sd = bet * std::sqrt(n * 4.0 * p * (1.0 - p));
  const double t = continuity ? threshold - bet : threshold;
  return {NormCdf((mean - t) / sd), NormCdf((-t - mean) / sd)};
}

}  // namespace p4mfe::ch04::ex
