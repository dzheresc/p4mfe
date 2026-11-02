#pragma once
// Exercise 30 (section 1.12): index futures arbitrage (cash-and-carry).
//
// The risk-free rate is 8% (continuous) and the dividend yield of a stock
// index is 3%. The index is at 12,000 and the futures price for delivery in
// three months is 12,100. Is there an arbitrage, and how do you exploit it?
//
// Solution. The no-arbitrage futures price (constant rates: futures = forward)
//   F = S e^((r - q) T) = 12000 e^(0.05 * 0.25) = 12000 e^(0.0125) = 12,150.94.
// The market price 12,100 is too low by 50.94: buy the cheap futures and sell
// the index ("reverse cash-and-carry"):
//   t = 0: go long one futures contract (no cost); short e^(-qT) = 0.99253
//          units of the index for 11,910.34 (dividends owed on the short are
//          covered by shorting more, so the short grows to 1 unit by T);
//          lend the 11,910.34 at 8%.
//   t = T: the loan returns 11,910.34 e^(0.02) = 12,150.94; pay 12,100 for one
//          unit of the index through the futures and close the short.
// Riskless profit at T: 12,150.94 - 12,100 = $50.94 per index unit.
// (If the market price were above F, do the opposite: short futures, buy
// e^(-qT) units of the index with borrowed money.)

#include <cmath>
#include <optional>
#include <p4mfe/ch01/forwards.hpp>

namespace p4mfe::ch01::ex {

struct FuturesArbitrage {
  double futures;  // +1 long, -1 short
  double
      units;  // index units at t = 0 (+ long, - short); |units| e^(qT) = 1 at T
  double cash;    // lent (+) or borrowed (-) at t = 0
  double profit;  // riskless profit at T
};

[[nodiscard]] inline std::optional<FuturesArbitrage> futures_arbitrage(
    double S, double F_market, double T, double r, double q = 0.0,
    double tol = 1e-9) {
  const double F = forward_price(S, T, r, q);
  const double gap = F - F_market;  // > 0: futures cheap
  if (std::abs(gap) <= tol) return std::nullopt;
  const double sign = gap > 0.0 ? 1.0 : -1.0;
  const double units = -sign * std::exp(-q * T);
  return FuturesArbitrage{sign, units, -units * S, std::abs(gap)};
}

[[nodiscard]] inline std::optional<FuturesArbitrage> ex30_arbitrage() {
  return futures_arbitrage(12000.0, 12100.0, 0.25, 0.08, 0.03);
}

}  // namespace p4mfe::ch01::ex
