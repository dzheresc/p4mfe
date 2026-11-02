#pragma once
// Chapter 1, sections 1.7-1.9: plain vanilla European options, arbitrage-free
// pricing, Put-Call parity and the arbitrage it rules out.
//
// Conventions: constant risk-free rate r and continuous dividend yield q,
// both continuously compounded; T is the time to maturity in years.

#include <algorithm>
#include <cmath>
#include <optional>

namespace p4mfe::ch01 {

// ---------------------------------------------------------------- 1.7 ------
enum class OptionType { kCall, kPut };

// Payoff at maturity: max(S - K, 0) for a call, max(K - S, 0) for a put.
[[nodiscard]] inline double CallPayoff(double spot, double strike) noexcept {
  return std::max(spot - strike, 0.0);
}
[[nodiscard]] inline double PutPayoff(double spot, double strike) noexcept {
  return std::max(strike - spot, 0.0);
}
[[nodiscard]] inline double Payoff(OptionType type, double spot,
                                   double strike) noexcept {
  return type == OptionType::kCall ? CallPayoff(spot, strike)
                                   : PutPayoff(spot, strike);
}

// ---------------------------------------------------------------- 1.8 ------
// Value at time 0 of one unit paid at time T: e^(-r T). A riskless position
// worth V at T is worth V e^(-r T) today, otherwise there is an arbitrage.
[[nodiscard]] inline double DiscountFactor(double r, double maturity) noexcept {
  return std::exp(-r * maturity);
}

struct PriceBounds {
  double lower;
  double upper;
};

// Model-free no-arbitrage bounds:
//   max(S e^(-qT) - K e^(-rT), 0) <= C <= S e^(-qT)
//   max(K e^(-rT) - S e^(-qT), 0) <= P <= K e^(-rT)
[[nodiscard]] inline PriceBounds CallBounds(double spot, double strike,
                                            double maturity, double r,
                                            double q = 0.0) {
  const double s = spot * std::exp(-q * maturity);
  const double k = strike * DiscountFactor(r, maturity);
  return {std::max(s - k, 0.0), s};
}
[[nodiscard]] inline PriceBounds PutBounds(double spot, double strike,
                                           double maturity, double r,
                                           double q = 0.0) {
  const double s = spot * std::exp(-q * maturity);
  const double k = strike * DiscountFactor(r, maturity);
  return {std::max(k - s, 0.0), k};
}

// ---------------------------------------------------------------- 1.9 ------
// Put-Call parity for European options with the same strike and maturity:
//   P + S e^(-qT) - C = K e^(-rT)
[[nodiscard]] inline double PutFromCall(double call, double spot, double strike,
                                        double maturity, double r,
                                        double q = 0.0) {
  return call - spot * std::exp(-q * maturity) +
         strike * DiscountFactor(r, maturity);
}
[[nodiscard]] inline double CallFromPut(double put, double spot, double strike,
                                        double maturity, double r,
                                        double q = 0.0) {
  return put + spot * std::exp(-q * maturity) -
         strike * DiscountFactor(r, maturity);
}

// Parity violation C - P - S e^(-qT) + K e^(-rT); zero when prices are
// consistent, positive when the call is rich relative to the put.
[[nodiscard]] inline double ParityViolation(double call, double put,
                                            double spot, double strike,
                                            double maturity, double r,
                                            double q = 0.0) {
  return call - put - spot * std::exp(-q * maturity) +
         strike * DiscountFactor(r, maturity);
}

// ------------------------------------------------------------- 1.9.1 -------
// A zero-cost portfolio that locks in a riskless profit when parity fails.
// Positions are signed: +1 long, -1 short.
struct ParityArbitrage {
  double calls;   // number of calls held
  double puts;    // number of puts held
  double shares;  // shares held at time 0 (dividends reinvested -> |shares|
                  // e^(qT) = 1 at T)
  double cash;    // cash at time 0, invested (+) or borrowed (-) at rate r
  double profit;  // riskless value of the portfolio at maturity T
};

// If |violation| > tol, returns the arbitrage portfolio, else std::nullopt.
// Call rich (violation > 0): sell the call, buy the put, buy e^(-qT) shares,
// invest the remaining cash. At T the options and stock are worth K whatever
// S_T is, so the profit is violation * e^(rT). Cheap call: the reverse.
[[nodiscard]] inline std::optional<ParityArbitrage> FindParityArbitrage(
    double call, double put, double spot, double strike, double maturity,
    double r, double q = 0.0, double tol = 1e-12) {
  const double v = ParityViolation(call, put, spot, strike, maturity, r, q);
  if (std::abs(v) <= tol) return std::nullopt;
  const double sign = v > 0.0 ? 1.0 : -1.0;
  const double shares = sign * std::exp(-q * maturity);
  return ParityArbitrage{
      .calls = -sign,
      .puts = sign,
      .shares = shares,
      .cash = sign * (call - put) - shares * spot,
      .profit = std::abs(v) * std::exp(r * maturity),
  };
}

}  // namespace p4mfe::ch01
