#pragma once
// Exercises 22-24 (section 1.12): Put-Call parity arbitrage, with and without
// bid-ask spreads.

#include <cmath>
#include <optional>
#include <p4mfe/ch01/options.hpp>

namespace p4mfe::ch01::ex {

// ----------------------------------------------------------- Exercise 22 ---
// A stock with spot $42 pays dividends continuously at 3%. Four-month put
// and call options with strike 40 trade at $2 and $4. The risk-free rate is
// 5%. Show that Put-Call parity fails and explain how to take advantage.
//
// Solution. T = 1/3. Parity requires
//   C - P = S e^(-qT) - K e^(-rT) = 42 e^(-0.01) - 40 e^(-0.05/3)
//         = 41.5821 - 39.3389 = 2.2432,
// but C - P = 4 - 2 = 2. The call is cheap relative to the put (equivalently
// the put is rich): violation C - P - S e^(-qT) + K e^(-rT) = -0.2432.
// Trade at t = 0: buy 1 call (-4), sell 1 put (+2), short e^(-qT) = 0.99005
// shares (+41.5821), invest the 39.5821 balance at 5%. Dividends owed on the
// short are paid by shorting more shares, so exactly one share is short at T.
// At T the long call + short put is worth S(T) - K, i.e. we buy the share for
// K = 40 whatever S(T) is and close the short. Riskless profit at T:
//   39.5821 e^(0.05/3) - 40 = 0.2432 e^(rT) = $0.2473.
struct Ex22Data {
  double S = 42.0, K = 40.0, T = 4.0 / 12.0, r = 0.05, q = 0.03, C = 4.0,
         P = 2.0;
};
[[nodiscard]] inline std::optional<ParityArbitrage> ex22_arbitrage(
    const Ex22Data& d = {}) {
  return parity_arbitrage(d.C, d.P, d.S, d.K, d.T, d.r, d.q);
}

// ----------------------------------------------------------- Exercise 24 ---
// (Solved before 23, which is a special case.) C_bid, C_ask, P_bid, P_ask are
// bid and ask prices of a European call and put with the same strike K and
// maturity T on an asset with spot S and dividend yield q; constant rate r.
// Find necessary and sufficient no-arbitrage conditions.
//
// Solution. Only two parity trades exist, and each must use the side of the
// market it trades at (buy at ask, sell at bid):
//   A. Buy the call (C_ask), sell the put (P_bid), short e^(-qT) shares,
//      invest the cash. Riskless profit at T:
//        e^(rT) [ S e^(-qT) - K e^(-rT) - (C_ask - P_bid) ].
//   B. Sell the call (C_bid), buy the put (P_ask), buy e^(-qT) shares,
//      borrow the cash. Riskless profit at T:
//        e^(rT) [ (C_bid - P_ask) - (S e^(-qT) - K e^(-rT)) ].
// No arbitrage  <=>  neither profit is positive  <=>
//      C_bid - P_ask  <=  S e^(-qT) - K e^(-rT)  <=  C_ask - P_bid,
// equivalently  P_bid + S e^(-qT) - K e^(-rT) <= C_ask  and
//               C_bid <= P_ask + S e^(-qT) - K e^(-rT).
// (The asset trades at S and cash is lent and borrowed at r without spread.)
struct BidAsk {
  double bid;
  double ask;
};

struct BidAskParityArbitrage {
  enum class Side {
    BuyCallSellPut,  // strategy A: synthetic long forward is cheap
    SellCallBuyPut,  // strategy B: synthetic long forward is rich
  } side;
  double profit;  // riskless profit at T per one call/put pair
};

[[nodiscard]] inline std::optional<BidAskParityArbitrage>
ex24_bid_ask_arbitrage(BidAsk C, BidAsk P, double S, double K, double T,
                       double r, double q = 0.0, double tol = 1e-12) {
  const double fwd =
      S * std::exp(-q * T) - K * std::exp(-r * T);  // PV of (F - K)
  const double growth = std::exp(r * T);
  if (const double a = fwd - (C.ask - P.bid); a > tol)
    return BidAskParityArbitrage{BidAskParityArbitrage::Side::BuyCallSellPut,
                                 a * growth};
  if (const double b = (C.bid - P.ask) - fwd; b > tol)
    return BidAskParityArbitrage{BidAskParityArbitrage::Side::SellCallBuyPut,
                                 b * growth};
  return std::nullopt;
}

// ----------------------------------------------------------- Exercise 23 ---
// Six-month European call, strike 40, on a non-dividend-paying stock at $42:
// bid $5, ask $5.5. Put with the same strike and maturity: bid $2.75, ask
// $3.25. r = 0. Is there an arbitrage?
//
// Solution. S - K e^(-rT) = 42 - 40 = 2, and
//   C_bid - P_ask = 5 - 3.25 = 1.75 <= 2 <= 2.75 = 5.5 - 2.75 = C_ask - P_bid,
// so by exercise 24 there is no arbitrage. (With mid prices, 5.25 - 3 = 2.25
// != 2 looks like a violation, but the spreads eat it.)
[[nodiscard]] inline std::optional<BidAskParityArbitrage> ex23_arbitrage() {
  return ex24_bid_ask_arbitrage({5.0, 5.5}, {2.75, 3.25}, 42.0, 40.0, 0.5, 0.0);
}

}  // namespace p4mfe::ch01::ex
