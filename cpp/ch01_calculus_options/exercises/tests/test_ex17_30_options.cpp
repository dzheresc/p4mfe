#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <p4mfe/ch01/exercises/ex17_put_call_forward.hpp>
#include <p4mfe/ch01/exercises/ex18_21_spreads.hpp>
#include <p4mfe/ch01/exercises/ex22_24_parity_arbitrage.hpp>
#include <p4mfe/ch01/exercises/ex25_27_trading_strategies.hpp>
#include <p4mfe/ch01/exercises/ex28_29_call_convexity.hpp>
#include <p4mfe/ch01/exercises/ex30_futures_arbitrage.hpp>

using namespace p4mfe::ch01;
using namespace p4mfe::ch01::ex;

namespace {
// Grid of terminal prices covering every strike used below.
template <typename F>
void ForGrid(F&& f, double lo = 0.0, double hi = 200.0, double step = 0.25) {
  for (double s = lo; s <= hi; s += step) f(s);
}
}  // namespace

TEST(Ch01Ex17, CallEqualsPutAtForward) {
  const double spot = 50, maturity = 0.75, r = 0.04, q = 0.01;
  const double forward = Ex17EqualValueStrike(spot, maturity, r, q);
  EXPECT_NEAR(forward, spot * std::exp((r - q) * maturity), 1e-12);
  // With any call price C, parity gives P; P == C exactly at K = F.
  const double call = 6.0;
  EXPECT_NEAR(PutFromCall(call, spot, forward, maturity, r, q), call, 1e-12);
  for (double strike : {forward - 5, forward + 5}) {
    const double put = PutFromCall(call, spot, strike, maturity, r, q);
    EXPECT_NEAR(call - put, Ex17CallMinusPut(spot, strike, maturity, r, q),
                1e-12);
    EXPECT_EQ(call > put, strike < forward);
  }
}

TEST(Ch01Ex18, Butterfly) {
  const auto p = Ex18Butterfly();
  EXPECT_DOUBLE_EQ(p.Payoff(20), 0);
  EXPECT_DOUBLE_EQ(p.Payoff(32), 2);
  EXPECT_DOUBLE_EQ(p.Payoff(35), 5);
  EXPECT_DOUBLE_EQ(p.Payoff(38), 2);
  EXPECT_DOUBLE_EQ(p.Payoff(60), 0);
  ForGrid([&](double s) {
    EXPECT_GE(p.Payoff(s), 0.0);
    EXPECT_LE(p.Payoff(s), 5.0);
  });
  const auto rows = PayoffTable(p, 25, 45, 20);
  ASSERT_EQ(rows.size(), 21u);
  EXPECT_DOUBLE_EQ(rows[10].first, 35.0);
  EXPECT_DOUBLE_EQ(rows[10].second, 5.0);
}

TEST(Ch01Ex19, BullAndBearSpreads) {
  const auto bull = Ex19BullSpread();
  const auto bear = Ex19BearSpread();
  ForGrid([&](double s) {
    EXPECT_DOUBLE_EQ(bull.Payoff(s), std::clamp(s - 30.0, 0.0, 5.0));
    EXPECT_DOUBLE_EQ(bear.Payoff(s), std::clamp(20.0 - s, 0.0, 5.0));
  });
}

TEST(Ch01Ex20, ButterflyPnL) {
  const auto p = Ex20Butterfly();
  EXPECT_DOUBLE_EQ(p.Cost(), 1.0);
  EXPECT_DOUBLE_EQ(p.Payoff(50), 5.0);
  EXPECT_DOUBLE_EQ(p.Pnl(50), 4.0);
  EXPECT_DOUBLE_EQ(p.Pnl(40), -1.0);
  EXPECT_DOUBLE_EQ(p.Pnl(kEx20BreakevenLow), 0.0);
  EXPECT_DOUBLE_EQ(p.Pnl(kEx20BreakevenHigh), 0.0);
  ForGrid([&](double s) {
    const bool inside = kEx20BreakevenLow < s && s < kEx20BreakevenHigh;
    EXPECT_EQ(p.Pnl(s) > 0.0, inside) << s;
  });
}

TEST(Ch01Ex21, Portfolio1Dominates) {
  for (double x : {20.0, 50.0, 100.0}) {
    const auto p1 = Ex21Portfolio1(x);
    const auto p2 = Ex21Portfolio2(x);
    ForGrid([&](double s) {
      const double d = p1.Payoff(s) - p2.Payoff(s);
      EXPECT_GE(d, 0.0);
      EXPECT_EQ(d > 0.0, std::abs(s - x) < 5.0) << x << ' ' << s;
    });
  }
}

TEST(Ch01Ex22, ParityArbitrageWithDividends) {
  const Ex22Data d;
  EXPECT_NEAR(
      ParityViolation(d.call, d.put, d.spot, d.strike, d.maturity, d.r, d.q),
      -0.2432, 1e-4);
  const auto a = Ex22Arbitrage();
  ASSERT_TRUE(a.has_value());
  EXPECT_DOUBLE_EQ(a->calls, 1.0);  // buy the cheap call
  EXPECT_DOUBLE_EQ(a->puts, -1.0);  // sell the put
  EXPECT_NEAR(a->shares, -std::exp(-0.01), 1e-15);
  EXPECT_NEAR(a->cash, 39.5821, 1e-4);
  EXPECT_NEAR(a->profit, 0.2473, 1e-4);
  // Riskless: the payoff at T is the same for every S(T).
  for (double spot_t : {10.0, 40.0, 70.0}) {
    const double v = a->calls * CallPayoff(spot_t, d.strike) +
                     a->puts * PutPayoff(spot_t, d.strike) +
                     a->shares * std::exp(d.q * d.maturity) * spot_t +
                     a->cash * std::exp(d.r * d.maturity);
    EXPECT_NEAR(v, a->profit, 1e-10);
  }
}

TEST(Ch01Ex23, NoArbitrageWithSpreads) {
  EXPECT_FALSE(Ex23Arbitrage().has_value());
  // Mid prices alone would show a parity violation of 0.25.
  EXPECT_NEAR(ParityViolation(5.25, 3.0, 42, 40, 0.5, 0.0), 0.25, 1e-12);
}

TEST(Ch01Ex24, BidAskConditions) {
  const double spot = 100, strike = 95, maturity = 1, r = 0.05, q = 0.02;
  const double fwd =
      spot * std::exp(-q * maturity) - strike * std::exp(-r * maturity);
  // Inside the band: no arbitrage.
  EXPECT_FALSE(Ex24BidAskArbitrage({10.0, 11.0},
                                   {10.0 - fwd - 0.5, 11.0 - fwd + 0.5}, spot,
                                   strike, maturity, r, q));
  // Call ask too low: buy call, sell put.
  auto a =
      Ex24BidAskArbitrage({8.0, 8.5}, {3.0, 3.2}, spot, strike, maturity, r, q);
  ASSERT_TRUE(a);
  EXPECT_EQ(a->side, BidAskParityArbitrage::Side::kBuyCallSellPut);
  EXPECT_NEAR(a->profit, (fwd - (8.5 - 3.0)) * std::exp(r * maturity), 1e-12);
  // Call bid too high: sell call, buy put.
  auto b = Ex24BidAskArbitrage({12.0, 12.5}, {1.0, 1.2}, spot, strike, maturity,
                               r, q);
  ASSERT_TRUE(b);
  EXPECT_EQ(b->side, BidAskParityArbitrage::Side::kSellCallBuyPut);
  EXPECT_NEAR(b->profit, ((12.0 - 1.2) - fwd) * std::exp(r * maturity), 1e-12);
  // Zero spreads reduce to plain parity.
  const double call = 9.0,
               put = PutFromCall(call, spot, strike, maturity, r, q);
  EXPECT_FALSE(Ex24BidAskArbitrage({call, call}, {put, put}, spot, strike,
                                   maturity, r, q, 1e-9));
}

TEST(Ch01Ex25, StockVersusCalls) {
  EXPECT_NEAR(kEx25Crossover, 1225.0 / 31.0, 1e-12);
  EXPECT_NEAR(Ex25StockReturn(kEx25Crossover), Ex25CallReturn(kEx25Crossover),
              1e-12);
  EXPECT_NEAR(Ex25StockReturn(40), 5.0 / 35.0, 1e-12);
  EXPECT_NEAR(Ex25CallReturn(40), 0.25, 1e-12);
  EXPECT_NEAR(Ex25CallReturn(45), 1.5, 1e-12);
  EXPECT_DOUBLE_EQ(Ex25CallReturn(30), -1.0);
  for (double s = 40; s <= 45; s += 0.5)
    EXPECT_GT(Ex25CallReturn(s), Ex25StockReturn(s));
  for (double s = 20; s < kEx25Crossover; s += 0.5)
    EXPECT_GT(Ex25StockReturn(s), Ex25CallReturn(s));
}

TEST(Ch01Ex26, ReplicatePiecewiseLinear) {
  const auto p = Ex26Portfolio();
  EXPECT_DOUBLE_EQ(p.cash, 0.0);
  EXPECT_DOUBLE_EQ(p.shares, 2.0);
  ASSERT_EQ(p.options.size(), 2u);
  EXPECT_DOUBLE_EQ(p.options[0].strike, 20.0);
  EXPECT_DOUBLE_EQ(p.options[0].quantity, -3.0);
  EXPECT_DOUBLE_EQ(p.options[1].strike, 40.0);
  EXPECT_DOUBLE_EQ(p.options[1].quantity, 2.0);
  ForGrid(
      [&](double s) { EXPECT_NEAR(p.Payoff(s), Ex26Target(s), 1e-12) << s; });
  // The generic replicator also rebuilds the exercise 18 butterfly.
  const double slopes[] = {0.0, 1.0, -1.0, 0.0};
  const double kinks[] = {30.0, 35.0, 40.0};
  const auto b = ReplicatePiecewiseLinear(0.0, slopes, kinks);
  ForGrid([&](double s) {
    EXPECT_DOUBLE_EQ(b.Payoff(s), Ex18Butterfly().Payoff(s));
  });
}

TEST(Ch01Ex27, RangeDigital) {
  const double c = 10, strike1 = 40, strike2 = 60;
  for (double eps : {1.0, 0.1, 0.01}) {
    const auto p = Ex27RangeDigital(c, strike1, strike2, eps);
    ForGrid(
        [&](double s) {
          const double v = p.Payoff(s);
          EXPECT_GE(v, -1e-9);
          EXPECT_LE(v, c + 1e-9);
          const bool on_ramp = (s > strike1 && s < strike1 + eps) ||
                               (s > strike2 - eps && s < strike2);
          if (!on_ramp && s != strike1 && s != strike2) {
            EXPECT_NEAR(v, Ex27Target(s, c, strike1, strike2), 1e-9) << s;
          }
        },
        0.0, 100.0, 0.005);
  }
  EXPECT_THROW((void)Ex27RangeDigital(c, strike1, strike2, 15.0),
               std::invalid_argument);
}

TEST(Ch01Ex28, ButterflyArbitrage) {
  const auto x = Ex28Data();
  EXPECT_NEAR(Ex29ConvexityGap(x), 8.0 / 3.0 + 2.0 - 5.0, 1e-12);
  EXPECT_FALSE(Ex29NoArbitrage(x));
  const auto p = Ex29Arbitrage(x);
  ASSERT_TRUE(p);
  ASSERT_EQ(p->options.size(), 3u);
  EXPECT_DOUBLE_EQ(p->options[0].quantity, 1.0);
  EXPECT_DOUBLE_EQ(p->options[1].quantity, -3.0);
  EXPECT_DOUBLE_EQ(p->options[2].quantity, 2.0);
  EXPECT_DOUBLE_EQ(p->Cost(), -1.0);  // $1 collected up front
  EXPECT_DOUBLE_EQ(p->Payoff(120), 20.0);
  ForGrid([&](double s) { EXPECT_GE(p->Payoff(s), 0.0); });
}

TEST(Ch01Ex29, ConvexityCondition) {
  const ThreeCalls ok{{100, 110, 120}, {12, 7, 3}};  // 7 <= (12 + 3)/2
  EXPECT_TRUE(Ex29NoArbitrage(ok));
  EXPECT_FALSE(Ex29Arbitrage(ok));
  const ThreeCalls bad{{100, 110, 120}, {12, 8, 3}};  // 8 > 7.5
  EXPECT_FALSE(Ex29NoArbitrage(bad));
  const auto p = Ex29Arbitrage(bad);
  ASSERT_TRUE(p);
  EXPECT_NEAR(p->Cost(), -1.0, 1e-12);  // weights (1, -2, 1): 12 - 16 + 3
  ForGrid([&](double s) { EXPECT_GE(p->Payoff(s), -1e-12); });
  // Edge case on the boundary is not an arbitrage.
  EXPECT_TRUE(Ex29NoArbitrage({{100, 110, 120}, {12, 7.5, 3}}));
  // Bull-spread edge: C2 < C3 is also arbitrage.
  const auto q = Ex29Arbitrage({{100, 110, 120}, {12, 2, 3}});
  ASSERT_TRUE(q);
  EXPECT_LT(q->Cost(), 0.0);
  ForGrid([&](double s) { EXPECT_GE(q->Payoff(s), 0.0); });
}

TEST(Ch01Ex30, IndexFuturesArbitrage) {
  const auto a = Ex30Arbitrage();
  ASSERT_TRUE(a);
  EXPECT_DOUBLE_EQ(a->futures, 1.0);  // futures are cheap: buy them
  EXPECT_NEAR(a->units, -std::exp(-0.0075), 1e-15);
  EXPECT_NEAR(a->cash, 11910.34, 1e-2);
  EXPECT_NEAR(a->profit, 50.94, 1e-2);
  // Riskless: lent cash + short index (grown to one unit) + long futures.
  for (double spot_t : {10000.0, 12000.0, 14000.0}) {
    const double v = a->cash * std::exp(0.08 * 0.25) +
                     a->units * std::exp(0.03 * 0.25) * spot_t +
                     (spot_t - 12100.0);
    EXPECT_NEAR(v, a->profit, 1e-8);
  }
  // Rich futures: the reverse trade.
  const auto b = FindFuturesArbitrage(12000.0, 12200.0, 0.25, 0.08, 0.03);
  ASSERT_TRUE(b);
  EXPECT_DOUBLE_EQ(b->futures, -1.0);
  EXPECT_GT(b->units, 0.0);
  EXPECT_NEAR(b->profit, 12200.0 - 12000.0 * std::exp(0.0125), 1e-9);
  EXPECT_FALSE(FindFuturesArbitrage(12000.0, 12000.0 * std::exp(0.0125), 0.25,
                                    0.08, 0.03));
}
