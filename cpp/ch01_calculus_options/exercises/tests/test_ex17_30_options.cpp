#include <p4mfe/ch01/exercises/ex17_put_call_forward.hpp>
#include <p4mfe/ch01/exercises/ex18_21_spreads.hpp>
#include <p4mfe/ch01/exercises/ex22_24_parity_arbitrage.hpp>
#include <p4mfe/ch01/exercises/ex25_27_trading_strategies.hpp>
#include <p4mfe/ch01/exercises/ex28_29_call_convexity.hpp>
#include <p4mfe/ch01/exercises/ex30_futures_arbitrage.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using namespace p4mfe::ch01;
using namespace p4mfe::ch01::ex;

namespace {
// Grid of terminal prices covering every strike used below.
template <typename F>
void for_grid(F&& f, double lo = 0.0, double hi = 200.0, double step = 0.25) {
    for (double s = lo; s <= hi; s += step) f(s);
}
}  // namespace

TEST(Ch01Ex17, CallEqualsPutAtForward) {
    const double S = 50, T = 0.75, r = 0.04, q = 0.01;
    const double F = ex17_equal_value_strike(S, T, r, q);
    EXPECT_NEAR(F, S * std::exp((r - q) * T), 1e-12);
    // With any call price C, parity gives P; P == C exactly at K = F.
    const double C = 6.0;
    EXPECT_NEAR(put_from_call(C, S, F, T, r, q), C, 1e-12);
    for (double K : {F - 5, F + 5}) {
        const double P = put_from_call(C, S, K, T, r, q);
        EXPECT_NEAR(C - P, ex17_call_minus_put(S, K, T, r, q), 1e-12);
        EXPECT_EQ(C > P, K < F);
    }
}

TEST(Ch01Ex18, Butterfly) {
    const auto p = ex18_butterfly();
    EXPECT_DOUBLE_EQ(p.payoff(20), 0);
    EXPECT_DOUBLE_EQ(p.payoff(32), 2);
    EXPECT_DOUBLE_EQ(p.payoff(35), 5);
    EXPECT_DOUBLE_EQ(p.payoff(38), 2);
    EXPECT_DOUBLE_EQ(p.payoff(60), 0);
    for_grid([&](double s) {
        EXPECT_GE(p.payoff(s), 0.0);
        EXPECT_LE(p.payoff(s), 5.0);
    });
    const auto rows = payoff_table(p, 25, 45, 20);
    ASSERT_EQ(rows.size(), 21u);
    EXPECT_DOUBLE_EQ(rows[10].first, 35.0);
    EXPECT_DOUBLE_EQ(rows[10].second, 5.0);
}

TEST(Ch01Ex19, BullAndBearSpreads) {
    const auto bull = ex19_bull_spread();
    const auto bear = ex19_bear_spread();
    for_grid([&](double s) {
        EXPECT_DOUBLE_EQ(bull.payoff(s), std::clamp(s - 30.0, 0.0, 5.0));
        EXPECT_DOUBLE_EQ(bear.payoff(s), std::clamp(20.0 - s, 0.0, 5.0));
    });
}

TEST(Ch01Ex20, ButterflyPnL) {
    const auto p = ex20_butterfly();
    EXPECT_DOUBLE_EQ(p.cost(), 1.0);
    EXPECT_DOUBLE_EQ(p.payoff(50), 5.0);
    EXPECT_DOUBLE_EQ(p.pnl(50), 4.0);
    EXPECT_DOUBLE_EQ(p.pnl(40), -1.0);
    EXPECT_DOUBLE_EQ(p.pnl(ex20_breakeven_low), 0.0);
    EXPECT_DOUBLE_EQ(p.pnl(ex20_breakeven_high), 0.0);
    for_grid([&](double s) {
        const bool inside = ex20_breakeven_low < s && s < ex20_breakeven_high;
        EXPECT_EQ(p.pnl(s) > 0.0, inside) << s;
    });
}

TEST(Ch01Ex21, Portfolio1Dominates) {
    for (double X : {20.0, 50.0, 100.0}) {
        const auto p1 = ex21_portfolio1(X);
        const auto p2 = ex21_portfolio2(X);
        for_grid([&](double s) {
            const double d = p1.payoff(s) - p2.payoff(s);
            EXPECT_GE(d, 0.0);
            EXPECT_EQ(d > 0.0, std::abs(s - X) < 5.0) << X << ' ' << s;
        });
    }
}

TEST(Ch01Ex22, ParityArbitrageWithDividends) {
    const Ex22Data d;
    EXPECT_NEAR(parity_violation(d.C, d.P, d.S, d.K, d.T, d.r, d.q), -0.2432, 1e-4);
    const auto a = ex22_arbitrage();
    ASSERT_TRUE(a.has_value());
    EXPECT_DOUBLE_EQ(a->calls, 1.0);   // buy the cheap call
    EXPECT_DOUBLE_EQ(a->puts, -1.0);   // sell the put
    EXPECT_NEAR(a->shares, -std::exp(-0.01), 1e-15);
    EXPECT_NEAR(a->cash, 39.5821, 1e-4);
    EXPECT_NEAR(a->profit, 0.2473, 1e-4);
    // Riskless: the payoff at T is the same for every S(T).
    for (double ST : {10.0, 40.0, 70.0}) {
        const double v = a->calls * call_payoff(ST, d.K) + a->puts * put_payoff(ST, d.K) +
                         a->shares * std::exp(d.q * d.T) * ST + a->cash * std::exp(d.r * d.T);
        EXPECT_NEAR(v, a->profit, 1e-10);
    }
}

TEST(Ch01Ex23, NoArbitrageWithSpreads) {
    EXPECT_FALSE(ex23_arbitrage().has_value());
    // Mid prices alone would show a parity violation of 0.25.
    EXPECT_NEAR(parity_violation(5.25, 3.0, 42, 40, 0.5, 0.0), 0.25, 1e-12);
}

TEST(Ch01Ex24, BidAskConditions) {
    const double S = 100, K = 95, T = 1, r = 0.05, q = 0.02;
    const double fwd = S * std::exp(-q * T) - K * std::exp(-r * T);
    // Inside the band: no arbitrage.
    EXPECT_FALSE(ex24_bid_ask_arbitrage({10.0, 11.0}, {10.0 - fwd - 0.5, 11.0 - fwd + 0.5}, S, K, T, r, q));
    // Call ask too low: buy call, sell put.
    auto a = ex24_bid_ask_arbitrage({8.0, 8.5}, {3.0, 3.2}, S, K, T, r, q);
    ASSERT_TRUE(a);
    EXPECT_EQ(a->side, BidAskParityArbitrage::Side::BuyCallSellPut);
    EXPECT_NEAR(a->profit, (fwd - (8.5 - 3.0)) * std::exp(r * T), 1e-12);
    // Call bid too high: sell call, buy put.
    auto b = ex24_bid_ask_arbitrage({12.0, 12.5}, {1.0, 1.2}, S, K, T, r, q);
    ASSERT_TRUE(b);
    EXPECT_EQ(b->side, BidAskParityArbitrage::Side::SellCallBuyPut);
    EXPECT_NEAR(b->profit, ((12.0 - 1.2) - fwd) * std::exp(r * T), 1e-12);
    // Zero spreads reduce to plain parity.
    const double C = 9.0, P = put_from_call(C, S, K, T, r, q);
    EXPECT_FALSE(ex24_bid_ask_arbitrage({C, C}, {P, P}, S, K, T, r, q, 1e-9));
}

TEST(Ch01Ex25, StockVersusCalls) {
    EXPECT_NEAR(ex25_crossover, 1225.0 / 31.0, 1e-12);
    EXPECT_NEAR(ex25_stock_return(ex25_crossover), ex25_call_return(ex25_crossover), 1e-12);
    EXPECT_NEAR(ex25_stock_return(40), 5.0 / 35.0, 1e-12);
    EXPECT_NEAR(ex25_call_return(40), 0.25, 1e-12);
    EXPECT_NEAR(ex25_call_return(45), 1.5, 1e-12);
    EXPECT_DOUBLE_EQ(ex25_call_return(30), -1.0);
    for (double s = 40; s <= 45; s += 0.5) EXPECT_GT(ex25_call_return(s), ex25_stock_return(s));
    for (double s = 20; s < ex25_crossover; s += 0.5) EXPECT_GT(ex25_stock_return(s), ex25_call_return(s));
}

TEST(Ch01Ex26, ReplicatePiecewiseLinear) {
    const auto p = ex26_portfolio();
    EXPECT_DOUBLE_EQ(p.cash, 0.0);
    EXPECT_DOUBLE_EQ(p.shares, 2.0);
    ASSERT_EQ(p.options.size(), 2u);
    EXPECT_DOUBLE_EQ(p.options[0].strike, 20.0);
    EXPECT_DOUBLE_EQ(p.options[0].quantity, -3.0);
    EXPECT_DOUBLE_EQ(p.options[1].strike, 40.0);
    EXPECT_DOUBLE_EQ(p.options[1].quantity, 2.0);
    for_grid([&](double s) { EXPECT_NEAR(p.payoff(s), ex26_target(s), 1e-12) << s; });
    // The generic replicator also rebuilds the exercise 18 butterfly.
    const double slopes[] = {0.0, 1.0, -1.0, 0.0};
    const double kinks[] = {30.0, 35.0, 40.0};
    const auto b = replicate_piecewise_linear(0.0, slopes, kinks);
    for_grid([&](double s) { EXPECT_DOUBLE_EQ(b.payoff(s), ex18_butterfly().payoff(s)); });
}

TEST(Ch01Ex27, RangeDigital) {
    const double c = 10, K1 = 40, K2 = 60;
    for (double eps : {1.0, 0.1, 0.01}) {
        const auto p = ex27_range_digital(c, K1, K2, eps);
        for_grid([&](double s) {
            const double v = p.payoff(s);
            EXPECT_GE(v, -1e-9);
            EXPECT_LE(v, c + 1e-9);
            const bool on_ramp = (s > K1 && s < K1 + eps) || (s > K2 - eps && s < K2);
            if (!on_ramp && s != K1 && s != K2) {
                EXPECT_NEAR(v, ex27_target(s, c, K1, K2), 1e-9) << s;
            }
        }, 0.0, 100.0, 0.005);
    }
    EXPECT_THROW((void)ex27_range_digital(c, K1, K2, 15.0), std::invalid_argument);
}

TEST(Ch01Ex28, ButterflyArbitrage) {
    const auto x = ex28_data();
    EXPECT_NEAR(ex29_convexity_gap(x), 8.0 / 3.0 + 2.0 - 5.0, 1e-12);
    EXPECT_FALSE(ex29_no_arbitrage(x));
    const auto p = ex29_arbitrage(x);
    ASSERT_TRUE(p);
    ASSERT_EQ(p->options.size(), 3u);
    EXPECT_DOUBLE_EQ(p->options[0].quantity, 1.0);
    EXPECT_DOUBLE_EQ(p->options[1].quantity, -3.0);
    EXPECT_DOUBLE_EQ(p->options[2].quantity, 2.0);
    EXPECT_DOUBLE_EQ(p->cost(), -1.0);  // $1 collected up front
    EXPECT_DOUBLE_EQ(p->payoff(120), 20.0);
    for_grid([&](double s) { EXPECT_GE(p->payoff(s), 0.0); });
}

TEST(Ch01Ex29, ConvexityCondition) {
    const ThreeCalls ok{{100, 110, 120}, {12, 7, 3}};  // 7 <= (12 + 3)/2
    EXPECT_TRUE(ex29_no_arbitrage(ok));
    EXPECT_FALSE(ex29_arbitrage(ok));
    const ThreeCalls bad{{100, 110, 120}, {12, 8, 3}};  // 8 > 7.5
    EXPECT_FALSE(ex29_no_arbitrage(bad));
    const auto p = ex29_arbitrage(bad);
    ASSERT_TRUE(p);
    EXPECT_NEAR(p->cost(), -1.0, 1e-12);  // weights (1, -2, 1): 12 - 16 + 3
    for_grid([&](double s) { EXPECT_GE(p->payoff(s), -1e-12); });
    // Edge case on the boundary is not an arbitrage.
    EXPECT_TRUE(ex29_no_arbitrage({{100, 110, 120}, {12, 7.5, 3}}));
    // Bull-spread edge: C2 < C3 is also arbitrage.
    const auto q = ex29_arbitrage({{100, 110, 120}, {12, 2, 3}});
    ASSERT_TRUE(q);
    EXPECT_LT(q->cost(), 0.0);
    for_grid([&](double s) { EXPECT_GE(q->payoff(s), 0.0); });
}

TEST(Ch01Ex30, IndexFuturesArbitrage) {
    const auto a = ex30_arbitrage();
    ASSERT_TRUE(a);
    EXPECT_DOUBLE_EQ(a->futures, 1.0);  // futures are cheap: buy them
    EXPECT_NEAR(a->units, -std::exp(-0.0075), 1e-15);
    EXPECT_NEAR(a->cash, 11910.34, 1e-2);
    EXPECT_NEAR(a->profit, 50.94, 1e-2);
    // Riskless: lent cash + short index (grown to one unit) + long futures.
    for (double ST : {10000.0, 12000.0, 14000.0}) {
        const double v = a->cash * std::exp(0.08 * 0.25) + a->units * std::exp(0.03 * 0.25) * ST + (ST - 12100.0);
        EXPECT_NEAR(v, a->profit, 1e-8);
    }
    // Rich futures: the reverse trade.
    const auto b = futures_arbitrage(12000.0, 12200.0, 0.25, 0.08, 0.03);
    ASSERT_TRUE(b);
    EXPECT_DOUBLE_EQ(b->futures, -1.0);
    EXPECT_GT(b->units, 0.0);
    EXPECT_NEAR(b->profit, 12200.0 - 12000.0 * std::exp(0.0125), 1e-9);
    EXPECT_FALSE(futures_arbitrage(12000.0, 12000.0 * std::exp(0.0125), 0.25, 0.08, 0.03));
}
