import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.exercises.ex17_put_call_forward import (
    ex17_call_minus_put,
    ex17_equal_value_strike,
)
from p4mfe.ch01_calculus_options.exercises.ex18_21_spreads import (
    EX20_BREAKEVEN_HIGH,
    EX20_BREAKEVEN_LOW,
    ex18_butterfly,
    ex19_bear_spread,
    ex19_bull_spread,
    ex20_butterfly,
    ex21_portfolio1,
    ex21_portfolio2,
)
from p4mfe.ch01_calculus_options.exercises.ex22_24_parity_arbitrage import (
    BidAsk,
    Ex22Data,
    Side,
    ex22_arbitrage,
    ex23_arbitrage,
    ex24_bid_ask_arbitrage,
)
from p4mfe.ch01_calculus_options.exercises.ex25_27_trading_strategies import (
    EX25_CROSSOVER,
    ex25_call_return,
    ex25_stock_return,
    ex26_portfolio,
    ex26_target,
    ex27_range_digital,
    ex27_target,
    replicate_piecewise_linear,
)
from p4mfe.ch01_calculus_options.exercises.ex28_29_call_convexity import (
    ThreeCalls,
    ex28_data,
    ex29_arbitrage,
    ex29_convexity_gap,
    ex29_no_arbitrage,
)
from p4mfe.ch01_calculus_options.exercises.ex30_futures_arbitrage import ex30_arbitrage, futures_arbitrage
from p4mfe.ch01_calculus_options.exercises.portfolio import payoff_table
from p4mfe.ch01_calculus_options.options import call_payoff, parity_violation, put_from_call, put_payoff

GRID = np.arange(0.0, 200.25, 0.25)


def test_ex17():
    S, T, r, q = 50.0, 0.75, 0.04, 0.01
    F = ex17_equal_value_strike(S, T, r, q)
    assert put_from_call(6.0, S, F, T, r, q) == pytest.approx(6.0, abs=1e-12)
    for K in (F - 5, F + 5):
        P = put_from_call(6.0, S, K, T, r, q)
        assert 6.0 - P == pytest.approx(
            ex17_call_minus_put(S, K, T, r, q), abs=1e-12
        )
        assert (6.0 > P) == (K < F)


def test_ex18():
    p = ex18_butterfly()
    assert [p.payoff(s) for s in (20, 32, 35, 38, 60)] == [0, 2, 5, 2, 0]
    v = p.payoff(GRID)
    assert v.min() >= 0 and v.max() == 5
    table = payoff_table(p, 25, 45, 20)
    assert table.shape == (21, 2) and tuple(table[10]) == (35.0, 5.0)


def test_ex19():
    np.testing.assert_array_equal(
        ex19_bull_spread().payoff(GRID), np.clip(GRID - 30, 0, 5)
    )
    np.testing.assert_array_equal(
        ex19_bear_spread().payoff(GRID), np.clip(20 - GRID, 0, 5)
    )


def test_ex20():
    p = ex20_butterfly()
    assert p.cost() == 1.0
    assert p.pnl(50) == 4.0 and p.pnl(40) == -1.0
    assert (
        p.pnl(EX20_BREAKEVEN_LOW) == 0.0 and p.pnl(EX20_BREAKEVEN_HIGH) == 0.0
    )
    np.testing.assert_array_equal(p.pnl(GRID) > 0, (GRID > 46) & (GRID < 54))


@pytest.mark.parametrize("X", [20.0, 50.0, 100.0])
def test_ex21(X):
    d = ex21_portfolio1(X).payoff(GRID) - ex21_portfolio2(X).payoff(GRID)
    assert d.min() >= 0
    np.testing.assert_array_equal(d > 0, np.abs(GRID - X) < 5)


def test_ex22():
    d = Ex22Data()
    assert parity_violation(d.C, d.P, d.S, d.K, d.T, d.r, d.q) == pytest.approx(
        -0.2432, abs=1e-4
    )
    a = ex22_arbitrage()
    assert (a.calls, a.puts) == (1.0, -1.0)
    assert a.cash == pytest.approx(39.5821, abs=1e-4)
    assert a.profit == pytest.approx(0.2473, abs=1e-4)
    for ST in (10.0, 40.0, 70.0):
        v = (
            a.calls * call_payoff(ST, d.K)
            + a.puts * put_payoff(ST, d.K)
            + a.shares * math.exp(d.q * d.T) * ST
            + a.cash * math.exp(d.r * d.T)
        )
        assert v == pytest.approx(a.profit, abs=1e-10)


def test_ex23():
    assert ex23_arbitrage() is None
    assert parity_violation(5.25, 3.0, 42, 40, 0.5, 0.0) == pytest.approx(0.25)


def test_ex24():
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.02
    fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (
        ex24_bid_ask_arbitrage(
            BidAsk(10, 11),
            BidAsk(10 - fwd - 0.5, 11 - fwd + 0.5),
            S,
            K,
            T,
            r,
            q,
        )
        is None
    )
    a = ex24_bid_ask_arbitrage(
        BidAsk(8.0, 8.5), BidAsk(3.0, 3.2), S, K, T, r, q
    )
    assert a.side is Side.BUY_CALL_SELL_PUT
    assert a.profit == pytest.approx((fwd - 5.5) * math.exp(r * T))
    b = ex24_bid_ask_arbitrage(
        BidAsk(12.0, 12.5), BidAsk(1.0, 1.2), S, K, T, r, q
    )
    assert b.side is Side.SELL_CALL_BUY_PUT
    assert b.profit == pytest.approx((10.8 - fwd) * math.exp(r * T))


def test_ex25():
    assert EX25_CROSSOVER == pytest.approx(1225 / 31)
    assert ex25_stock_return(EX25_CROSSOVER) == pytest.approx(
        ex25_call_return(EX25_CROSSOVER)
    )
    assert ex25_call_return(40) == pytest.approx(0.25)
    assert ex25_call_return(45) == pytest.approx(1.5)
    s = np.arange(40, 45.5, 0.5)
    assert np.all(ex25_call_return(s) > ex25_stock_return(s))
    s = np.arange(20, EX25_CROSSOVER, 0.5)
    assert np.all(ex25_stock_return(s) > ex25_call_return(s))


def test_ex26():
    p = ex26_portfolio()
    assert (p.cash, p.shares) == (0.0, 2.0)
    assert [(leg.strike, leg.quantity) for leg in p.options] == [
        (20.0, -3.0),
        (40.0, 2.0),
    ]
    np.testing.assert_allclose(p.payoff(GRID), ex26_target(GRID), atol=1e-12)
    b = replicate_piecewise_linear(0.0, [0, 1, -1, 0], [30, 35, 40])
    np.testing.assert_array_equal(b.payoff(GRID), ex18_butterfly().payoff(GRID))


@pytest.mark.parametrize("eps", [1.0, 0.1, 0.01])
def test_ex27(eps):
    c, K1, K2 = 10.0, 40.0, 60.0
    s = np.arange(0.0, 100.0, 0.005)
    v = ex27_range_digital(c, K1, K2, eps).payoff(s)
    assert v.min() >= -1e-9 and v.max() <= c + 1e-9
    # Away from the two ramps (and the endpoints K1, K2, where the replica is 0) it is exact.
    off_ramp = ~(
        ((s >= K1 - 1e-9) & (s < K1 + eps))
        | ((s > K2 - eps) & (s <= K2 + 1e-9))
    )
    np.testing.assert_allclose(
        v[off_ramp], ex27_target(s, c, K1, K2)[off_ramp], atol=1e-9
    )
    with pytest.raises(ValueError):
        ex27_range_digital(c, K1, K2, 15.0)


def test_ex28():
    x = ex28_data()
    assert ex29_convexity_gap(x) == pytest.approx(8 / 3 + 2 - 5)
    assert not ex29_no_arbitrage(x)
    p = ex29_arbitrage(x)
    assert [leg.quantity for leg in p.options] == pytest.approx(
        [1.0, -3.0, 2.0]
    )
    assert p.cost() == pytest.approx(-1.0)
    assert p.payoff(120) == pytest.approx(20.0)
    assert p.payoff(GRID).min() >= -1e-12


def test_ex29():
    ok = ThreeCalls((100, 110, 120), (12, 7, 3))
    assert ex29_no_arbitrage(ok) and ex29_arbitrage(ok) is None
    assert ex29_no_arbitrage(ThreeCalls((100, 110, 120), (12, 7.5, 3)))
    bad = ThreeCalls((100, 110, 120), (12, 8, 3))
    p = ex29_arbitrage(bad)
    assert p.cost() == pytest.approx(-1.0)
    assert p.payoff(GRID).min() >= -1e-12
    q = ex29_arbitrage(ThreeCalls((100, 110, 120), (12, 2, 3)))
    assert q.cost() < 0 and q.payoff(GRID).min() >= 0


def test_ex30():
    a = ex30_arbitrage()
    assert a.futures == 1.0
    assert a.units == pytest.approx(-math.exp(-0.0075))
    assert a.cash == pytest.approx(11910.34, abs=1e-2)
    assert a.profit == pytest.approx(50.94, abs=1e-2)
    for ST in (10000.0, 12000.0, 14000.0):
        v = (
            a.cash * math.exp(0.02)
            + a.units * math.exp(0.0075) * ST
            + (ST - 12100.0)
        )
        assert v == pytest.approx(a.profit, abs=1e-8)
    b = futures_arbitrage(12000.0, 12200.0, 0.25, 0.08, 0.03)
    assert b.futures == -1.0 and b.units > 0
    assert (
        futures_arbitrage(12000.0, 12000.0 * math.exp(0.0125), 0.25, 0.08, 0.03)
        is None
    )
