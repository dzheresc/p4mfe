import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.forwards import (
    Dividend,
    forward_price,
    forward_price_discrete,
    long_forward_value,
    synthetic_forward_value,
)
from p4mfe.ch01_calculus_options.options import (
    OptionType,
    call_bounds,
    call_from_put,
    call_payoff,
    discount_factor,
    parity_arbitrage,
    parity_violation,
    payoff,
    put_bounds,
    put_from_call,
    put_payoff,
)

S, K, T, r, q, C = 42.0, 40.0, 0.5, 0.05, 0.02, 4.0


def test_payoffs():
    assert call_payoff(110, 100) == 10 and call_payoff(90, 100) == 0
    assert put_payoff(90, 100) == 10 and payoff(OptionType.PUT, 110, 100) == 0
    ST = np.array([50.0, 100.0, 150.0])
    np.testing.assert_allclose(
        call_payoff(ST, 100) - put_payoff(ST, 100), ST - 100
    )


def test_parity_round_trip():
    P = put_from_call(C, S, K, T, r, q)
    assert call_from_put(P, S, K, T, r, q) == pytest.approx(C, abs=1e-12)
    assert parity_violation(C, P, S, K, T, r, q) == pytest.approx(
        0.0, abs=1e-12
    )
    assert parity_arbitrage(C, P, S, K, T, r, q) is None


def test_bounds():
    P = put_from_call(C, S, K, T, r, q)
    cb, pb = call_bounds(S, K, T, r, q), put_bounds(S, K, T, r, q)
    assert cb.lower <= C <= cb.upper
    assert pb.lower <= P <= pb.upper


@pytest.mark.parametrize("C_, P_, q_", [(5.0, 1.0, 0.02), (2.0, 1.0, 0.0)])
def test_arbitrage_is_riskless(C_, P_, q_):
    arb = parity_arbitrage(C_, P_, S, K, T, r, q_)
    assert arb is not None and arb.profit > 0
    assert (
        arb.calls * C_ + arb.puts * P_ + arb.shares * S + arb.cash
        == pytest.approx(0.0, abs=1e-12)
    )
    ST = np.array([0.0, 20.0, K, 60.0, 200.0])
    value_T = (
        arb.calls * call_payoff(ST, K)
        + arb.puts * put_payoff(ST, K)
        + arb.shares * math.exp(q_ * T) * ST
        + arb.cash * math.exp(r * T)
    )
    np.testing.assert_allclose(value_T, arb.profit, atol=1e-10)


def test_forward_price_and_value():
    F = forward_price(100, 1.5, 0.04, 0.01)
    assert F == pytest.approx(100 * math.exp(0.03 * 1.5), abs=1e-12)
    assert long_forward_value(100, F, 1.5, 0.04, 0.01) == pytest.approx(
        0.0, abs=1e-12
    )
    assert long_forward_value(100, 95, 1.5, 0.04, 0.01) == pytest.approx(
        (F - 95) * discount_factor(0.04, 1.5)
    )


def test_discrete_dividends():
    divs = [Dividend(0.25, 1.0), Dividend(0.75, 1.0)]
    expected = (
        50 - math.exp(-0.06 * 0.25) - math.exp(-0.06 * 0.75)
    ) * math.exp(0.06)
    assert forward_price_discrete(50, 1, 0.06, divs) == pytest.approx(
        expected, abs=1e-12
    )
    assert forward_price_discrete(50, 1, 0.06, []) == pytest.approx(
        forward_price(50, 1, 0.06)
    )


def test_synthetic_forward_is_parity():
    P = put_from_call(C, S, K, T, r, q)
    assert synthetic_forward_value(C, P) == pytest.approx(
        long_forward_value(S, K, T, r, q), abs=1e-12
    )
    F = forward_price(S, T, r, q)
    assert put_from_call(C, S, F, T, r, q) == pytest.approx(C, abs=1e-12)
