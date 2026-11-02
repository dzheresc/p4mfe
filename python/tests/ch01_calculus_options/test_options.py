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

spot, strike, maturity, r, q, call = 42.0, 40.0, 0.5, 0.05, 0.02, 4.0


def test_payoffs():
    assert call_payoff(110, 100) == 10 and call_payoff(90, 100) == 0
    assert put_payoff(90, 100) == 10 and payoff(OptionType.PUT, 110, 100) == 0
    spot_t = np.array([50.0, 100.0, 150.0])
    np.testing.assert_allclose(
        call_payoff(spot_t, 100) - put_payoff(spot_t, 100), spot_t - 100
    )


def test_parity_round_trip():
    put = put_from_call(call, spot, strike, maturity, r, q)
    assert call_from_put(put, spot, strike, maturity, r, q) == pytest.approx(
        call, abs=1e-12
    )
    assert parity_violation(
        call, put, spot, strike, maturity, r, q
    ) == pytest.approx(0.0, abs=1e-12)
    assert parity_arbitrage(call, put, spot, strike, maturity, r, q) is None


def test_bounds():
    put = put_from_call(call, spot, strike, maturity, r, q)
    cb, pb = call_bounds(spot, strike, maturity, r, q), put_bounds(
        spot, strike, maturity, r, q
    )
    assert cb.lower <= call <= cb.upper
    assert pb.lower <= put <= pb.upper


@pytest.mark.parametrize("call, put, q_", [(5.0, 1.0, 0.02), (2.0, 1.0, 0.0)])
def test_arbitrage_is_riskless(call, put, q_):
    arb = parity_arbitrage(call, put, spot, strike, maturity, r, q_)
    assert arb is not None and arb.profit > 0
    assert (
        arb.calls * call + arb.puts * put + arb.shares * spot + arb.cash
        == pytest.approx(0.0, abs=1e-12)
    )
    spot_t = np.array([0.0, 20.0, strike, 60.0, 200.0])
    value_at_maturity = (
        arb.calls * call_payoff(spot_t, strike)
        + arb.puts * put_payoff(spot_t, strike)
        + arb.shares * math.exp(q_ * maturity) * spot_t
        + arb.cash * math.exp(r * maturity)
    )
    np.testing.assert_allclose(value_at_maturity, arb.profit, atol=1e-10)


def test_forward_price_and_value():
    forward = forward_price(100, 1.5, 0.04, 0.01)
    assert forward == pytest.approx(100 * math.exp(0.03 * 1.5), abs=1e-12)
    assert long_forward_value(100, forward, 1.5, 0.04, 0.01) == pytest.approx(
        0.0, abs=1e-12
    )
    assert long_forward_value(100, 95, 1.5, 0.04, 0.01) == pytest.approx(
        (forward - 95) * discount_factor(0.04, 1.5)
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
    put = put_from_call(call, spot, strike, maturity, r, q)
    assert synthetic_forward_value(call, put) == pytest.approx(
        long_forward_value(spot, strike, maturity, r, q), abs=1e-12
    )
    forward = forward_price(spot, maturity, r, q)
    assert put_from_call(call, spot, forward, maturity, r, q) == pytest.approx(
        call, abs=1e-12
    )
