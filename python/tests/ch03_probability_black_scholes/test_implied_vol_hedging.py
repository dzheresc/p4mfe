import math

import pytest

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_greeks, bs_price
from p4mfe.ch03_probability_black_scholes.hedging import (
    delta_gamma_hedge,
    delta_hedge,
    portfolio_change_approx,
)
from p4mfe.ch03_probability_black_scholes.implied_vol import implied_vol_bisection, implied_vol_newton

SPOT, MATURITY, R, Q = 30.0, 0.5, 0.03, 0.01


@pytest.mark.parametrize("t", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize(
    "strike", [20.0, 28.0, 30.0 * math.exp((R - Q) * MATURITY), 35.0, 45.0]
)
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_newton_recovers_sigma(t, strike, sigma):
    price = bs_price(t, SPOT, strike, MATURITY, sigma, R, Q)
    res = implied_vol_newton(t, price, SPOT, strike, MATURITY, R, Q)
    # sigma is only pinned down to ~(price rounding)/vega; vega is ~1e-7 for the
    # deep in/out-of-the-money, low-vol cases.
    vega = bs_greeks(t, SPOT, strike, MATURITY, sigma, R, Q).vega
    assert res.sigma == pytest.approx(sigma, abs=1e-8 + 1e-12 / vega)
    assert bs_price(
        t, SPOT, strike, MATURITY, res.sigma, R, Q
    ) == pytest.approx(price, abs=1e-12)
    assert res.iterations < 30


def test_bisection_agrees_with_newton():
    n = implied_vol_newton(OptionType.CALL, 2.5, SPOT, 33, MATURITY, R, Q)
    b = implied_vol_bisection(OptionType.CALL, 2.5, SPOT, 33, MATURITY, R, Q)
    assert n.sigma == pytest.approx(b.sigma, abs=1e-9)
    assert b.iterations > n.iterations


def test_rejects_arbitrageable_prices():
    with pytest.raises(ValueError):
        implied_vol_newton(OptionType.CALL, 31.0, 30, 30, 0.5, 0.03)
    with pytest.raises(ValueError):
        implied_vol_newton(OptionType.CALL, 0.5, 40, 30, 0.5, 0.03)


def test_delta_hedge():
    spot0, strike, maturity0, sigma, r = 50, 50, 0.25, 0.3, 0.05
    g = bs_greeks(OptionType.CALL, spot0, strike, maturity0, sigma, r)
    value = bs_call(spot0, strike, maturity0, sigma, r)
    pos = delta_hedge(-1000, value, g.delta, spot0)
    assert pos.shares == pytest.approx(1000 * g.delta)
    assert pos.value(value, spot0, r, 0.0) == pytest.approx(0.0, abs=1e-9)
    d_spot, dt = 0.5, 1 / 252
    value1 = bs_call(spot0 + d_spot, strike, maturity0 - dt, sigma, r)
    pnl = pos.value(value1, spot0 + d_spot, r, dt)
    approx = (
        -1000 * portfolio_change_approx(0.0, g.gamma, d_spot, g.theta, dt)
        + pos.cash * r * dt
    )
    assert abs(pnl) < 0.1 * abs(1000 * (value1 - value))
    assert pnl == pytest.approx(approx, abs=0.05 * abs(approx) + 0.5)


def test_delta_gamma_hedge():
    spot0, sigma, r = 50, 0.3, 0.05
    gp = bs_greeks(OptionType.CALL, spot0, 50, 0.25, sigma, r)
    gh = bs_greeks(OptionType.CALL, spot0, 55, 0.5, sigma, r)
    pd, pg = -1000 * gp.delta, -1000 * gp.gamma
    h = delta_gamma_hedge(pd, pg, gh.delta, gh.gamma)
    assert pg + h.hedge_options * gh.gamma == pytest.approx(0.0, abs=1e-10)
    assert pd + h.hedge_options * gh.delta + h.shares == pytest.approx(
        0.0, abs=1e-10
    )

    def value(s):
        return (
            -1000 * bs_call(s, 50, 0.25, sigma, r)
            + h.hedge_options * bs_call(s, 55, 0.5, sigma, r)
            + h.shares * s
        )

    d_spot = 2.0
    delta_only = (
        -1000
        * (
            bs_call(spot0 + d_spot, 50, 0.25, sigma, r)
            - bs_call(spot0, 50, 0.25, sigma, r)
        )
        - pd * d_spot
    )
    assert abs(value(spot0 + d_spot) - value(spot0)) < 0.1 * abs(delta_only)
    with pytest.raises(ValueError):
        delta_gamma_hedge(1, 1, 0.5, 0.0)
