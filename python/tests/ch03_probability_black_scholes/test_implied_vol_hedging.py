import math

import pytest

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_greeks, bs_price
from p4mfe.ch03_probability_black_scholes.hedging import (
    delta_gamma_hedge,
    delta_hedge,
    portfolio_change_approx,
)
from p4mfe.ch03_probability_black_scholes.implied_vol import implied_vol_bisection, implied_vol_newton

S, T, R, Q = 30.0, 0.5, 0.03, 0.01


@pytest.mark.parametrize("t", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("K", [20.0, 28.0, 30.0 * math.exp((R - Q) * T), 35.0, 45.0])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_newton_recovers_sigma(t, K, sigma):
    price = bs_price(t, S, K, T, sigma, R, Q)
    res = implied_vol_newton(t, price, S, K, T, R, Q)
    # sigma is only pinned down to ~(price rounding)/vega; vega is ~1e-7 for the
    # deep in/out-of-the-money, low-vol cases.
    vega = bs_greeks(t, S, K, T, sigma, R, Q).vega
    assert res.sigma == pytest.approx(sigma, abs=1e-8 + 1e-12 / vega)
    assert bs_price(t, S, K, T, res.sigma, R, Q) == pytest.approx(price, abs=1e-12)
    assert res.iterations < 30


def test_bisection_agrees_with_newton():
    n = implied_vol_newton(OptionType.CALL, 2.5, S, 33, T, R, Q)
    b = implied_vol_bisection(OptionType.CALL, 2.5, S, 33, T, R, Q)
    assert n.sigma == pytest.approx(b.sigma, abs=1e-9)
    assert b.iterations > n.iterations


def test_rejects_arbitrageable_prices():
    with pytest.raises(ValueError):
        implied_vol_newton(OptionType.CALL, 31.0, 30, 30, 0.5, 0.03)
    with pytest.raises(ValueError):
        implied_vol_newton(OptionType.CALL, 0.5, 40, 30, 0.5, 0.03)


def test_delta_hedge():
    S0, K, T0, sigma, r = 50, 50, 0.25, 0.3, 0.05
    g = bs_greeks(OptionType.CALL, S0, K, T0, sigma, r)
    V = bs_call(S0, K, T0, sigma, r)
    pos = delta_hedge(-1000, V, g.delta, S0)
    assert pos.shares == pytest.approx(1000 * g.delta)
    assert pos.value(V, S0, r, 0.0) == pytest.approx(0.0, abs=1e-9)
    dS, dt = 0.5, 1 / 252
    V1 = bs_call(S0 + dS, K, T0 - dt, sigma, r)
    pnl = pos.value(V1, S0 + dS, r, dt)
    approx = -1000 * portfolio_change_approx(0.0, g.gamma, dS, g.theta, dt) + pos.cash * r * dt
    assert abs(pnl) < 0.1 * abs(1000 * (V1 - V))
    assert pnl == pytest.approx(approx, abs=0.05 * abs(approx) + 0.5)


def test_delta_gamma_hedge():
    S0, sigma, r = 50, 0.3, 0.05
    gp = bs_greeks(OptionType.CALL, S0, 50, 0.25, sigma, r)
    gh = bs_greeks(OptionType.CALL, S0, 55, 0.5, sigma, r)
    pd, pg = -1000 * gp.delta, -1000 * gp.gamma
    h = delta_gamma_hedge(pd, pg, gh.delta, gh.gamma)
    assert pg + h.hedge_options * gh.gamma == pytest.approx(0.0, abs=1e-10)
    assert pd + h.hedge_options * gh.delta + h.shares == pytest.approx(0.0, abs=1e-10)

    def value(s):
        return -1000 * bs_call(s, 50, 0.25, sigma, r) + h.hedge_options * bs_call(s, 55, 0.5, sigma, r) + h.shares * s

    dS = 2.0
    delta_only = -1000 * (bs_call(S0 + dS, 50, 0.25, sigma, r) - bs_call(S0, 50, 0.25, sigma, r)) - pd * dS
    assert abs(value(S0 + dS) - value(S0)) < 0.1 * abs(delta_only)
    with pytest.raises(ValueError):
        delta_gamma_hedge(1, 1, 0.5, 0.0)
