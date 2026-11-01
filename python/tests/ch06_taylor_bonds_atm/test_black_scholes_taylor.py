import math

import pytest

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_greeks, bs_price, bs_put
from p4mfe.ch06_taylor_bonds_atm.bs_atm import (
    atm_call_approx,
    atm_call_linearized,
    atm_greeks_approx,
    atm_implied_vol_approx,
    atm_price_approx,
    atm_price_exact,
    atm_put_approx,
    atm_relative_error,
    atm_relative_error_bound,
    norm_cdf_linear,
)
from p4mfe.ch06_taylor_bonds_atm.greeks_taylor import (
    delta_neutral_change,
    option_change_delta,
    option_change_delta_gamma,
    option_change_taylor,
    theta_from_pde,
)
from p4mfe.normal import norm_cdf


# 6.8.1 ---------------------------------------------------------------------
@pytest.mark.parametrize("T", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.35])
def test_atm_zero_rates(T, sigma):
    S = 40
    C = bs_call(S, S, T, sigma, 0.0)
    assert atm_price_approx(S, T, sigma) / C - 1 == pytest.approx(0.0, abs=sigma**2 * T / 20)
    assert atm_implied_vol_approx(C, S, T) / sigma - 1 == pytest.approx(0.0, abs=sigma**2 * T / 20)


@pytest.mark.parametrize("r", [0.0, 0.02, 0.05])
@pytest.mark.parametrize("q", [0.0, 0.01, 0.03])
def test_atm_with_rates(r, q):
    S, T, sigma = 50, 0.5, 0.25
    tol = 0.015 * atm_price_approx(S, T, sigma)
    assert atm_call_approx(S, T, sigma, r, q) == pytest.approx(bs_call(S, S, T, sigma, r, q), abs=tol)
    assert atm_put_approx(S, T, sigma, r, q) == pytest.approx(bs_put(S, S, T, sigma, r, q), abs=tol)
    assert atm_call_approx(S, T, sigma, r, q) - atm_put_approx(S, T, sigma, r, q) == pytest.approx((r - q) * T * S)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_atm_greeks(option_type):
    S, T, sigma = 60, 0.25, 0.2
    g, a = bs_greeks(option_type, S, S, T, sigma, 0.0), atm_greeks_approx(option_type, S, T, sigma)
    assert a.delta == pytest.approx(g.delta, abs=1e-4)
    assert a.gamma == pytest.approx(g.gamma, rel=2e-3)
    assert a.vega == pytest.approx(g.vega, rel=2e-3)
    assert a.theta == pytest.approx(g.theta, rel=2e-3)
    assert a.rho == pytest.approx(g.rho, abs=1e-3)


# 6.8.2 ---------------------------------------------------------------------
def test_derivation():
    for x in (-0.05, 0.01, 0.1):
        assert norm_cdf_linear(x) == pytest.approx(norm_cdf(x), abs=abs(x) ** 3 / 10)
    S, T, sigma, r, q = 100, 0.5, 0.3, 0.03, 0.01
    exact = bs_call(S, S, T, sigma, r, q)
    lin = atm_call_linearized(S, T, sigma, r, q)
    assert lin == pytest.approx(exact, rel=0.01)
    assert atm_call_approx(S, T, sigma, r, q) == pytest.approx(lin, abs=0.01 * exact)
    assert atm_call_linearized(S, T, sigma, 0, 0) == pytest.approx(atm_price_approx(S, T, sigma), abs=1e-12)


# 6.8.3 ---------------------------------------------------------------------
@pytest.mark.parametrize("T", [1 / 12, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.6])
def test_precision(T, sigma):
    e, b = atm_relative_error(T, sigma), atm_relative_error_bound(T, sigma)
    assert 0.75 * b < e <= b
    assert atm_price_exact(80, T, sigma) == pytest.approx(bs_call(80, 80, T, sigma, 0.0), abs=1e-12)


# 6.9 -----------------------------------------------------------------------
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_option_change_approximation(option_type):
    S, K, T, sigma, r, q = 50, 52, 0.5, 0.3, 0.04, 0.01
    V = bs_price(option_type, S, K, T, sigma, r, q)
    g = bs_greeks(option_type, S, K, T, sigma, r, q)
    dt = 1 / 252
    dS, ds, dr = sigma * S * math.sqrt(dt), 0.005, 0.001
    exact = bs_price(option_type, S + dS, K, T - dt, sigma + ds, r + dr, q) - V
    assert abs(option_change_taylor(g, dS, dt, ds, dr) - exact) < 0.01 * abs(exact)
    dV_S = bs_price(option_type, S + 3 * dS, K, T, sigma, r, q) - V
    assert abs(option_change_delta_gamma(g, 3 * dS) - dV_S) < abs(option_change_delta(g, 3 * dS) - dV_S) / 5
    assert theta_from_pde(V, S, g.delta, g.gamma, sigma, r, q) == pytest.approx(g.theta, abs=1e-10)


def test_delta_neutral_gamma_pnl():
    S, K, T, sigma, r = 100, 100, 0.75, 0.25, 0.03
    C = bs_call(S, K, T, sigma, r)
    g = bs_greeks(OptionType.CALL, S, K, T, sigma, r)
    Pi, dt = C - g.delta * S, 1 / 252
    be = sigma * S * math.sqrt(dt)
    for m in (0.0, 0.5, 1.0, 2.0):
        actual = (bs_call(S + m * be, K, T - dt, sigma, r) - C) - g.delta * m * be
        assert delta_neutral_change(Pi, S, g.gamma, sigma, r, m * be, dt) == pytest.approx(actual, abs=5e-3)
    assert delta_neutral_change(Pi, S, g.gamma, sigma, r, 0.5 * be, dt) < r * Pi * dt
    assert delta_neutral_change(Pi, S, g.gamma, sigma, r, 2.0 * be, dt) > r * Pi * dt
