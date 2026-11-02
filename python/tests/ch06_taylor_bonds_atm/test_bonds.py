import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.bonds import (
    CashFlows,
    bond_cash_flows,
    bond_convexity,
    bond_duration,
    bond_metrics,
    bond_price,
    bond_price_discrete,
    dollar_duration,
    dv01,
    price_approx_duration,
)
from p4mfe.ch06_taylor_bonds_atm.dollar_duration import (
    DollarRisk,
    bond_dollar_risk,
    dollar_convexity,
    convexity_from_dollar,
    duration_from_dollar,
    duration_hedge_units,
    dv01_central,
    dv01_reprice,
    immunize,
    portfolio_dollar_risk,
    scale,
    value_change_approx,
    value_change_approx_convexity,
)
from p4mfe.ch06_taylor_bonds_atm.duration_convexity import (
    cash_flow_dispersion,
    convexity_discrete,
    duration_derivative,
    portfolio_convexity,
    portfolio_duration,
)
from p4mfe.ch06_taylor_bonds_atm.yield_curve_shifts import (
    curve_shift_metrics,
    shifted_curve_price,
    shifted_curve_price_approx,
    yield_change_from_shift,
)


def zero_curve(t):
    return 0.0125 + math.log1p(2 * t) / 200


# 6.5 -----------------------------------------------------------------------
def test_flat_curve_matches_yield_metrics():
    cf = bond_cash_flows(5.0, 0.06, 2)
    y = 0.045
    flat = lambda t: y  # noqa: E731
    m, my = curve_shift_metrics(cf, flat), bond_metrics(cf, y)
    assert m.price == pytest.approx(my.price, abs=1e-12)
    assert m.duration == pytest.approx(my.duration, abs=1e-12)
    assert m.convexity == pytest.approx(my.convexity, abs=1e-12)
    for dr in (-0.01, 0.0005, 0.02):
        assert shifted_curve_price(cf, flat, dr) == pytest.approx(
            bond_price(cf, y + dr), abs=1e-12
        )
        assert yield_change_from_shift(cf, flat, dr) == pytest.approx(
            dr, abs=1e-12
        )


@pytest.mark.parametrize("dr", [-0.005, 0.001, 0.01])
def test_taylor_approximation_of_shifted_price(dr):
    cf = bond_cash_flows(10.0, 0.05, 2)
    m = curve_shift_metrics(cf, zero_curve)
    exact = shifted_curve_price(cf, zero_curve, dr)
    err2 = abs(shifted_curve_price_approx(cf, zero_curve, dr) - exact)
    err1 = abs(price_approx_duration(m.price, m.duration, dr) - exact)
    assert err2 < err1 / 10
    assert err2 < m.price * (10 * abs(dr)) ** 3 / 6
    assert yield_change_from_shift(cf, zero_curve, dr) / dr == pytest.approx(
        1.0, abs=5e-3
    )


# 6.6 -----------------------------------------------------------------------
def test_duration_convexity_connections():
    cf = bond_cash_flows(7.5, 0.04, 2)
    y = 0.05
    m = bond_metrics(cf, y)
    assert cash_flow_dispersion(cf, y) > 0
    h = 1e-6
    dD = (bond_duration(cf, y + h) - bond_duration(cf, y - h)) / (2 * h)
    assert dD == pytest.approx(
        duration_derivative(m.duration, m.convexity), abs=1e-6
    )
    assert dD < 0
    assert cash_flow_dispersion(
        bond_cash_flows(4.0, 0.0, 2), y
    ) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("m", [1, 2, 12])
def test_discrete_convexity(m):
    cf = bond_cash_flows(3.0, 0.05, 2)
    y, h = 0.06, 1e-5
    B = bond_price_discrete(cf, y, m)
    fd = (
        (
            bond_price_discrete(cf, y + h, m)
            - 2 * B
            + bond_price_discrete(cf, y - h, m)
        )
        / h**2
        / B
    )
    assert convexity_discrete(cf, y, m) == pytest.approx(fd, abs=1e-3)
    assert convexity_discrete(cf, y, 1_000_000) == pytest.approx(
        bond_convexity(cf, y), abs=1e-4
    )


def test_portfolio_averages():
    y = 0.05
    b1, b2 = bond_cash_flows(2.0, 0.04, 2), bond_cash_flows(9.0, 0.06, 2)
    n1, n2 = 30, 70
    both = CashFlows(
        np.concatenate([b1.times, b2.times]),
        np.concatenate([n1 * b1.amounts, n2 * b2.amounts]),
    )
    m1, m2, mp = bond_metrics(b1, y), bond_metrics(b2, y), bond_metrics(both, y)
    V = [n1 * m1.price, n2 * m2.price]
    assert portfolio_duration(V, [m1.duration, m2.duration]) == pytest.approx(
        mp.duration, abs=1e-12
    )
    assert portfolio_convexity(
        V, [m1.convexity, m2.convexity]
    ) == pytest.approx(mp.convexity, abs=1e-12)
    with pytest.raises(ValueError):
        portfolio_duration([1, -1], [2, 3])


# 6.7, 6.7.1 ----------------------------------------------------------------
def test_dollar_risk_and_dv01():
    cf = bond_cash_flows(5.0, 0.05, 2)
    y = 0.04
    r = bond_dollar_risk(cf, y)
    assert r.dollar_duration == pytest.approx(dollar_duration(cf, y), abs=1e-12)
    assert r.dv01() == pytest.approx(dv01(cf, y), abs=1e-15)
    assert r.dollar_convexity == pytest.approx(
        dollar_convexity(cf, y), abs=1e-12
    )
    assert duration_from_dollar(r) == pytest.approx(
        bond_duration(cf, y), abs=1e-12
    )
    assert convexity_from_dollar(r) == pytest.approx(
        bond_convexity(cf, y), abs=1e-12
    )
    assert dv01_reprice(cf, y) == pytest.approx(
        r.dv01() + 0.5 * r.dollar_convexity * 1e-8, abs=1e-8
    )
    assert abs(dv01_reprice(cf, y) - r.dv01()) > 1e-6
    assert dv01_central(cf, y) == pytest.approx(r.dv01(), abs=1e-8)
    for dy in (-0.01, 0.002):
        dV = bond_price(cf, y + dy) - r.value
        assert (
            abs(value_change_approx_convexity(r, dy) - dV)
            < abs(value_change_approx(r, dy) - dV) / 10
        )


def test_zero_value_portfolio():
    y = 0.05
    r1 = bond_dollar_risk(bond_cash_flows(2.0, 0.05, 2), y)
    r2 = bond_dollar_risk(bond_cash_flows(10.0, 0.05, 2), y)
    p = portfolio_dollar_risk([1.0, -r1.value / r2.value], [r1, r2])
    assert p.value == pytest.approx(0.0, abs=1e-12)
    assert p.dollar_duration < 0
    with pytest.raises(ZeroDivisionError):
        duration_from_dollar(
            DollarRisk(0.0, p.dollar_duration, p.dollar_convexity)
        )


# 6.7.2 ---------------------------------------------------------------------
def test_immunization():
    y = 0.045
    port_cf = bond_cash_flows(7.0, 0.05, 2)
    h1_cf, h2_cf = bond_cash_flows(2.0, 0.03, 2), bond_cash_flows(20.0, 0.06, 2)
    P = scale(bond_dollar_risk(port_cf, y), 1000.0)
    H1, H2 = bond_dollar_risk(h1_cf, y), bond_dollar_risk(h2_cf, y)
    im = immunize(P, H1, H2)
    total = portfolio_dollar_risk([1.0, im.units1, im.units2], [P, H1, H2])
    assert total.value + im.cash == pytest.approx(0.0, abs=1e-8)
    assert total.dollar_duration == pytest.approx(0.0, abs=1e-7)
    assert total.dollar_convexity == pytest.approx(0.0, abs=1e-5)
    q = duration_hedge_units(P, H1)

    def value(dy, u1, u2, cash):
        return (
            1000 * bond_price(port_cf, y + dy)
            + u1 * bond_price(h1_cf, y + dy)
            + u2 * bond_price(h2_cf, y + dy)
            + cash
        )

    for dy in (-0.01, -0.001, 0.005):
        immunized = value(dy, im.units1, im.units2, im.cash)
        duration_only = value(dy, q, 0, -(P.value + q * H1.value))
        unhedged = value(dy, 0, 0, -P.value)
        assert abs(immunized) < abs(duration_only) / 5
        assert abs(duration_only) < abs(unhedged) / 5
    with pytest.raises(ValueError):
        immunize(P, H1, scale(H1, 2.0))
