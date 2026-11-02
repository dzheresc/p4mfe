import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.bonds import (
    BondQuote,
    CashFlows,
    bond_cash_flows,
    bond_convexity,
    bond_duration,
    bond_metrics,
    bond_price,
    bond_price_discrete,
    bond_price_from_inst_rate,
    bond_price_from_zero_rate,
    bond_yield,
    bootstrap_zero_rates,
    dv01,
    macaulay_duration,
    modified_duration,
    price_approx_duration,
    price_approx_duration_convexity,
    zero_coupon_bond_price,
    zero_rate_from_zero_coupon_price,
)
from p4mfe.ch02_integration_bonds.rates import continuous_from_discrete


def fd1(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def fd2(f, x, h=1e-4):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def test_cash_flows():
    cf = bond_cash_flows(14 / 12, 0.05, 2)
    assert cf.times == pytest.approx([2 / 12, 8 / 12, 14 / 12], abs=1e-14)
    assert list(cf.amounts) == [2.5, 2.5, 102.5]
    assert len(bond_cash_flows(3, 0.06, 2).times) == 6
    assert len(bond_cash_flows(3, 0.0, 2).times) == 1


def test_price_from_curves():
    cf = bond_cash_flows(3, 0.06, 2)
    assert bond_price_from_zero_rate(cf, lambda t: 0.05) == pytest.approx(
        bond_price(cf, 0.05), abs=1e-12
    )
    r_inst = lambda t: 0.0525 + np.log1p(2 * t) / 200
    r_zero = lambda t: 0.0525 + ((1 + 2 * t) * math.log1p(2 * t) - 2 * t) / (
        400 * t
    )
    assert bond_price_from_inst_rate(cf, r_inst) == pytest.approx(
        bond_price_from_zero_rate(cf, r_zero), abs=1e-10
    )


@pytest.mark.parametrize("times", [[1.0, 0.5], [-0.5, 1.0]])
def test_price_from_inst_rate_rejects_unsorted_times(times):
    cf = CashFlows(np.array(times), np.array([3.0, 103.0]))
    with pytest.raises(ValueError):
        bond_price_from_inst_rate(cf, lambda t: 0.05 + 0 * t)


def test_duration_and_convexity_are_yield_derivatives():
    cf = bond_cash_flows(5, 0.04, 4)
    y = 0.045
    B = lambda s: bond_price(cf, s)
    m = bond_metrics(cf, y)
    assert m.price == B(y)
    assert m.duration == pytest.approx(-fd1(B, y) / m.price, abs=1e-8)
    assert m.convexity == pytest.approx(fd2(B, y) / m.price, abs=1e-4)
    assert bond_duration(cf, y) == m.duration
    assert bond_convexity(cf, y) == m.convexity
    assert m.duration < 5
    assert dv01(cf, y) == pytest.approx(
        (B(y - 1e-4) - B(y + 1e-4)) / 2, abs=1e-8
    )


def test_price_approximations():
    cf = bond_cash_flows(10, 0.05, 2)
    y, dy = 0.05, 0.01
    m = bond_metrics(cf, y)
    exact = bond_price(cf, y + dy)
    e1 = abs(price_approx_duration(m.price, m.duration, dy) - exact)
    e2 = abs(
        price_approx_duration_convexity(m.price, m.duration, m.convexity, dy)
        - exact
    )
    assert e2 < e1 / 10
    assert price_approx_duration(m.price, m.duration, dy) < exact
    assert price_approx_duration(m.price, m.duration, -dy) < bond_price(
        cf, y - dy
    )


def test_discrete_yield():
    cf = bond_cash_flows(3, 0.06, 2)
    assert bond_price_discrete(cf, 0.06, 2) == pytest.approx(100.0, abs=1e-12)
    yc = continuous_from_discrete(0.06, 2)
    assert bond_price(cf, yc) == pytest.approx(100.0, abs=1e-12)
    assert macaulay_duration(cf, 0.06, 2) == pytest.approx(
        bond_duration(cf, yc), abs=1e-12
    )
    B = lambda s: bond_price_discrete(cf, s, 2)
    assert modified_duration(cf, 0.06, 2) == pytest.approx(
        -fd1(B, 0.06) / 100.0, abs=1e-8
    )


def test_zero_coupon_bond():
    T, r = 4, 0.035
    P = zero_coupon_bond_price(T, r)
    assert P == pytest.approx(100 * math.exp(-0.14), abs=1e-12)
    assert zero_rate_from_zero_coupon_price(P, T) == pytest.approx(r, abs=1e-15)
    cf = bond_cash_flows(T, 0.0, 2)
    assert bond_yield(cf, P) == pytest.approx(r, abs=1e-12)
    assert bond_duration(cf, r) == pytest.approx(T, abs=1e-14)
    assert bond_convexity(cf, r) == pytest.approx(T * T, abs=1e-12)


def test_bootstrap():
    zero = lambda t: 0.03 + 0.01 * math.log1p(t)
    quotes = []
    for T in (2.0, 0.5, 1.5, 1.0):
        c = 0.04 + 0.01 * T
        quotes.append(
            BondQuote(
                T, c, bond_price_from_zero_rate(bond_cash_flows(T, c, 2), zero)
            )
        )
    curve = bootstrap_zero_rates(quotes, 2)
    assert curve.times == pytest.approx([0.5, 1.0, 1.5, 2.0], abs=1e-15)
    assert curve.rates == pytest.approx(
        [zero(t) for t in curve.times], abs=1e-13
    )
    with pytest.raises(ValueError):
        bootstrap_zero_rates(
            [BondQuote(0.5, 0.04, 99.0), BondQuote(1.5, 0.04, 98.0)], 2
        )


@pytest.mark.parametrize("y", [0.0, 0.02, 0.065, 0.15])
def test_yield_round_trip(y):
    cf = bond_cash_flows(7.3, 0.05, 2)
    assert bond_yield(cf, bond_price(cf, y)) == pytest.approx(y, abs=1e-12)


def test_premium_bond_yield_below_coupon():
    assert bond_yield(bond_cash_flows(3, 0.06, 2), 101.0) < 0.06
