import pytest

from p4mfe.ch02_integration_bonds.bonds import (
    bond_cash_flows,
    bond_metrics,
    bond_price,
    bond_yield,
    price_approx_duration_convexity,
)
from p4mfe.ch02_integration_bonds.exercises.ex14_19_duration_convexity import (
    EX17_DURATION_ESTIMATE,
    ex14_metrics,
    ex15_metrics,
    ex16_metrics,
    ex17_convexity_estimate,
    ex17_exact_relative_change,
    ex18_new_price,
    ex19_duration_slope,
)


def _check_against_price(cf, y):
    h = 1e-4
    B, Bp, Bm = bond_price(cf, y), bond_price(cf, y + h), bond_price(cf, y - h)
    m = bond_metrics(cf, y)
    assert m.duration == pytest.approx(-(Bp - Bm) / (2 * h) / B, abs=1e-7)
    assert m.convexity == pytest.approx((Bp - 2 * B + Bm) / h**2 / B, abs=1e-5)


def test_ex14():
    m = ex14_metrics()
    assert m.price == pytest.approx(92.9839, abs=1e-4)
    assert m.duration == pytest.approx(2.352418, abs=1e-6)
    assert m.convexity == pytest.approx(5.736739, abs=1e-6)
    _check_against_price(bond_cash_flows(2.5, 0.06, 2), 0.09)


def test_ex15():
    m = ex15_metrics()
    assert m.price == pytest.approx(101.7049, abs=1e-4)
    assert m.duration == pytest.approx(1.118911, abs=1e-6)
    assert m.convexity == pytest.approx(1.285705, abs=1e-6)
    cf = bond_cash_flows(14 / 12, 0.08, 4)
    assert cf.times[0] == pytest.approx(2 / 12)
    _check_against_price(cf, 0.07)


@pytest.mark.parametrize("y", [0.02, 0.06, 0.12])
def test_ex16(y):
    ms = [ex16_metrics(10.0, c, 2, y) for c in (0.0, 0.01, 0.03, 0.05, 0.08, 0.15, 0.5)]
    assert all(a.price < b.price for a, b in zip(ms, ms[1:]))
    assert all(a.duration > b.duration for a, b in zip(ms, ms[1:]))
    assert ms[0].duration == pytest.approx(10.0)


def test_ex17():
    assert ex17_exact_relative_change() == pytest.approx(-0.00995017, abs=1e-8)
    assert EX17_DURATION_ESTIMATE == -0.01
    assert ex17_convexity_estimate() == pytest.approx(-0.00995, abs=1e-15)
    cf = bond_cash_flows(10.0, 0.0, 1)
    for y in (0.01, 0.05, 0.1):
        m = bond_metrics(cf, y)
        assert (m.duration, m.convexity) == pytest.approx((10.0, 100.0))
        assert bond_price(cf, y + 0.001) / m.price - 1 == pytest.approx(ex17_exact_relative_change(), abs=1e-14)


def test_ex18():
    assert ex18_new_price() == pytest.approx(103.785, abs=1e-12)
    cf = bond_cash_flows(5.0, 0.0905, 2)
    y = bond_yield(cf, 102.0)
    m = bond_metrics(cf, y)
    true_price = bond_price(cf, y - 0.005)
    assert true_price > ex18_new_price(m.price, m.duration, -0.005)
    assert true_price == pytest.approx(price_approx_duration_convexity(m.price, m.duration, m.convexity, -0.005), abs=1e-3)


def test_ex19():
    for cf in (bond_cash_flows(2.5, 0.06, 2), bond_cash_flows(14 / 12, 0.08, 4),
               bond_cash_flows(30.0, 0.04, 2), bond_cash_flows(7.0, 0.0, 1)):
        for y in (0.01, 0.05, 0.09):
            m = bond_metrics(cf, y)
            slope = ex19_duration_slope(cf, y)
            assert m.convexity == pytest.approx(m.duration**2 - slope, rel=1e-6)
            assert slope <= 1e-9
