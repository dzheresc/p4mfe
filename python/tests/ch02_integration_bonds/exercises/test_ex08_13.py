import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.bonds import bond_cash_flows, bond_price_from_inst_rate
from p4mfe.ch02_integration_bonds.exercises.ex08_09_rate_curves import (
    ex08_shift,
    ex09_annual_zero_rate,
    ex09_inst_rate,
    ex09_semiannual_zero_rate,
    ex09_zero_rate,
)
from p4mfe.ch02_integration_bonds.exercises.ex10_13_bond_pricing import (
    ex10_discount_factors,
    ex10_price,
    ex11_par_yield,
    ex12_discount_factor,
    ex12_inst_rate,
    ex12_price,
    ex12_zero_rate,
    ex13_discount_factor,
    ex13_inst_rate,
    ex13_integral,
    ex13_price,
    ex13_tol,
)
from p4mfe.ch02_integration_bonds.rates import (
    discount_factor_discrete,
    discount_factor_from_inst_rate,
    discrete_from_continuous,
    inst_rate_from_zero_rate,
    zero_rate_from_inst_rate,
)

TIMES = (0.25, 1.0, 3.0, 10.0)


def test_ex08_parallel_shift():
    zero1 = lambda t: 0.03 + 0.02 * (1 - np.exp(-t)) / (1 + t) + 0.004 * np.sin(t)  # noqa: E731
    dr = 0.0075
    zero2 = ex08_shift(zero1, dr)
    for t in TIMES:
        assert inst_rate_from_zero_rate(zero2, t) - inst_rate_from_zero_rate(
            zero1, t
        ) == pytest.approx(dr, abs=1e-9)
    r1 = lambda t: 0.04 + 0.01 * np.log1p(t) - 0.002 * t * np.exp(-t)  # noqa: E731
    r2 = lambda t: r1(t) + dr  # noqa: E731
    for t in TIMES:
        assert zero_rate_from_inst_rate(r2, t) - zero_rate_from_inst_rate(
            r1, t
        ) == pytest.approx(dr, abs=1e-12)


@pytest.mark.parametrize("t", [0.5, 1.0, 2.0, 5.0, 30.0])
def test_ex09_curves(t):
    assert ex09_inst_rate(t) == pytest.approx(
        inst_rate_from_zero_rate(ex09_zero_rate, t), abs=1e-9
    )
    assert ex09_zero_rate(t) == pytest.approx(
        zero_rate_from_inst_rate(ex09_inst_rate, t), abs=1e-12
    )
    D = math.exp(-ex09_zero_rate(t) * t)
    assert discount_factor_discrete(
        ex09_annual_zero_rate(t), t, 1
    ) == pytest.approx(D, abs=1e-14)
    assert discount_factor_discrete(
        ex09_semiannual_zero_rate(t), t, 2
    ) == pytest.approx(D, abs=1e-14)
    assert ex09_annual_zero_rate(t) == pytest.approx(
        discrete_from_continuous(ex09_zero_rate(t), 1), abs=1e-15
    )
    assert ex09_semiannual_zero_rate(t) == pytest.approx(
        discrete_from_continuous(ex09_zero_rate(t), 2), abs=1e-15
    )


def test_ex10_ex11():
    d = ex10_discount_factors()
    assert d == pytest.approx(
        [0.975310, 0.948854, 0.922886, 0.895834], abs=1e-6
    )
    assert ex10_price() == pytest.approx(98.9406, abs=1e-4)
    assert ex10_price() == pytest.approx(2.5 * sum(d) + 100 * d[-1], abs=1e-12)
    c = ex11_par_yield()
    assert c == pytest.approx(0.055661, abs=1e-6)
    assert ex10_price(c) == pytest.approx(100.0, abs=1e-10)


def test_ex12():
    for t in (0.5, 1.0, 1.5, 2.0, 7.0):
        assert ex12_zero_rate(t) == pytest.approx(
            zero_rate_from_inst_rate(ex12_inst_rate, t), abs=1e-12
        )
        assert ex12_discount_factor(t) == pytest.approx(
            discount_factor_from_inst_rate(ex12_inst_rate, t), abs=1e-12
        )
    assert [
        ex12_discount_factor(t) for t in (0.5, 1.0, 1.5, 2.0)
    ] == pytest.approx([0.974782, 0.949394, 0.924083, 0.898994], abs=1e-6)
    assert ex12_price() == pytest.approx(99.2675, abs=1e-4)
    assert ex12_price() == pytest.approx(
        bond_price_from_inst_rate(bond_cash_flows(2, 0.05, 2), ex12_inst_rate),
        abs=1e-10,
    )


def test_ex13():
    assert ex13_integral(1.0, 1e-6).intervals == 8
    assert ex13_integral(2.0, 1e-6).intervals == 16
    assert ex13_integral(3.0, 1e-8).intervals == 64
    for t in (1.0, 2.0, 3.0):
        assert ex13_discount_factor(t, ex13_tol(t)) == pytest.approx(
            ex13_discount_factor(t, 1e-14), abs=ex13_tol(t)
        )
    assert ex13_discount_factor(1.0, 1e-6) == pytest.approx(0.956595, abs=5e-7)
    assert ex13_discount_factor(2.0, 1e-6) == pytest.approx(0.910128, abs=5e-7)
    assert ex13_discount_factor(3.0, 1e-8) == pytest.approx(
        0.86574103, abs=5e-9
    )
    assert ex13_price() == pytest.approx(100.2364, abs=1e-4)
    assert ex13_price() == pytest.approx(
        bond_price_from_inst_rate(bond_cash_flows(3, 0.05, 1), ex13_inst_rate),
        abs=1e-6,
    )
