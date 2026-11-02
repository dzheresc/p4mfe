import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.rates import (
    continuous_from_discrete,
    discount_factor,
    discount_factor_discrete,
    discount_factor_from_inst_rate,
    discrete_from_continuous,
    forward_rate,
    forward_rate_from_discounts,
    forward_rate_from_inst_rate,
    inst_rate_from_zero_rate,
    zero_rate_from_discount,
    zero_rate_from_inst_rate,
)


# r(t) = 0.0525 + ln(1 + 2t)/200 and its closed-form integral
def r_inst(t):
    return 0.0525 + np.log1p(2 * t) / 200


def r_int(t):
    return 0.0525 * t + ((1 + 2 * t) * math.log1p(2 * t) - 2 * t) / 400


def r_zero(t):
    return r_int(t) / t


def test_constant_rate():
    assert discount_factor(0.05, 2.0) == pytest.approx(
        math.exp(-0.1), rel=1e-15
    )
    assert zero_rate_from_discount(
        discount_factor(0.05, 2.0), 2.0
    ) == pytest.approx(0.05, abs=1e-15)
    assert discount_factor_from_inst_rate(
        lambda s: 0.05 + 0 * s, 2.0
    ) == pytest.approx(math.exp(-0.1), abs=1e-14)


@pytest.mark.parametrize("t", [0.25, 1.0, 3.5])
def test_curve_relations(t):
    assert discount_factor_from_inst_rate(r_inst, t) == pytest.approx(
        math.exp(-r_int(t)), abs=1e-12
    )
    assert zero_rate_from_inst_rate(r_inst, t) == pytest.approx(
        r_zero(t), abs=1e-12
    )
    assert inst_rate_from_zero_rate(r_zero, t) == pytest.approx(
        r_inst(t), abs=1e-8
    )


def test_forward_rates_agree():
    t1, t2 = 0.5, 2.0
    f = forward_rate(r_zero(t1), t1, r_zero(t2), t2)
    assert forward_rate_from_discounts(
        math.exp(-r_int(t1)), t1, math.exp(-r_int(t2)), t2
    ) == pytest.approx(f, abs=1e-13)
    assert forward_rate_from_inst_rate(r_inst, t1, t2) == pytest.approx(
        f, abs=1e-12
    )
    assert f > r_zero(t2)
    assert math.exp(r_zero(t1) * t1 + f * (t2 - t1)) == pytest.approx(
        math.exp(r_zero(t2) * t2), abs=1e-13
    )
    assert forward_rate(0.04, 1, 0.04, 3) == pytest.approx(0.04, abs=1e-15)
    with pytest.raises(ValueError):
        forward_rate(0.04, 3, 0.04, 1)


def test_compounding_equivalence():
    r = continuous_from_discrete(0.06, 2)
    assert r == pytest.approx(2 * math.log(1.03), abs=1e-15)
    assert discrete_from_continuous(r, 2) == pytest.approx(0.06, abs=1e-15)
    for t in (0.5, 1.0, 7.25):
        assert discount_factor_discrete(0.06, t, 2) == pytest.approx(
            discount_factor(r, t), abs=1e-15
        )
    assert (
        discrete_from_continuous(0.05, 1)
        > discrete_from_continuous(0.05, 12)
        > 0.05
    )
