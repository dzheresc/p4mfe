import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.bonds import bond_cash_flows, bond_price_from_zero_rate
from p4mfe.ch05_newton_bootstrapping.bootstrapping import zero_rate_linear
from p4mfe.ch05_newton_bootstrapping.exercises.ex05_central_difference_newton import (
    ex05_F,
    ex05_jacobian,
    ex05_solve,
)
from p4mfe.ch05_newton_bootstrapping.exercises.ex06_atm_delta_dividend import (
    ex06_delta,
    ex06_delta_dq,
    ex06_solve,
    ex06_upper_bound,
)
from p4mfe.ch05_newton_bootstrapping.exercises.ex07_treasury_bootstrap import (
    EX07_QUOTES,
    ex07_bootstrap_compounded,
    ex07_curve_continuous,
    ex07_curve_semiannual,
    ex07_discount,
    ex07_to_compounded,
)
from p4mfe.ch05_newton_bootstrapping.exercises.ex08_bootstrap_overnight import (
    EX08_QUOTES,
    ex08_curve,
    ex08_hint_residual,
)
from p4mfe.ch05_newton_bootstrapping.exercises.ex09_bootstrap_pricing_error import (
    ex09_curve,
    ex09_model_price,
    ex09_relative_error,
)
from p4mfe.ch05_newton_bootstrapping.newton_nd import Difference, finite_difference_jacobian


def reprices(curve, quotes, tol=1e-9):
    for bq in quotes:
        cf = bond_cash_flows(bq.maturity, bq.coupon_rate, 2)
        assert bond_price_from_zero_rate(
            cf, lambda t: zero_rate_linear(curve, t)
        ) == pytest.approx(bq.price, abs=tol)


@pytest.mark.parametrize("scheme", [Difference.CENTRAL, Difference.FORWARD])
def test_ex05_roots(scheme):
    assert np.linalg.norm(ex05_F([-1.0, 3.0, 1.0])) == 0.0
    a = ex05_solve([1, 2, 3], scheme)
    b = ex05_solve([2, 2, 2], scheme)
    np.testing.assert_allclose(
        a.x, [-1.690550760, 1.983107243, -0.884558078], atol=1e-8
    )
    # From (2, 2, 2) the iterates go through a chaotic phase; which root catches them depends on rounding.
    roots = ([-1.690550760, 1.983107243, -0.884558078], [-1.0, 3.0, 1.0])
    assert min(np.max(np.abs(b.x - np.array(r))) for r in roots) < 1e-8
    assert a.iterations == 9
    assert b.iterations > 20


def test_ex05_central_jacobian_second_order():
    x = np.array([1.0, 2.0, 3.0])
    J = ex05_jacobian(x)
    err_c = np.max(
        np.abs(
            finite_difference_jacobian(ex05_F, x, 1e-4, Difference.CENTRAL) - J
        )
    )
    err_f = np.max(
        np.abs(
            finite_difference_jacobian(ex05_F, x, 1e-4, Difference.FORWARD) - J
        )
    )
    assert err_c < 1e-6
    assert 1e-4 < err_f < 1e-2


def test_ex06():
    res = ex06_solve()
    assert res.x == pytest.approx(0.066906, abs=5e-7)
    assert res.iterations <= 6
    assert ex06_delta(res.x) == pytest.approx(0.5, abs=1e-12)
    assert ex06_delta(0.0) > 0.5 > ex06_delta(ex06_upper_bound())
    assert res.x < ex06_upper_bound()
    assert all(ex06_delta_dq(0.01 * k) < 0 for k in range(100))
    h = 1e-6
    assert ex06_delta_dq(0.05) == pytest.approx(
        (ex06_delta(0.05 + h) - ex06_delta(0.05 - h)) / (2 * h), abs=1e-8
    )
    assert ex06_solve(0.2).x == pytest.approx(res.x, abs=1e-12)


def test_ex07_continuous():
    c = ex07_curve_continuous()
    np.testing.assert_allclose(
        c.rates,
        [
            0.05234096,
            0.05063562,
            0.04728926,
            0.04758188,
            0.04630336,
            0.04677207,
        ],
        atol=5e-9,
    )
    reprices(c, EX07_QUOTES)
    np.testing.assert_allclose(
        ex07_bootstrap_compounded(EX07_QUOTES, 0).rates, c.rates, atol=1e-12
    )


def test_ex07_semiannual_is_higher():
    c, s = ex07_curve_continuous(), ex07_curve_semiannual()
    for rc, rs in zip(c.rates, s.rates):
        assert rs > rc
        assert rs - rc == pytest.approx(0.25 * rc * rc, abs=2e-5)
        assert rs == pytest.approx(ex07_to_compounded(rc, 2), abs=1e-7)
    for bq in EX07_QUOTES:
        cf = bond_cash_flows(bq.maturity, bq.coupon_rate, 2)
        B = sum(
            a * ex07_discount(zero_rate_linear(s, t), t, 2)
            for t, a in zip(cf.times, cf.amounts)
        )
        assert B == pytest.approx(bq.price, abs=1e-9)


def test_ex08():
    c = ex08_curve()
    np.testing.assert_allclose(
        c.rates,
        [0.05, 0.05063562, 0.04936960, 0.05298265, 0.04532568, 0.03211897],
        atol=5e-9,
    )
    assert ex08_hint_residual(c) == pytest.approx(0.0, abs=1e-9)
    assert zero_rate_linear(c, 7 / 6) == pytest.approx(
        (3 * c.rates[2] + c.rates[3]) / 4, abs=1e-15
    )
    reprices(c, EX08_QUOTES)


def test_ex09():
    c = ex09_curve()
    np.testing.assert_allclose(
        c.rates,
        [0.01089965, 0.01390016, 0.01209855, 0.02682389, 0.03737166],
        atol=5e-9,
    )
    assert c.rates[0] == pytest.approx(-math.log(0.994565) / 0.5, abs=1e-14)
    assert ex09_model_price(c) == pytest.approx(108.193042, abs=5e-7)
    assert ex09_relative_error(c) == pytest.approx(0.00586210, abs=5e-9)
    B = sum(
        (102.25 if k == 6 else 2.25)
        * math.exp(-zero_rate_linear(c, 0.5 * k) * 0.5 * k)
        for k in range(1, 7)
    )
    assert ex09_model_price(c) == pytest.approx(B, abs=1e-12)
