import math

import pytest

from p4mfe.ch07_finite_differences_bs_pde.exercises.ex01_02_difference_orders import ex_observed_order
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex06_butterfly_spread import (
    SpreadMarket,
    ex06_arrow_debreu_payoff,
    ex06_butterfly_payoff,
    ex06_butterfly_value,
    ex06_call_kk,
    ex06_scaled_butterfly,
    ex_call,
)
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex07_bull_spread import ex07_cash_or_nothing, ex07_digital_payoff, ex07_scaled_bull
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex08_butterfly_delta import ex08_integral
from p4mfe.ch07_finite_differences_bs_pde.finite_differences import central_difference


def test_ex06_value_and_payoff():
    m = SpreadMarket()
    x = 2.0
    parts = ex_call(m.K - x, m) - 2 * ex_call(m.K, m) + ex_call(m.K + x, m)
    assert ex06_butterfly_value(lambda k: ex_call(k, m), m.K, x) == pytest.approx(parts, rel=1e-14)
    assert parts > 0
    for S in (40.0, 48.0, 49.0, 50.0, 50.5, 52.0, 60.0):
        assert ex06_butterfly_payoff(S, m.K, x) == pytest.approx(max(x - abs(S - m.K), 0.0), abs=1e-12)


def test_ex06_scaled_butterfly_tends_to_ckk():
    m = SpreadMarket()
    ckk = ex06_call_kk(m)
    minus_ck = lambda k: ex07_cash_or_nothing(SpreadMarket(K=k))  # noqa: E731
    assert -central_difference(minus_ck, m.K, 1e-3) == pytest.approx(ckk, abs=1e-8)
    err = lambda x: ex06_scaled_butterfly(x, m) - ckk  # noqa: E731
    assert ex_observed_order(err, 1.0) == pytest.approx(2.0, abs=0.1)
    assert abs(err(0.05)) < 1e-6


def test_ex06_arrow_debreu_payoff():
    K = 50.0
    for x in (1.0, 0.1, 1e-3, 1e-6):
        assert ex06_arrow_debreu_payoff(K, K, x) == pytest.approx(1.0, abs=1e-12)
    for S in (49.0, 49.99, 50.001, 51.0):
        assert ex06_arrow_debreu_payoff(S, K, 2.0) > 0
        assert ex06_arrow_debreu_payoff(S, K, 1e-4) == 0.0


def test_ex07_scaled_bull_tends_to_cash_or_nothing():
    m = SpreadMarket()
    limit = ex07_cash_or_nothing(m)
    assert -central_difference(lambda k: ex_call(k, m), m.K, 1e-3) == pytest.approx(limit, abs=1e-8)
    err = lambda x: ex07_scaled_bull(x, m) - limit  # noqa: E731
    assert ex_observed_order(err, 0.2) == pytest.approx(1.0, abs=0.1)
    assert err(0.01) / (-0.5 * 0.01 * ex06_call_kk(m)) == pytest.approx(1.0, abs=0.02)
    values = [ex07_scaled_bull(x, m) for x in (4.0, 2.0, 1.0, 0.5, 0.1)]
    assert all(b > a for a, b in zip(values, values[1:]))
    assert values[-1] < limit


def test_ex07_digital_payoff():
    K = 50.0
    for S in (30.0, 49.9, 50.0):
        assert ex07_digital_payoff(S, K, 1e-3) == 0.0
    for S in (50.01, 51.0, 80.0):
        assert ex07_digital_payoff(S, K, 1e-3) == pytest.approx(1.0, abs=1e-9)
    assert ex07_digital_payoff(50.5, K, 2.0) == pytest.approx(0.25, abs=1e-12)


def test_ex08_nascent_delta():
    K = 50.0
    for x in (5.0, 1.0, 0.1):
        assert ex08_integral(lambda S: 1.0, K, x) == pytest.approx(1.0, abs=1e-12)
    g = lambda S: math.cos(S / 7)  # noqa: E731
    g2 = -math.cos(K / 7) / 49
    for x in (2.0, 1.0, 0.5):
        assert (ex08_integral(g, K, x) - g(K)) / (x * x / 12 * g2) == pytest.approx(1.0, abs=0.01)
    gk = lambda S: abs(S - 50.3) + math.sqrt(S)  # noqa: E731
    for x in (1.0, 0.1, 1e-3):
        assert abs(ex08_integral(gk, K, x) - gk(K)) <= 2 * x
    assert ex08_integral(gk, K, 1e-4) == pytest.approx(gk(K), abs=1e-4)
