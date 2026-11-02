import math

import pytest

from p4mfe.ch07_finite_differences_bs_pde.finite_differences import (
    FdScheme,
    backward_difference,
    central_difference,
    fd_error_table,
    fd_observed_order,
    first_difference,
    forward_difference,
    optimal_step,
    second_central_difference,
    truncation_order,
)


def f(x):
    return math.exp(-x * x / 2) * math.sin(x)


def df(x):
    return math.exp(-x * x / 2) * (math.cos(x) - x * math.sin(x))


def d2f(x):
    return math.exp(-x * x / 2) * (
        (x * x - 2) * math.sin(x) - 2 * x * math.cos(x)
    )


# 7.1 -----------------------------------------------------------------------
def test_schemes():
    a, h = 0.7, 1e-3
    assert forward_difference(f, a, h) == pytest.approx(df(a), abs=1e-3)
    assert central_difference(f, a, h) == pytest.approx(df(a), abs=1e-6)
    assert second_central_difference(f, a, h) == pytest.approx(d2f(a), abs=1e-5)
    assert forward_difference(f, a, h) - df(a) == pytest.approx(
        0.5 * h * d2f(a), abs=1e-6
    )
    assert backward_difference(f, a, h) - df(a) == pytest.approx(
        -0.5 * h * d2f(a), abs=1e-6
    )
    avg = 0.5 * (forward_difference(f, a, h) + backward_difference(f, a, h))
    assert avg == pytest.approx(central_difference(f, a, h), abs=1e-12)
    assert first_difference(f, a, h, FdScheme.BACKWARD) == backward_difference(
        f, a, h
    )
    with pytest.raises(ValueError):
        central_difference(f, a, 0.0)


def test_exact_for_polynomials():
    assert forward_difference(lambda x: 3 * x - 1, 2.0, 0.5) == pytest.approx(
        3.0
    )
    assert central_difference(
        lambda x: x * x - 3 * x, 2.0, 0.5
    ) == pytest.approx(1.0)
    assert second_central_difference(
        lambda x: x**3, 2.0, 0.5
    ) == pytest.approx(12.0)


@pytest.mark.parametrize("scheme", list(FdScheme))
def test_observed_order(scheme):
    assert fd_observed_order(f, 0.3, df(0.3), scheme, 1e-2) == pytest.approx(
        truncation_order(scheme), abs=0.05
    )


def test_observed_order_second():
    assert fd_observed_order(
        f, 0.3, d2f(0.3), FdScheme.CENTRAL, 1e-2, 2
    ) == pytest.approx(2.0, abs=0.05)


# 7.3.1 ---------------------------------------------------------------------
def test_round_off_and_optimal_step():
    a = 1.0
    rows = fd_error_table(f, a, df(a), FdScheme.CENTRAL, 0.1, 12)
    assert rows[3].h == pytest.approx(1e-4)
    assert rows[0].error / rows[1].error == pytest.approx(100, abs=5)
    assert abs(rows[10].error) > 100 * abs(rows[4].error)
    assert optimal_step(FdScheme.FORWARD) == pytest.approx(1.49e-8, abs=1e-10)
    assert optimal_step(FdScheme.CENTRAL) == pytest.approx(6.06e-6, abs=1e-8)
    assert optimal_step(FdScheme.CENTRAL, 2) == pytest.approx(1.22e-4, abs=1e-6)
    assert optimal_step(FdScheme.CENTRAL, 1, -50.0) == pytest.approx(
        50 * optimal_step(FdScheme.CENTRAL)
    )
    assert (
        abs(central_difference(f, a, optimal_step(FdScheme.CENTRAL)) - df(a))
        < 1e-10
    )
    assert (
        abs(forward_difference(f, a, optimal_step(FdScheme.FORWARD)) - df(a))
        < 1e-7
    )
    assert (
        abs(
            second_central_difference(f, a, optimal_step(FdScheme.CENTRAL, 2))
            - d2f(a)
        )
        < 1e-7
    )
