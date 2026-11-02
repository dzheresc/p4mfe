import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import (
    RiemannRule,
    compounding_factor,
    continuous_compounding_factor,
    derivative,
    integral_derivative,
    lhopital_limit,
    riemann_sum,
    second_derivative,
)


def test_product_and_chain_rule():
    x = 0.7
    assert derivative(lambda t: t * t * math.exp(t), x) == pytest.approx(
        (x * x + 2 * x) * math.exp(x), abs=1e-9
    )
    assert derivative(lambda t: math.sin(t * t), x) == pytest.approx(
        2 * x * math.cos(x * x), abs=1e-9
    )


def test_second_derivative():
    assert second_derivative(math.log, 2.0) == pytest.approx(-0.25, abs=1e-6)


def test_riemann_sums_converge():
    f = lambda x: x * np.exp(x)  # integral over [0, 1] is 1
    assert riemann_sum(f, 0, 1, 1000) == pytest.approx(1.0, abs=1e-6)
    assert riemann_sum(f, 0, 1, 100_000, RiemannRule.LEFT) == pytest.approx(
        1.0, abs=1e-4
    )
    assert riemann_sum(f, 0, 1, 100_000, RiemannRule.RIGHT) == pytest.approx(
        1.0, abs=1e-4
    )
    assert (
        riemann_sum(f, 0, 1, 10, RiemannRule.LEFT)
        < 1.0
        < riemann_sum(f, 0, 1, 10, RiemannRule.RIGHT)
    )


def test_leibniz_rule():
    # I(t) = int_0^t e^(-t x) dx = (1 - e^(-t^2)) / t
    t = 1.3
    expected = (2 * t * t * math.exp(-t * t) - 1 + math.exp(-t * t)) / (t * t)
    got = integral_derivative(
        lambda x, s: np.exp(-s * x), lambda s: 0.0, lambda s: s, t
    )
    assert got == pytest.approx(expected, abs=1e-6)


def test_discrete_compounding_converges():
    r, T = 0.05, 2.0
    limit = continuous_compounding_factor(r, T)
    errors = [
        abs(compounding_factor(r, T, n) - limit) for n in (1, 2, 12, 365, 1e6)
    ]
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-8


def test_lhopital():
    assert lhopital_limit(math.sin, lambda x: x, 0.0) == pytest.approx(
        1.0, abs=1e-9
    )
    assert lhopital_limit(
        lambda x: 1 - math.cos(x), lambda x: x * x, 0.0
    ) == pytest.approx(0.5, abs=1e-6)
    assert lhopital_limit(math.log, lambda x: x - 1, 1.0) == pytest.approx(
        1.0, abs=1e-9
    )
