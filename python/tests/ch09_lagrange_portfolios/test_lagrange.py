import numpy as np
import pytest

from p4mfe.ch09_lagrange_portfolios.lagrange import (
    CriticalPointType,
    classify_constrained,
    lagrange_stationary_point,
    lagrangian_gradient,
)


# 9.1 -----------------------------------------------------------------------
def test_linear_on_circle():
    def f(v):
        return v[0] + 2 * v[1]

    def g(v):
        return [v[0] ** 2 + v[1] ** 2 - 5]

    mx = lagrange_stationary_point(f, g, [0.8, 1.5])
    assert mx.x == pytest.approx([1, 2], abs=1e-8)
    assert mx.lam[0] == pytest.approx(-0.5, abs=1e-8)
    assert classify_constrained(f, g, mx.x, mx.lam) is CriticalPointType.MAXIMUM
    mn = lagrange_stationary_point(f, g, [-1.0, -1.0])
    assert f(mn.x) == pytest.approx(-5, abs=1e-8)
    assert classify_constrained(f, g, mn.x, mn.lam) is CriticalPointType.MINIMUM
    assert lagrangian_gradient(f, g, mx.x, mx.lam) == pytest.approx(
        np.zeros(3), abs=1e-8
    )


# 9.1.1 ---------------------------------------------------------------------
def test_two_constraints():
    def f(v):
        return float(np.dot(v, v))

    def g(v):
        return [v[0] + v[1] + v[2] - 3, v[0] - v[1] - 1]

    r = lagrange_stationary_point(f, g, [0, 0, 0])
    assert r.x == pytest.approx([1.5, 0.5, 1.0], abs=1e-8)
    assert classify_constrained(f, g, r.x, r.lam) is CriticalPointType.MINIMUM


def test_box_volume():
    def f(v):
        return v[0] * v[1] * v[2]

    def g(v):
        return [2 * (v[0] * v[1] + v[1] * v[2] + v[2] * v[0]) - 6]

    r = lagrange_stationary_point(f, g, [0.8, 1.2, 1.1])
    assert r.x == pytest.approx([1, 1, 1], abs=1e-8)
    assert classify_constrained(f, g, r.x, r.lam) is CriticalPointType.MAXIMUM


def test_saddle_and_errors():
    def f(v):
        return v[0] ** 2 - v[1] ** 2 + v[2]

    def g(v):
        return [v[2]]

    r = lagrange_stationary_point(f, g, [0.3, -0.2, 0.1])
    assert r.lam[0] == pytest.approx(-1, abs=1e-8)
    assert classify_constrained(f, g, r.x, r.lam) is CriticalPointType.SADDLE
    with pytest.raises(ValueError):
        lagrange_stationary_point(f, lambda v: list(v), [0, 0, 0])
