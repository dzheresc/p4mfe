import math

import numpy as np
import pytest

from p4mfe.ch05_newton_bootstrapping.newton_nd import (
    Difference,
    approximate_newton_nd,
    finite_difference_jacobian,
    newton_nd,
    solve_linear,
)
from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances

ROOT = np.array([1.0, 2.0, 3.0])
X0 = [1.3, 1.7, 2.6]


def F3(x):
    return np.array(
        [
            x[0] ** 3 + x[1] * x[2] - 7,
            x[0] ** 2 * x[1] + x[2] - 5,
            x[0] + x[1] + x[2] ** 2 - 12,
        ]
    )


def J3(x):
    return np.array(
        [
            [3 * x[0] ** 2, x[2], x[1]],
            [2 * x[0] * x[1], x[0] ** 2, 1],
            [1, 1, 2 * x[2]],
        ]
    )


def test_solve_linear():
    x = solve_linear([[0, 2, 1], [1, 1, 1], [2, 1, 3]], [7, 6, 13])
    np.testing.assert_allclose(x, ROOT, atol=1e-14)
    with pytest.raises(RuntimeError):
        solve_linear([[1, 2], [2, 4]], [1, 2])


def test_newton_finds_root():
    res = newton_nd(F3, J3, X0, Tolerances(1e-12, 1e-12))
    np.testing.assert_allclose(res.x, ROOT, atol=1e-12)
    assert res.iterations <= 8


def test_quadratic_convergence():
    x, e_prev = np.array(X0), None
    for _ in range(4):
        x = newton_nd(F3, J3, x, Tolerances(1e300, 1e300, 1)).x
        e = float(np.linalg.norm(x - ROOT))
        if e_prev is not None and e > 1e-13:
            assert e / e_prev**2 < 5
        e_prev = e


def test_finite_difference_jacobian():
    x = np.array([0.7, -1.2, 2.1])
    np.testing.assert_allclose(
        finite_difference_jacobian(F3, x, 1e-6), J3(x), atol=1e-4
    )
    np.testing.assert_allclose(
        finite_difference_jacobian(F3, x, 1e-5, Difference.CENTRAL),
        J3(x),
        atol=1e-8,
    )


def test_approximate_newton_matches_newton():
    exact = newton_nd(F3, J3, X0, Tolerances(1e-12, 1e-12))
    approx = approximate_newton_nd(F3, X0, Tolerances(1e-12, 1e-12))
    np.testing.assert_allclose(approx.x, exact.x, atol=1e-12)
    assert approx.iterations >= exact.iterations


def test_two_dimensional():
    def F(v):
        return np.array([v[0] ** 2 + v[1] ** 2 - 4, math.exp(v[0]) + v[1] - 1])

    res = approximate_newton_nd(F, [-2.0, 1.0], Tolerances(1e-12, 1e-12))
    assert np.linalg.norm(F(res.x)) < 1e-12
    assert res.x[0] < 0
