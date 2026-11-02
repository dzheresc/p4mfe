import math

import numpy as np
import pytest

from p4mfe.ch07_finite_differences_bs_pde.exercises.ex09_euler_convergence import ex09_bound, ex09_closed_form, ex09_max_error, ex09_solve
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex10_bvp_linear_system import ex10_solve, ex10_system
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex11_first_order_system import ex11_euler, ex11_f, ex11_ode_residual, ex11_rk4
from p4mfe.ch07_finite_differences_bs_pde.ode import solve_bvp


def test_ex09_closed_form():
    for n in (1, 5, 64):
        s = ex09_solve(n)
        for i in range(n + 1):
            assert s.y[i] == pytest.approx(ex09_closed_form(i, n), rel=1e-13)


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_ex09_first_order(n):
    s = ex09_solve(n)
    err = ex09_max_error(n)
    assert err == pytest.approx(math.e - s.y[-1], abs=1e-14)
    assert err < ex09_bound(n)
    assert n * err == pytest.approx(
        math.e / 2 - 11 * math.e / (24 * n), abs=2 / n**2
    )
    assert np.all(s.y <= np.exp(s.x) + 1e-15)


def test_ex10_system_n6():
    s = ex10_system(6)
    A = np.array(
        [
            [-5, 2.5, 0, 0, 0],
            [13, -23, 11, 0, 0],
            [0, 28.5, -53, 25.5, 0],
            [0, 0, 50, -95, 46],
            [0, 0, 0, 77.5, -149],
        ]
    )
    b = np.array([-3.5, 0, 0, 0, -36.25])
    np.testing.assert_array_equal(s.matrix(), A)
    np.testing.assert_array_equal(s.rhs, b)
    Y = ex10_solve(6).y[1:-1]
    np.testing.assert_allclose(
        Y, [1.385311, 1.370622, 1.228661, 1.021815, 0.774770], atol=5e-7
    )
    np.testing.assert_allclose(A @ Y, b, atol=1e-12)
    np.testing.assert_allclose(np.linalg.solve(A, b), Y, atol=1e-12)


@pytest.mark.parametrize("n", [6, 20])
def test_ex10_same_as_generic_bvp_solver(n):
    a = ex10_solve(n)
    b = solve_bvp(
        lambda x: -1 / (3 * x),
        lambda x: 1 / (3 * x * x),
        lambda x: 0.0,
        0.0,
        1.0,
        1.0,
        0.5,
        n,
    )
    np.testing.assert_allclose(a.y, b.y, atol=1e-10)


def test_ex10_no_convergence():
    for m in (1.0, 1 / 3):
        assert 3 * m * (m - 1) - m + 1 == pytest.approx(0.0, abs=1e-15)
    mids = [ex10_solve(n).y[n // 2] for n in (6, 12, 24, 48, 96)]
    assert all(b > a + 0.2 for a, b in zip(mids, mids[1:]))
    assert mids[-1] > 2.5


def test_ex11_system():
    d = ex11_f(2.0, [3.0, 5.0])
    assert d[0] == 5.0 and d[1] == -4 * 3 + 2 * 5
    ref = ex11_rk4([1.0, 0.0], 0.0, 1.0, 1000)
    for k in (1, 250, 500, 999):
        assert abs(ex11_ode_residual(ref, k)) < 1e-4
    for k in (100, 700):
        assert (ref.Y[k + 1, 0] - ref.Y[k - 1, 0]) / 2e-3 == pytest.approx(
            ref.Y[k, 1], abs=1e-5
        )
    err = lambda n: abs(ex11_euler([1.0, 0.0], 0.0, 1.0, n).Y[-1, 0] - ref.Y[-1, 0])  # noqa: E731
    assert math.log2(err(100) / err(200)) == pytest.approx(1.0, abs=0.05)
