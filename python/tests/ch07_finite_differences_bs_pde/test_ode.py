import math

import pytest

from p4mfe.ch07_finite_differences_bs_pde.ode import (
    OdeScheme,
    euler_backward,
    euler_forward,
    leapfrog,
    max_error,
    solve_bvp,
    solve_ivp,
    solve_tridiagonal,
)


# 7.2 -----------------------------------------------------------------------
@pytest.mark.parametrize("scheme", list(OdeScheme))
def test_ivp_convergence(scheme):
    def f(x, y):
        return -2 * x * y

    def exact(x):
        return math.exp(-x * x)

    e1 = max_error(solve_ivp(f, 0, 1, 2, 100, scheme), exact)
    e2 = max_error(solve_ivp(f, 0, 1, 2, 200, scheme), exact)
    assert math.log2(e1 / e2) == pytest.approx(2.0 if scheme is OdeScheme.CENTRAL else 1.0, abs=0.15)
    assert e2 < 1e-2


def test_forward_steps():
    sol = euler_forward(lambda x, y: -2 * x * y, 0, 1, 2, 4)
    assert sol.x[-1] == 2.0
    assert sol.y[1] == 1.0
    assert sol.y[2] == 0.5


def test_backward_nonlinear():
    sol = euler_backward(lambda x, y: y * y, 0, 0.5, 1, 400)
    assert sol.y[-1] == pytest.approx(1.0, abs=2e-3)
    h = 1 / 400
    for k in range(0, 400, 57):
        assert sol.y[k + 1] - sol.y[k] - h * sol.y[k + 1] ** 2 == pytest.approx(0, abs=1e-12)


def test_stability():
    def f(x, y):
        return -20 * y

    assert abs(euler_forward(f, 0, 1, 3, 20).y[-1]) > 1e3
    assert abs(euler_backward(f, 0, 1, 3, 20).y[-1]) < 1e-6
    assert abs(leapfrog(f, 0, 1, 3, 600).y[-1]) > 1.0


def test_tridiagonal():
    x = solve_tridiagonal([0, 1, 1], [2, 3, 4], [1, 1, 0], [3, 5, 5])
    assert x == pytest.approx([1, 1, 1], abs=1e-14)
    with pytest.raises(ValueError):
        solve_tridiagonal([0], [1, 2], [0, 0], [1, 1])


def test_bvp():
    def solve(n):
        return solve_bvp(lambda x: 0.0, lambda x: -1.0, lambda x: 0.0, 0, 1, 0, math.sinh(1), n)

    e1, e2 = max_error(solve(20), math.sinh), max_error(solve(40), math.sinh)
    assert math.log2(e1 / e2) == pytest.approx(2.0, abs=0.05)
    assert e2 < 1e-5

    def exact(x):
        return math.exp(x) / 4 + x * math.exp(-x)

    sol = solve_bvp(lambda x: 2.0, lambda x: 1.0, math.exp, 0, 1, exact(0), exact(1), 200)
    assert max_error(sol, exact) < 1e-5
