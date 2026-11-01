import math

import pytest

from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances, bisection, convergence_order, newton, secant


def _observed_order(xs, root):
    e = [abs(x - root) for x in xs if abs(x - root) > 1e-11]
    return convergence_order(e[-3], e[-2], e[-1])


def test_all_three_find_sqrt2():
    f, df = (lambda x: x * x - 2), (lambda x: 2 * x)
    tol = Tolerances(1e-12, 1e-12)
    b, n, s = bisection(f, 1, 2, tol), newton(f, df, 1, tol), secant(f, 2, 1, tol)
    for x in (b.x, n.x, s.x):
        assert x == pytest.approx(math.sqrt(2), abs=1e-12)
    assert n.iterations < s.iterations < b.iterations
    assert len(n.iterates) == n.iterations + 1
    assert len(b.iterates) == b.iterations


def test_bisection_halves_the_bracket():
    res = bisection(lambda x: math.cos(x) - x, 0, 1, Tolerances(1.0, 1e-6))
    assert res.iterations == 21
    assert res.x == pytest.approx(0.7390851332151607, abs=1e-6)


def test_both_stopping_criteria_must_hold():
    f = lambda x: 1e-6 * (x - 1)  # noqa: E731
    assert newton(f, lambda x: 1e-6, 3.0, Tolerances(1e-3, 1e-12)).x == pytest.approx(1, abs=1e-12)
    assert bisection(f, 0, 3, Tolerances(1e-3, 1e-9)).x == pytest.approx(1, abs=1e-9)


def test_convergence_orders():
    f, root = (lambda x: math.exp(x) - 2), math.log(2)
    tol = Tolerances(1e-15, 1e-15)
    assert _observed_order(newton(f, math.exp, 2.0, tol).iterates, root) == pytest.approx(2, abs=0.1)
    assert _observed_order(secant(f, 2.1, 2.0, tol).iterates, root) == pytest.approx(1.618, abs=0.15)


def test_quartic_with_logistic_term():
    f = lambda x: x**4 - 5 * x * x + 4 - 1 / (1 + math.exp(x**3))  # noqa: E731

    def df(x):
        e = math.exp(x**3)
        return 4 * x**3 - 10 * x + 3 * x * x * e / (1 + e) ** 2

    n = newton(f, df, 0.5)
    assert abs(f(n.x)) < 1e-12
    assert bisection(f, 0.5, 1.0).x == pytest.approx(n.x, abs=1e-9)
    assert secant(f, 0.49, 0.5).x == pytest.approx(n.x, abs=1e-9)


def test_failures():
    f, df = (lambda x: x * x + 1), (lambda x: 2 * x)
    with pytest.raises(ValueError):
        bisection(f, -1, 1)
    with pytest.raises(ValueError):
        bisection(f, 1, -1)
    with pytest.raises(RuntimeError):
        newton(f, df, 0.0)
    with pytest.raises(RuntimeError):
        newton(f, df, 0.5, Tolerances(1e-9, 1e-9, 50))


def test_exact_root_stops_immediately():
    f = lambda x: x - 0.5  # noqa: E731
    assert newton(f, lambda x: 1.0, 0.5).iterations == 0
    assert bisection(f, 0, 1).iterations == 1
