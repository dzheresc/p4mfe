import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import (
    derivative,
    integral_derivative,
    riemann_sum,
    second_derivative,
)
from p4mfe.ch01_calculus_options.exercises.ex11_delta_integral import Ex11Params, ex11_b, ex11_g, ex11_g_prime
from p4mfe.ch01_calculus_options.exercises.ex12_13_integral_limits import (
    ex12_local_average,
    ex13_gaussian_average,
)
from p4mfe.ch01_calculus_options.exercises.ex14_bond_yield_derivatives import ex14_continuous, ex14_discrete
from p4mfe.ch01_calculus_options.exercises.ex15_quadratic_taylor import (
    EX15_A,
    ex15_f,
    ex15_gradient,
    ex15_hessian,
    ex15_taylor2,
)
from p4mfe.ch01_calculus_options.exercises.ex16_heat_equation import ex16_u, ex16_u_t, ex16_u_x, ex16_u_xx
from p4mfe.ch01_calculus_options.multivariable import gradient, hessian

P11 = Ex11Params(100.0, 0.5, 0.3, 0.04)


@pytest.mark.parametrize("x", [60.0, 90.0, 100.0, 120.0, 180.0])
def test_ex11(x):
    by_integral = riemann_sum(lambda y: np.exp(-0.5 * y * y), 0.0, ex11_b(P11, x), 4000) / math.sqrt(2 * math.pi)
    assert by_integral == pytest.approx(ex11_g(P11, x), abs=1e-7)
    assert derivative(lambda s: ex11_g(P11, s), x) == pytest.approx(ex11_g_prime(P11, x), abs=1e-8)
    leibniz = integral_derivative(
        lambda y, s: np.exp(-0.5 * y * y), lambda s: 0.0, lambda s: ex11_b(P11, s), x
    ) / math.sqrt(2 * math.pi)
    assert leibniz == pytest.approx(ex11_g_prime(P11, x), abs=1e-8)


def test_ex12_second_order():
    f = lambda x: np.cos(x) + x**3  # noqa: E731
    errs = [abs(ex12_local_average(f, 0.7, h, 2000) - f(0.7)) for h in (0.4, 0.2, 0.1, 0.05)]
    for e0, e1 in zip(errs, errs[1:]):
        assert e0 / e1 == pytest.approx(4.0, abs=0.2)
    assert ex12_local_average(lambda x: np.abs(x - 1), 1.0, 1e-4, 2000) == pytest.approx(0.0, abs=1e-4)


def test_ex13():
    for sigma in (1.0, 0.1, 0.01):
        assert ex13_gaussian_average(np.cos, 0.4, sigma) == pytest.approx(
            math.cos(0.4) * math.exp(-0.5 * sigma**2), abs=1e-10
        )
    g = lambda x: np.tanh(5 * x) + 1 / (1 + x * x)  # noqa: E731
    errs = [abs(ex13_gaussian_average(g, 0.3, s) - g(0.3)) for s in (0.5, 0.1, 0.02, 0.004)]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 1e-3


def test_ex14():
    times = [0.5 * i for i in range(1, 7)]
    amounts = [3.0] * 5 + [103.0]
    y = 0.055
    d = ex14_continuous(times, amounts, y)
    f = lambda s: ex14_continuous(times, amounts, s).value  # noqa: E731
    assert derivative(f, y) == pytest.approx(d.first, abs=1e-6)
    assert second_derivative(f, y) == pytest.approx(d.second, abs=1e-3)
    for m in (1.0, 2.0, 12.0):
        e = ex14_discrete(times, amounts, y, m)
        g = lambda s: ex14_discrete(times, amounts, s, m).value  # noqa: E731
        assert derivative(g, y) == pytest.approx(e.first, abs=1e-6)
        assert second_derivative(g, y) == pytest.approx(e.second, abs=1e-3)
    big = ex14_discrete(times, amounts, y, 1e7)
    assert big.value == pytest.approx(d.value, abs=1e-4)
    assert big.second == pytest.approx(d.second, abs=1e-3)


def test_ex15():
    np.testing.assert_array_equal(ex15_gradient(EX15_A), [5.0, -1.0, -3.0])
    assert ex15_f(EX15_A) == 3.0
    np.testing.assert_allclose(gradient(ex15_f, EX15_A), [5, -1, -3], atol=1e-8)
    np.testing.assert_allclose(hessian(ex15_f, EX15_A), ex15_hessian(), atol=1e-5)
    rng = np.random.default_rng(1)
    for x in rng.uniform(-10, 10, size=(20, 3)):
        assert ex15_taylor2(x) == pytest.approx(ex15_f(x), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("t", [0.1, 0.5, 2.0])
def test_ex16(t):
    for x in (-2.0, -0.3, 0.0, 0.8, 3.0):
        assert derivative(lambda s: ex16_u(x, s), t) == pytest.approx(ex16_u_t(x, t), abs=1e-7)
        assert second_derivative(lambda s: ex16_u(s, t), x) == pytest.approx(ex16_u_xx(x, t), abs=1e-5)
        assert derivative(lambda s: ex16_u(s, t), x) == pytest.approx(ex16_u_x(x, t), abs=1e-8)
        assert ex16_u_t(x, t) == ex16_u_xx(x, t)
    assert riemann_sum(lambda s: ex16_u(s, t), -40, 40, 20000) == pytest.approx(1.0, abs=1e-10)
