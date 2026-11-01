import math

import numpy as np
import pytest

from p4mfe.ch06_taylor_bonds_atm.taylor import (
    partials_2d,
    taylor_linear,
    taylor_linear_2d,
    taylor_linear_nd,
    taylor_polynomial,
    taylor_quadratic,
    taylor_quadratic_2d,
    taylor_quadratic_nd,
    taylor_remainder_bound,
    taylor_remainder_integral,
)
from p4mfe.ch06_taylor_bonds_atm.taylor_series import (
    binomial_series,
    cos_series,
    exp_series,
    geometric_series,
    log1p_series,
    norm_cdf_series,
    sin_series,
)
from p4mfe.normal import norm_cdf


# 6.1 -----------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 4, 8])
@pytest.mark.parametrize("x", [-0.7, 0.3, 1.0])
def test_exp_polynomial_within_remainder_bound(n, x):
    err = abs(math.exp(x) - taylor_polynomial([1.0] * (n + 1), 0.0, x))
    assert err <= taylor_remainder_bound(math.exp(max(0.0, x)), 0.0, x, n) * (1 + 1e-12)


def test_polynomial_around_nonzero_point():
    d = [0.0, 1.0, -1.0, 2.0, -6.0]  # ln x at 1
    assert taylor_polynomial(d, 1.0, 1.1) == pytest.approx(0.1 - 0.01 / 2 + 0.001 / 3 - 0.0001 / 4, abs=1e-15)
    with pytest.raises(ValueError):
        taylor_polynomial([], 0.0, 1.0)


@pytest.mark.parametrize("a", [0.0, 0.5])
@pytest.mark.parametrize("x", [-0.8, 0.2, 1.5])
def test_integral_remainder_is_exact(a, x):
    d = [math.sin(a), math.cos(a), -math.sin(a), -math.cos(a)]
    R = taylor_remainder_integral(np.sin, a, x, 3)
    assert taylor_polynomial(d, a, x) + R == pytest.approx(math.sin(x), abs=1e-11)


def test_linear_and_quadratic_error_orders():
    f = lambda x: math.exp(x) * math.cos(x)  # noqa: E731
    a = 0.4
    e1 = lambda h: abs(f(a + h) - taylor_linear(f, a, a + h))  # noqa: E731
    e2 = lambda h: abs(f(a + h) - taylor_quadratic(f, a, a + h))  # noqa: E731
    assert math.log2(e1(0.02) / e1(0.01)) == pytest.approx(2.0, abs=0.1)
    assert math.log2(e2(0.02) / e2(0.01)) == pytest.approx(3.0, abs=0.1)


# 6.2 -----------------------------------------------------------------------
def test_multivariable_quadratic():
    f = lambda v: math.exp(v[0] - v[1]) + v[0] * v[1] * v[2] ** 2  # noqa: E731
    a = np.array([0.2, -0.1, 0.5])
    h = np.array([1e-2, -2e-2, 1.5e-2])
    e_lin = abs(f(a + h) - taylor_linear_nd(f, a, a + h))
    e_quad = abs(f(a + h) - taylor_quadratic_nd(f, a, a + h))
    e_quad2 = abs(f(a + h / 2) - taylor_quadratic_nd(f, a, a + h / 2))
    assert e_quad < e_lin / 20
    assert math.log2(e_quad / e_quad2) == pytest.approx(3.0, abs=0.3)
    q = lambda v: 1 + 2 * v[0] - v[1] + 3 * v[0] * v[1] + v[1] ** 2  # noqa: E731
    assert taylor_quadratic_nd(q, [0.0, 0.0], [0.7, -1.3]) == pytest.approx(q([0.7, -1.3]), abs=1e-6)


# 6.2.1 ---------------------------------------------------------------------
def test_two_variables():
    f = lambda x, y: math.exp(x) * math.sin(y)  # noqa: E731
    s = math.sin(math.pi / 4)
    d = partials_2d(f, 0.0, math.pi / 4)
    assert d.fx == pytest.approx(s, abs=1e-9) and d.fy == pytest.approx(s, abs=1e-9)
    assert d.fxx == pytest.approx(s, abs=1e-6) and d.fxy == pytest.approx(s, abs=1e-6)
    assert d.fyy == pytest.approx(-s, abs=1e-6)
    dx, dy = 0.05, -0.03
    exact = f(dx, math.pi / 4 + dy)
    assert abs(exact - taylor_quadratic_2d(d, dx, dy)) < 1e-4
    assert abs(exact - taylor_quadratic_2d(d, dx, dy)) < abs(exact - taylor_linear_2d(d, dx, dy)) / 10


# 6.3, 6.3.1 ----------------------------------------------------------------
@pytest.mark.parametrize("x", [-0.9, -0.3, 0.2, 0.75])
def test_classical_expansions(x):
    assert exp_series(x, 25) == pytest.approx(math.exp(x), abs=1e-15)
    assert log1p_series(x, 2000) == pytest.approx(math.log1p(x), abs=2e-12)
    assert geometric_series(x, 500) == pytest.approx(1 / (1 - x), abs=1e-12)
    assert binomial_series(x, 0.5, 500) == pytest.approx(math.sqrt(1 + x), abs=1e-12)
    assert binomial_series(x, -2.0, 800) == pytest.approx(1 / (1 + x) ** 2, abs=1e-9)
    assert sin_series(x, 12) == pytest.approx(math.sin(x), abs=1e-15)
    assert cos_series(x, 12) == pytest.approx(math.cos(x), abs=1e-15)


def test_series_edge_cases():
    assert exp_series(10.0, 60) / math.exp(10.0) == pytest.approx(1.0, abs=1e-14)
    assert sin_series(6.0, 30) == pytest.approx(math.sin(6.0), abs=1e-12)
    assert binomial_series(3.0, 3.0, 50) == 64.0
    assert log1p_series(1.0, 1000) == pytest.approx(math.log(2), abs=1 / 1001)
    with pytest.raises(ValueError):
        log1p_series(1.5, 10)


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.1, 1.0, 3.0])
def test_norm_cdf_series(x):
    assert norm_cdf_series(x, 60) == pytest.approx(norm_cdf(x), abs=1e-14)
