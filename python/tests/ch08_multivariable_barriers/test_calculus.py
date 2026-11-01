import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import derivative
from p4mfe.ch08_multivariable_barriers.chain_rule import (
    chain_rule_derivative,
    chain_rule_gradient,
    chain_rule_jacobian,
)
from p4mfe.ch08_multivariable_barriers.double_integrals import (
    change_of_variables_integral,
    double_integral,
    double_integral_region,
    from_polar,
    gaussian_disk_integral,
    gaussian_square_integral,
    jacobian_determinant,
    polar_integral,
)
from p4mfe.ch08_multivariable_barriers.extrema import (
    CriticalPointType,
    classify_critical_point,
    classify_hessian,
    determinant,
    find_critical_point,
    leading_principal_minors,
)


# 8.1 -----------------------------------------------------------------------
@pytest.mark.parametrize("t", [-0.5, 0.3, 1.1])
def test_chain_rule_curve(t):
    def f(v):
        return v[0] ** 2 * v[1] + math.exp(v[1])

    def g(s):
        return np.array([math.cos(s), s * s])

    exact = -2 * math.cos(t) * math.sin(t) * t * t + (math.cos(t) ** 2 + math.exp(t * t)) * 2 * t
    assert chain_rule_derivative(f, g, t) == pytest.approx(exact, abs=1e-7)
    assert chain_rule_derivative(f, g, t) == pytest.approx(derivative(lambda s: f(g(s)), t), abs=1e-7)


def test_chain_rule_gradient_jacobian():
    def f(v):
        return v[0] * v[1]

    def G(p):
        return np.array([p[0] * math.cos(p[1]), p[0] * math.sin(p[1])])

    x = [1.5, 0.4]
    grad = chain_rule_gradient(f, G, x)
    assert grad[0] == pytest.approx(x[0] * math.sin(2 * x[1]), abs=1e-7)
    assert grad[1] == pytest.approx(x[0] ** 2 * math.cos(2 * x[1]), abs=1e-7)
    J = chain_rule_jacobian(lambda v: np.array([math.hypot(v[0], v[1]), math.atan2(v[1], v[0])]), G, x)
    assert J == pytest.approx(np.eye(2), abs=1e-7)


# 8.2 -----------------------------------------------------------------------
def test_double_integrals():
    assert double_integral(lambda x, y: x * y * y, 0, 1, 0, 2) == pytest.approx(4 / 3, abs=1e-12)

    def f(x, y):
        return np.exp(-x * y) * np.cos(x + y)

    assert double_integral(f, 0, 1, 0, 2) == pytest.approx(double_integral(lambda y, x: f(x, y), 0, 2, 0, 1), abs=1e-12)
    assert double_integral_region(lambda x, y: np.ones_like(y), 0, 1, lambda x: 0.0, lambda x: x) == pytest.approx(0.5)
    assert double_integral_region(lambda x, y: y, 0, 1, lambda x: 0.0, lambda x: x) == pytest.approx(1 / 6)


# 8.3 -----------------------------------------------------------------------
def test_change_of_variables():
    assert polar_integral(lambda x, y: np.ones_like(y), 0, 1) == pytest.approx(math.pi, abs=1e-10)
    assert jacobian_determinant(from_polar, 2.0, 0.7) == pytest.approx(2.0, abs=1e-8)
    assert change_of_variables_integral(lambda x, y: np.ones_like(y), from_polar, 0, 1, 0, 2 * math.pi) == \
        pytest.approx(math.pi, abs=1e-8)

    def lin(u, v):
        return u + v, u - v

    exact = 2 * double_integral(lambda u, v: (u + v) ** 2, 0, 1, 0, 1)
    assert change_of_variables_integral(lambda x, y: x * x, lin, 0, 1, 0, 1) == pytest.approx(exact, abs=1e-8)
    assert polar_integral(lambda x, y: x * x + y * y, 1, 2) == pytest.approx(math.pi * 15 / 2, abs=1e-9)


# 8.6 -----------------------------------------------------------------------
@pytest.mark.parametrize("R", [0.5, 1.0, 3.0])
def test_normal_density_integral(R):
    disk = polar_integral(lambda x, y: np.exp(-0.5 * (x * x + y * y)), 0, R)
    assert disk == pytest.approx(gaussian_disk_integral(R), abs=1e-8)
    assert gaussian_disk_integral(R) < gaussian_square_integral(R) < gaussian_disk_integral(R * math.sqrt(2))


def test_normal_density_limit():
    assert gaussian_disk_integral(40) == pytest.approx(2 * math.pi, abs=1e-15)
    assert math.sqrt(gaussian_square_integral(12)) == pytest.approx(math.sqrt(2 * math.pi), abs=1e-12)


# 8.4 -----------------------------------------------------------------------
def test_extrema():
    def bowl(v):
        return (v[0] - 1) ** 2 + 2 * (v[1] + 0.5) ** 2 + v[0] * v[1]

    x = find_critical_point(bowl, [0.0, 0.0])
    assert x == pytest.approx([10 / 7, -6 / 7], abs=1e-7)
    assert classify_critical_point(bowl, x) is CriticalPointType.MINIMUM
    assert classify_critical_point(lambda v: -bowl(v), x) is CriticalPointType.MAXIMUM
    assert classify_critical_point(lambda v: v[0] ** 2 - v[1] ** 2, [0, 0]) is CriticalPointType.SADDLE
    assert classify_critical_point(lambda v: v[0] ** 4 + v[1] ** 2, [0, 0]) is CriticalPointType.DEGENERATE

    def cubic(v):
        return v[0] ** 3 - 3 * v[0] + v[1] ** 2

    m = find_critical_point(cubic, [2.0, 1.0])
    assert m[0] == pytest.approx(1.0, abs=1e-7)
    assert classify_critical_point(cubic, m) is CriticalPointType.MINIMUM
    s = find_critical_point(cubic, [-2.0, 1.0])
    assert s[0] == pytest.approx(-1.0, abs=1e-7)
    assert classify_critical_point(cubic, s) is CriticalPointType.SADDLE

    assert classify_hessian(np.diag([2, 3, -1])) is CriticalPointType.SADDLE
    assert classify_hessian(np.diag([-1, -2, -3])) is CriticalPointType.MAXIMUM
    assert determinant([[0, 1], [1, 0]]) == pytest.approx(-1.0)
    assert leading_principal_minors([[2, 1], [1, 2]]) == pytest.approx([2.0, 3.0])
