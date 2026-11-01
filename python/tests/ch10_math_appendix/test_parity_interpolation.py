import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.integration import integrate
from p4mfe.ch10_math_appendix.interpolation import (
    divided_differences,
    interpolating_polynomial,
    interpolation_error_bound,
    lagrange_basis,
    lagrange_interpolate,
    newton_interpolate,
    polynomial_value,
)
from p4mfe.ch10_math_appendix.parity import (
    Parity,
    even_part,
    is_even,
    is_odd,
    odd_part,
    parity,
    symmetric_integral,
)
from p4mfe.normal import norm_cdf, norm_pdf

PTS = [0.1, 0.5, 1.0, 1.7, 2.5, 4.0]


# 10.1 ----------------------------------------------------------------------
def test_classifies_classical_functions():
    assert parity(math.cos, PTS) is Parity.EVEN
    assert parity(math.sin, PTS) is Parity.ODD
    assert parity(math.exp, PTS) is Parity.NEITHER
    assert parity(norm_pdf, PTS) is Parity.EVEN
    assert is_odd(lambda x: norm_cdf(x) - 0.5, PTS, 1e-14)


def test_products_and_derivatives():
    assert is_even(lambda x: math.sin(x) * x**3, PTS)
    assert is_odd(lambda x: math.cos(x) * x**3, PTS)
    assert is_odd(lambda x: 4 * x**3 - math.sin(x), PTS)  # (x^4 + cos x)'


def test_even_odd_decomposition():
    for x in PTS:
        assert even_part(math.exp, x) == pytest.approx(math.cosh(x), rel=1e-13)
        assert odd_part(math.exp, x) == pytest.approx(math.sinh(x), rel=1e-13)
    assert is_even(lambda x: even_part(math.exp, x), PTS)
    assert is_odd(lambda x: odd_part(math.exp, x), PTS)


def test_symmetric_integrals():
    f = lambda x: np.exp(x) + x**3 * np.cos(x)
    assert symmetric_integral(f, 1.5) == pytest.approx(2 * math.sinh(1.5), abs=1e-10)
    assert symmetric_integral(f, 1.5) == pytest.approx(integrate(f, -1.5, 1.5, 400), abs=1e-10)
    assert symmetric_integral(np.sin, 3.0) == pytest.approx(0.0, abs=1e-15)
    pdf = lambda x: np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    assert symmetric_integral(pdf, 2.0) == pytest.approx(2 * norm_cdf(2.0) - 1, abs=1e-10)
    with pytest.raises(ValueError):
        symmetric_integral(f, -1.0)


# 10.2 ----------------------------------------------------------------------
def test_reproduces_polynomials_exactly():
    a = [1.0, -2.0, 0.5, 3.0]
    xs = [-1.0, 0.0, 0.7, 2.0]
    ys = [polynomial_value(a, x) for x in xs]
    c = divided_differences(xs, ys)
    np.testing.assert_allclose(interpolating_polynomial(xs, ys), a, atol=1e-12)
    assert c[-1] == pytest.approx(3.0, abs=1e-12)
    for x in [-2.0, 0.3, 1.1, 5.0]:
        assert lagrange_interpolate(xs, ys, x) == pytest.approx(polynomial_value(a, x), abs=1e-10)
        assert newton_interpolate(xs, c, x) == pytest.approx(polynomial_value(a, x), abs=1e-10)


def test_lagrange_basis_is_kronecker_delta():
    xs = [0.0, 1.0, 3.0, 4.5]
    for i in range(len(xs)):
        for j, xj in enumerate(xs):
            assert lagrange_basis(xs, i, xj) == pytest.approx(1.0 if i == j else 0.0, abs=1e-15)
    assert sum(lagrange_basis(xs, i, 2.2) for i in range(len(xs))) == pytest.approx(1.0, abs=1e-14)


def test_error_within_bound():
    xs = [i * math.pi / 8 for i in range(5)]
    ys = [math.sin(x) for x in xs]
    for x in [0.1, 0.5, 1.0, 1.5]:
        assert abs(math.sin(x) - lagrange_interpolate(xs, ys, x)) <= interpolation_error_bound(1.0, xs, x)


def test_rejects_bad_nodes():
    with pytest.raises(ValueError):
        lagrange_interpolate([0.0, 1.0, 1.0], [1.0, 2.0, 3.0], 0.5)
    with pytest.raises(ValueError):
        divided_differences([0.0, 1.0], [1.0, 2.0, 3.0])
