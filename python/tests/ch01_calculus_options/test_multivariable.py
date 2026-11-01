import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.multivariable import gradient, hessian


def f(v):  # x^2 y + e^(x y)
    return v[0] ** 2 * v[1] + math.exp(v[0] * v[1])


def test_gradient_two_variables():
    x, y = 0.5, -1.2
    g = gradient(f, [x, y])
    assert g == pytest.approx([2 * x * y + y * math.exp(x * y), x * x + x * math.exp(x * y)], abs=1e-8)


def test_hessian_two_variables():
    x, y = 0.5, -1.2
    e = math.exp(x * y)
    H = hessian(f, [x, y])
    expected = np.array([[2 * y + y * y * e, 2 * x + e + x * y * e], [2 * x + e + x * y * e, x * x * e]])
    assert H == pytest.approx(expected, abs=1e-6)
    assert H[0, 1] == H[1, 0]


def test_three_variables():
    g = gradient(lambda v: v[0] * v[1] * v[2] + v[2] ** 2, [1.0, 2.0, 3.0])
    assert g == pytest.approx([6.0, 3.0, 8.0], abs=1e-8)
