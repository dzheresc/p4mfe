import math

import numpy as np
import pytest

from p4mfe.ch03_probability_black_scholes.normal_rv import lognormal_pdf, normal_pdf
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution, exponential_distribution
from p4mfe.ch04_lognormal_risk_neutral.change_of_variables import (
    exp_transform_pdf,
    linear_transform_pdf,
    square_transform_pdf,
    transformed_pdf,
    transformed_pdf_numeric,
)


def phi(x):
    return normal_pdf(x, 0.0, 1.0)


def test_linear_of_standard_normal_is_normal():
    a, b = -2.0, 1.5
    f = linear_transform_pdf(phi, a, b)
    for y in [-4.0, -1.0, 0.0, 1.5, 3.0, 6.0]:
        assert f(y) == pytest.approx(normal_pdf(y, b, 2.0), abs=1e-16)


def test_general_formula_matches_linear():
    a, b = 3.0, -1.0
    f1 = transformed_pdf(
        phi, lambda y: (y - b) / a, lambda y: np.full_like(y, 1 / a)
    )
    f2 = transformed_pdf_numeric(phi, lambda y: (y - b) / a)
    for y in [-5.0, -1.0, 0.0, 2.0]:
        assert f1(y) == pytest.approx(normal_pdf(y, b, a), abs=1e-16)
        assert f2(y) == pytest.approx(normal_pdf(y, b, a), abs=1e-10)


def test_exp_of_normal_is_lognormal():
    mu, sigma = 0.2, 0.4
    fY = exp_transform_pdf(lambda x: normal_pdf(x, mu, sigma))
    ys = np.array([0.1, 0.5, 1.0, 1.2, 2.0, 5.0])
    np.testing.assert_allclose(fY(ys), lognormal_pdf(ys, mu, sigma), rtol=1e-14)
    assert fY(0.0) == 0.0
    assert fY(-1.0) == 0.0


def test_sqrt_of_exponential_is_rayleigh():
    lam = 1.5
    X = exponential_distribution(lam)
    f = transformed_pdf_numeric(X.pdf, lambda y: y * y, lambda y: y >= 0)
    for y in [0.1, 0.5, 1.0, 2.0]:
        assert f(y) == pytest.approx(
            2 * lam * y * math.exp(-lam * y * y), abs=1e-9
        )
    Y = ContinuousDistribution(f, 0.0, math.inf)
    assert Y.cdf(50.0) == pytest.approx(1.0, abs=1e-8)
    assert Y.mean() == pytest.approx(0.5 * math.sqrt(math.pi / lam), abs=1e-8)


def test_square_of_standard_normal_is_chi_squared():
    f = square_transform_pdf(phi)
    for y in [0.01, 0.5, 1.0, 2.0, 7.0]:
        assert f(y) == pytest.approx(
            math.exp(-0.5 * y) / math.sqrt(2 * math.pi * y), abs=1e-15
        )
    assert f(-1.0) == 0.0
    assert ContinuousDistribution(f, 0.0, math.inf).mean(1e-8) == pytest.approx(
        1.0, abs=1e-6
    )
