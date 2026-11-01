import math

import pytest

from p4mfe.normal import norm_cdf, norm_inv, norm_pdf


def test_pdf_at_zero():
    assert norm_pdf(0.0) == pytest.approx(0.3989422804014327, abs=1e-15)


def test_cdf_known_values():
    assert norm_cdf(0.0) == 0.5
    assert norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert norm_cdf(-1.0) == pytest.approx(0.15865525393145707, abs=1e-15)


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.5, 4.0])
def test_cdf_symmetry(x):
    assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-15)


def test_inverse_cdf():
    assert norm_inv(0.5) == 0.0
    assert norm_inv(0.975) == pytest.approx(1.959963984540054, abs=1e-14)
    assert norm_inv(0.995) == pytest.approx(2.5758293035489004, abs=1e-14)
    for p in [1e-12, 1e-6, 0.01, 0.02425, 0.2, 0.5, 0.7, 0.97575, 0.999, 1 - 1e-9]:
        x = norm_inv(p)
        assert norm_cdf(x) == pytest.approx(p, abs=1e-13 * min(p, 1 - p) + 2.3e-16)
        if p >= 1e-6:
            assert norm_inv(1 - p) == pytest.approx(-x, abs=1e-9)
    assert norm_inv(0.0) == -math.inf
    assert norm_inv(1.0) == math.inf
