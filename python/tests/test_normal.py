import pytest

from p4mfe.normal import norm_cdf, norm_pdf


def test_pdf_at_zero():
    assert norm_pdf(0.0) == pytest.approx(0.3989422804014327, abs=1e-15)


def test_cdf_known_values():
    assert norm_cdf(0.0) == 0.5
    assert norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert norm_cdf(-1.0) == pytest.approx(0.15865525393145707, abs=1e-15)


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.5, 4.0])
def test_cdf_symmetry(x):
    assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-15)
