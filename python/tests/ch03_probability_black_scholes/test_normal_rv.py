import math

import numpy as np
import pytest

from p4mfe.ch03_probability_black_scholes.cdf_approx import norm_cdf_poly, norm_cdf_simpson
from p4mfe.ch03_probability_black_scholes.normal_rv import (
    NormalParams,
    linear_combination,
    log_price_distribution,
    lognormal_cdf,
    lognormal_mean,
    lognormal_pdf,
    lognormal_variance,
    norm_cdf,
    norm_pdf,
    normal_cdf,
    normal_pdf,
    prob_price_above,
)
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution


def test_normal_shift_scale():
    assert normal_pdf(3.0, 1.0, 2.0) == pytest.approx(
        norm_pdf(1.0) / 2, abs=1e-16
    )
    assert normal_cdf(1.0, 1.0, 2.0) == pytest.approx(0.5, abs=1e-16)
    assert normal_cdf(1 + 1.96 * 2, 1.0, 2.0) == pytest.approx(
        0.9750021048517795, abs=1e-13
    )


def test_linear_combination():
    combo = linear_combination(
        2.0, NormalParams(1.0, 4.0), 1.0, NormalParams(-2.0, 9.0), 0.5
    )
    assert combo.mean == 0.0
    assert combo.variance == pytest.approx(16 + 9 + 2 * 2 * 0.5 * 2 * 3)


def test_lognormal_moments_match_integration():
    mu, sigma = 0.1, 0.3
    lognormal = ContinuousDistribution(
        lambda y: lognormal_pdf(y, mu, sigma), 0.0, math.inf
    )
    assert lognormal.mean() == pytest.approx(
        lognormal_mean(mu, sigma), abs=1e-8
    )
    assert lognormal.variance() == pytest.approx(
        lognormal_variance(mu, sigma), abs=1e-8
    )
    assert lognormal_cdf(math.exp(mu), mu, sigma) == pytest.approx(
        0.5, abs=1e-15
    )


def test_price_model():
    spot0, maturity, sigma, mu, q = 50, 2, 0.25, 0.08, 0.01
    ln_spot_t = log_price_distribution(spot0, maturity, sigma, mu, q)
    assert lognormal_mean(ln_spot_t.mean, ln_spot_t.stddev) == pytest.approx(
        spot0 * math.exp((mu - q) * maturity), rel=1e-13
    )
    strike, r = 55, 0.04
    d2 = (math.log(spot0 / strike) + (r - q - 0.5 * sigma**2) * maturity) / (
        sigma * math.sqrt(maturity)
    )
    assert prob_price_above(
        strike, spot0, maturity, sigma, r, q
    ) == pytest.approx(norm_cdf(d2), abs=1e-14)


def test_cdf_poly_within_bound():
    for x in np.arange(-8.0, 8.0, 0.01):
        assert abs(norm_cdf_poly(x) - norm_cdf(x)) < 7.5e-8


@pytest.mark.parametrize(
    "x", [-6.0, -2.5, -1.0, -0.1, 0.0, 0.3, 1.0, 1.96, 4.0]
)
def test_cdf_simpson(x):
    assert norm_cdf_simpson(x) == pytest.approx(norm_cdf(x), abs=1e-12)
